"""
Stage 1 training loop: pretrain on simulated spectral data.

Implements:
  - Noam LR schedule (warmup + inverse-sqrt decay)
  - Gradient accumulation for effective batch size = 4096 tokens
  - bf16 autocast throughout (no GradScaler — bf16 does not need gradient scaling)
  - torch.compile for graph optimization
  - Early stopping on validation Top-1 accuracy (patience=20k steps)
  - Checkpoint saving (best 5 by Top-1)
  - Full wandb metric logging per Section 7.1
"""

import heapq
import json
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np          # FIX (Issue 4): moved to top-level so evaluate() can use it
import selfies as sf
import torch
import torch.nn as nn
import wandb
from rdkit import Chem
from torch import Tensor
# FIX (Issue 5): GradScaler import removed — bf16 does not require gradient scaling.
# GradScaler is designed for float16's limited dynamic range. bfloat16 has the
# same dynamic range as float32 and does not suffer from gradient underflow.
# Using GradScaler with bf16 adds overhead and can cause incorrect scale updates.
from tqdm import tqdm

from .config import SpectroConfig
from .dataset import make_dataloader
from .model import SpectroModel

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
# ─── Noam LR schedule ─────────────────────────────────────────────────────────

class NoamScheduler(torch.optim.lr_scheduler.LambdaLR):
    """
    lr(step) = d_model^{-0.5} * min(step^{-0.5}, step * warmup^{-1.5})

    The multiplicative factor is computed against a base_lr of 1.0 in the
    optimizer; the Noam formula encodes the absolute scale.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, d_model: int, warmup_steps: int) -> None:
        self.d_model = d_model
        self.warmup  = warmup_steps
        # We pass lr=1.0 to Adam and let the schedule encode the absolute scale
        super().__init__(optimizer, lr_lambda=self._noam)

    def _noam(self, step: int) -> float:
        step = max(step, 1)
        return self.d_model ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5))


# ─── Checkpoint manager (keep best N by Top-1) ────────────────────────────────

@dataclass
class CheckpointEntry:
    top1: float
    step: int
    path: Path

    def __lt__(self, other: "CheckpointEntry") -> bool:
        return self.top1 < other.top1   # min-heap on top1


class CheckpointManager:
    def __init__(self, ckpt_dir: Path, keep: int = 5) -> None:
        self.ckpt_dir = ckpt_dir
        self.keep = keep
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._heap: list[CheckpointEntry] = []   # min-heap

    def save(self, model: SpectroModel, optimizer, scheduler, step: int, top1: float) -> None:
        path = self.ckpt_dir / f"step_{step:07d}_top1_{top1:.4f}.pt"
        torch.save(
            {
                "step": step,
                "top1": top1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            path,
        )

        entry = CheckpointEntry(top1=top1, step=step, path=path)
        heapq.heappush(self._heap, entry)

        # Evict worst checkpoint if over limit
        if len(self._heap) > self.keep:
            worst = heapq.heappop(self._heap)
            worst.path.unlink(missing_ok=True)

    def best_path(self) -> Path | None:
        if not self._heap:
            return None
        return max(self._heap, key=lambda e: e.top1).path


# ─── Metrics helpers ──────────────────────────────────────────────────────────

def _selfies_ids_to_smiles(ids: list[int], idx2token: dict[int, str], cfg: SpectroConfig) -> str | None:
    """Convert decoded token-id sequence → canonical SMILES via SELFIES → RDKit."""
    tokens = []
    for i in ids:
        if i in (cfg.bos_idx, cfg.pad_idx):
            continue
        if i == cfg.eos_idx:
            break
        tokens.append(idx2token.get(i, ""))
    selfies_str = "".join(tokens)
    try:
        smiles = sf.decoder(selfies_str)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


@torch.no_grad()
def evaluate(
    model: SpectroModel,
    val_loader,
    cfg: SpectroConfig,
    idx2token: dict[int, str],
    max_batches: int = 200,
    device: torch.device = torch.device("cuda"),
) -> dict[str, float]:
    """
    Runs greedy decoding (beam_size=1 for speed) on up to max_batches.
    Returns top1_acc, validity_rate, mean_tanimoto.
    """
    from rdkit.Chem import DataStructs
    from rdkit.Chem import AllChem

    model.eval()
    total = correct = valid = 0
    tanimoto_scores: list[float] = []

    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break

        # Move to device
        batch_dev = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}
        true_ids = batch_dev.pop("selfies_ids")  # (B, max_len+1)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            beam_results = model.beam_search(batch_dev, beam_size=1)

        B = true_ids.size(0)
        for b in range(B):
            total += 1
            pred_ids = beam_results[b][0] if beam_results[b] else []
            pred_smi = _selfies_ids_to_smiles(pred_ids, idx2token, cfg)
            true_smi = _selfies_ids_to_smiles(true_ids[b].tolist(), idx2token, cfg)

            if pred_smi is not None:
                valid += 1
            if pred_smi is not None and true_smi is not None:
                if pred_smi == true_smi:
                    correct += 1
                # Tanimoto
                try:
                    fp_pred = AllChem.GetMorganFingerprintAsBitVect(
                        Chem.MolFromSmiles(pred_smi), 2, 1024
                    )
                    fp_true = AllChem.GetMorganFingerprintAsBitVect(
                        Chem.MolFromSmiles(true_smi), 2, 1024
                    )
                    tanimoto_scores.append(DataStructs.TanimotoSimilarity(fp_pred, fp_true))
                except Exception:
                    pass

    model.train()
    return {
        "top1_acc":       correct / max(total, 1),
        "validity_rate":  valid   / max(total, 1),
        "mean_tanimoto":  float(np.mean(tanimoto_scores)) if tanimoto_scores else 0.0,
    }


# ─── Main training loop ───────────────────────────────────────────────────────

def train(
    cfg: SpectroConfig,
    physical_batch_size: int = 512,
    accumulation_steps: int = 8,    # effective batch = 512 * 8 = 4096
    device_str: str = "cuda",
    num_workers: int = 4,
    compile_model: bool = True,
) -> None:

    device = torch.device(device_str)
    print(f"--- Debug: Using device: {device} ---")
    if device.type == "cuda":
        print(f"--- Debug: GPU Name: {torch.cuda.get_device_name(0)} ---")
        print(f"--- Debug: Initial VRAM Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB ---")
    # ── Dataloaders ──────────────────────────────────────────────────────────
    train_loader = make_dataloader(cfg, "train", physical_batch_size, num_workers)
    val_loader   = make_dataloader(cfg, "val",   physical_batch_size, num_workers)

    # Load vocab for evaluation
    with open(cfg.vocab_path) as f:
        vocab_data = json.load(f)
    idx2token = {int(k): v for k, v in vocab_data["idx2token"].items()}

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SpectroModel(cfg).to(device)
    if compile_model and hasattr(torch, "compile"):
        print("--- Debug: Compiling model (this takes high RAM and time)... ---")
        model = torch.compile(model)
    else:
        print("--- Debug: Skipping torch.compile ---")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    # Pass lr=1.0; Noam formula encodes the absolute scale
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0,
        betas=(0.9, 0.998),
        weight_decay=cfg.weight_decay,
    )
    scheduler = NoamScheduler(optimizer, d_model=cfg.d_model, warmup_steps=cfg.warmup_steps)
    # FIX (Issue 5): GradScaler removed. bf16 (bfloat16) has the same exponent
    # range as float32 so gradients never underflow — scaling is unnecessary and
    # harmful. The old code created a scaler but bf16 made it a near-no-op anyway;
    # removing it makes the intent explicit and avoids subtle interactions.

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(
        ignore_index=cfg.pad_idx,
        label_smoothing=cfg.label_smoothing,
    )

    # ── Checkpoint manager ────────────────────────────────────────────────────
    ckpt_mgr = CheckpointManager(cfg.output_dir / "checkpoints", keep=5)

    # ── Early stopping state ──────────────────────────────────────────────────
    best_top1    = 0.0
    no_improve   = 0
    min_steps    = 100_000
    patience     = 20_000
    eval_every   = 2_000
    save_every   = 5_000
    log_every    = 500

    global_step  = 0
    optimizer.zero_grad()

    # ── wandb ─────────────────────────────────────────────────────────────────
    # assumed already initialized externally

    train_iter  = iter(train_loader)
    running_loss = deque(maxlen=log_every)

    print("Starting Stage 1 pretraining…")
    t0 = time.time()

    while global_step < cfg.max_steps:
        # ── Get next batch ────────────────────────────────────────────────────
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch_dev = {k: v.to(device, non_blocking=True) if isinstance(v, Tensor) else v
                     for k, v in batch.items()}
        if global_step == 0:
            sample_key = list(batch_dev.keys())[0]
            if isinstance(batch_dev[sample_key], Tensor):
                print(f"--- Debug: Tensor device check: {batch_dev[sample_key].device} ---")
                
        selfies_ids = batch_dev.pop("selfies_ids")   # (B, max_len+1)
        tgt_in  = selfies_ids[:, :-1]                # (B, max_len)   — decoder input
        tgt_out = selfies_ids[:, 1:]                 # (B, max_len)   — decoder target (shifted)

        # ── NMR-only curriculum: force 1H and 13C active, mask everything else ──
        # Remove this block entirely when advancing to Stage 2 (add MS/MS).
        _B = tgt_in.size(0)
        batch_dev["available"] = torch.zeros(_B, 6, dtype=torch.bool, device=device)
        batch_dev["available"][:, 1] = True   # 1H-NMR always present
        batch_dev["available"][:, 2] = True   # 13C-NMR always present

        # ── Forward + loss ────────────────────────────────────────────────────
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(batch_dev, tgt_in)        # (B, max_len, vocab_size)
            # Flatten for cross-entropy
            loss = criterion(
                logits.reshape(-1, cfg.vocab_size),  # (B*T, V)
                tgt_out.reshape(-1),                 # (B*T,)
            ) / accumulation_steps

        # FIX (Issue 5): plain backward — no scaler needed for bf16
        loss.backward()
        running_loss.append(loss.item() * accumulation_steps)

        # ── Gradient accumulation step ────────────────────────────────────────
        if (global_step + 1) % accumulation_steps == 0:
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        global_step += 1

        # ── Logging ───────────────────────────────────────────────────────────
        if global_step % log_every == 0:
            lr = scheduler.get_last_lr()[0]
            wandb.log({
                "train/loss":      np.mean(running_loss),
                "train/lr":        lr,
                "train/grad_norm": grad_norm.item() if isinstance(grad_norm, Tensor) else grad_norm,
                "train/step":      global_step,
                "train/elapsed_h": (time.time() - t0) / 3600,
            }, step=global_step)

        # ── Evaluation ────────────────────────────────────────────────────────
        if global_step % eval_every == 0:
            metrics = evaluate(model, val_loader, cfg, idx2token, max_batches=200, device=device)
            wandb.log({f"val/{k}": v for k, v in metrics.items()}, step=global_step)
            print(
                f"step={global_step:>7d}  "
                f"loss={np.mean(running_loss):.4f}  "
                f"top1={metrics['top1_acc']:.4f}  "
                f"validity={metrics['validity_rate']:.4f}  "
                f"tanimoto={metrics['mean_tanimoto']:.4f}"
            )

            # Early stopping check
            if metrics["top1_acc"] > best_top1:
                best_top1  = metrics["top1_acc"]
                no_improve = 0
            else:
                no_improve += eval_every

            if global_step >= min_steps and no_improve >= patience:
                print(f"Early stopping at step {global_step}: no improvement for {patience} steps.")
                break

        # ── Checkpoint ────────────────────────────────────────────────────────
        if global_step % save_every == 0:
            # Use cached metrics if recent, otherwise skip for speed
            top1 = best_top1
            ckpt_mgr.save(model, optimizer, scheduler, global_step, top1)

    # Final checkpoint restore
    best = ckpt_mgr.best_path()
    if best is not None:
        print(f"Restoring best checkpoint: {best}")
        ckpt = torch.load(best, map_location=device)
        model.load_state_dict(ckpt["model"])

    print(f"Stage 1 complete. Best Top-1: {best_top1:.4f}")
    wandb.log({"final/best_top1": best_top1})


# ─── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-batch", type=int, default=512)
    parser.add_argument("--accum-steps",    type=int, default=8)
    parser.add_argument("--device",         type=str, default="cuda")
    parser.add_argument("--workers",        type=int, default=4)
    parser.add_argument("--no-compile",     action="store_true")
    args = parser.parse_args()

    cfg = SpectroConfig()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    wandb.init(
        project="spectro-elucidation",
        name="stage1_nmr_only_test",
        config=cfg.__dict__,
    )

    train(
        cfg=cfg,
        physical_batch_size=args.physical_batch,
        accumulation_steps=args.accum_steps,
        device_str=args.device,
        num_workers=args.workers,
        compile_model=not args.no_compile,
    )