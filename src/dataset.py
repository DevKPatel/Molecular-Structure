"""
Dataset and DataLoader utilities for multimodal spectroscopic structure elucidation.

Reads from the parquet metadata file produced by Phase A (A13), lazy-loads
spectral arrays from the original chunk parquet files, applies:
  - rare functional group oversampling
  - modality-dropout augmentation
  - spectral data augmentation (noise, peak shifts)
  - spectral tokenization matching each encoder's expected format
"""

import functools
import json
import ast
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from .config import SpectroConfig


# ─── Multiplicity index map ───────────────────────────────────────────────────

MULT_MAP: dict[str, int] = {
    "s": 1, "d": 2, "t": 3, "q": 4,
    "m": 5, "dd": 6, "dt": 7, "ddd": 8,
    "td": 9, "qd": 10, "tt": 11,
}
MULT_OTHER = 12


# ─── Peak list parser ─────────────────────────────────────────────────────────

def _parse_peak_list(raw: Any) -> list[dict]:
    """Parse a peak list that may be stored as a list or a stringified list."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            return ast.literal_eval(raw)
        except Exception:
            return []
    return []


# ─── Chunk cache ─────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=32)
def _read_chunk_cached(path_str: str) -> pd.DataFrame:
    """
    Cache up to 32 chunk DataFrames in RAM (~3-4 GB total).
    Avoids re-reading the same parquet file on every __getitem__ call.
    """
    return pd.read_parquet(path_str)


# ─── Spectral tokenization helpers ───────────────────────────────────────────
# All tokenize_* functions return plain np.ndarray pairs (tokens, mask).
# Conversion to Tensor is done once in __getitem__, keeping the interface clean.

def tokenize_h_nmr(
    peaks: list[dict], max_peaks: int, augment: bool
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        tokens: (max_peaks, 5) float32 — [ppm_start, ppm_end, mult_idx, nH, integration]
        mask:   (max_peaks,) bool      — True where padding
    """
    tokens = np.zeros((max_peaks, 5), dtype=np.float32)
    mask   = np.ones(max_peaks, dtype=bool)

    for i, pk in enumerate(peaks[:max_peaks]):
        shift = np.random.uniform(-0.1, 0.1) if augment else 0.0
        start = float(
            pk.get("start",
            pk.get("ppm_start",
            pk.get("rangeMin", pk.get("centroid", 0.0))))
        ) + shift
        end = float(
            pk.get("end",
            pk.get("ppm_end",
            pk.get("rangeMax", start)))
        ) + shift
        mult = MULT_MAP.get(
            str(pk.get("multiplicity", pk.get("category", "m"))).lower(),
            MULT_OTHER
        )
        nH = float(pk.get("n_hydrogen", pk.get("nH", 1.0)))
        intgr = float(pk.get("integration", 1.0))

        if augment and intgr < 0.1 and random.random() < 0.20:
            continue  # simulate detection threshold dropout

        tokens[i] = [start, end, mult, nH, intgr]
        mask[i]   = False

    return tokens, mask


def tokenize_c_nmr(
    peaks: list[dict], max_peaks: int, augment: bool
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        tokens: (max_peaks, 2) float32 — [ppm_position, intensity]
        mask:   (max_peaks,) bool
    """
    tokens = np.zeros((max_peaks, 2), dtype=np.float32)
    mask   = np.ones(max_peaks, dtype=bool)
    shift  = np.random.uniform(-2.0, 2.0) if augment else 0.0

    for i, pk in enumerate(peaks[:max_peaks]):
        ppm       = float(pk.get("centroid", pk.get("ppm", 0.0))) + shift
        intensity = float(pk.get("intensity", 1.0))
        tokens[i] = [ppm, intensity]
        mask[i]   = False

    return tokens, mask


def tokenize_hsqc(
    peaks: list[dict], max_peaks: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        tokens: (max_peaks, 3) float32 — [x_ppm, y_ppm, integration]
        mask:   (max_peaks,) bool
    """
    tokens = np.zeros((max_peaks, 3), dtype=np.float32)
    mask   = np.ones(max_peaks, dtype=bool)

    for i, pk in enumerate(peaks[:max_peaks]):
        x           = float(pk.get("x_ppm", pk.get("x", 0.0)))
        y           = float(pk.get("y_ppm", pk.get("y", 0.0)))
        integration = float(pk.get("integration", 1.0))
        tokens[i]   = [x, y, integration]
        mask[i]     = False

    return tokens, mask


def tokenize_ms(
    peaks_by_mode: list[list[dict]], max_peaks: int, augment: bool
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        tokens: (3, max_peaks, 2) float32 — [m/z, intensity] per energy mode
        mask:   (3, max_peaks) bool
    Note: caller passes either the 3 positive or 3 negative modes; hence dim-0 = 3.
    """
    n_modes = len(peaks_by_mode)
    tokens  = np.zeros((n_modes, max_peaks, 2), dtype=np.float32)
    mask    = np.ones((n_modes, max_peaks), dtype=bool)

    for m, peaks in enumerate(peaks_by_mode):
        i_valid = 0  # ensures no gaps when skipping peaks

        for pk in peaks:
            if i_valid >= max_peaks:
                break

            # ── Parse peak ─────────────────────────────────────────────
            if isinstance(pk, (list, tuple, np.ndarray)):
                # Your dataset format: [mz, intensity]
                mz  = float(pk[0])
                rel = float(pk[1])
            else:
                # Fallback for dict-based datasets
                mz = float(
                    pk.get("mz",
                    pk.get("m/z",
                    pk.get("mass",
                    pk.get("centroid", 0.0))))
                )
                rel = float(
                    pk.get("intensity",
                    pk.get("relative_intensity",
                    pk.get("abundance",
                    pk.get("height", 1.0))))
                )

            # ── Augmentation ───────────────────────────────────────────
            if augment:
                if rel < 5.0 and random.random() < 0.50:
                    continue  # skip low-intensity peaks
                mz += np.random.uniform(-0.01, 0.01)  # simulate mass noise

            # ── Store token ────────────────────────────────────────────
            tokens[m, i_valid] = [mz, rel]
            mask[m, i_valid]   = False
            i_valid += 1

    return tokens, mask


def augment_ir(ir: np.ndarray) -> np.ndarray:
    """Gaussian noise (SNR 40–60 dB) + random polynomial baseline drift."""
    snr_db       = np.random.uniform(40, 60)
    signal_power = np.mean(ir ** 2) + 1e-8
    noise_std    = np.sqrt(signal_power / (10 ** (snr_db / 10)))
    ir           = ir + np.random.normal(0, noise_std, ir.shape)

    degree = np.random.randint(1, 4)
    x      = np.linspace(0, 1, len(ir))
    coeffs = np.random.uniform(-0.02, 0.02, degree + 1)
    ir     = ir + np.polyval(coeffs, x)

    return np.clip(ir, 0.0, None).astype(np.float32)


# ─── Dataset ──────────────────────────────────────────────────────────────────

class SpectroDataset(Dataset):
    """
    Lazy-loading dataset over the scaffold-split metadata parquet.

    __getitem__ always returns ALL keys regardless of modality availability.
    Missing modalities are represented by zero tensors with all-True masks.
    The 'available' tensor communicates to the model which modalities are real.

    Keys returned:
        ir:           (1800,)           float32  — zeros if unavailable
        h_nmr:        (nmr_max_peaks, 5)  float32
        h_nmr_mask:   (nmr_max_peaks,)    bool    — all True if unavailable
        c_nmr:        (nmr_max_peaks, 2)  float32
        c_nmr_mask:   (nmr_max_peaks,)    bool
        hsqc:         (hsqc_max_peaks, 3) float32
        hsqc_mask:    (hsqc_max_peaks,)   bool
        ms_pos:       (3, ms_max_peaks, 2) float32
        ms_pos_mask:  (3, ms_max_peaks)    bool
        ms_neg:       (3, ms_max_peaks, 2) float32
        ms_neg_mask:  (3, ms_max_peaks)    bool
        available:    (6,)              bool     — always present
        selfies_ids:  (max_decoder_len+1,) int64 — always present
    """

    def __init__(
        self,
        metadata_path: Path,
        data_root: Path,
        vocab_path: Path,
        cfg: SpectroConfig,
        split: str = "train",
        augment: bool = True,
        modality_dropout: bool = True,
    ) -> None:
        self.cfg              = cfg
        self.data_root        = data_root
        self.augment          = augment
        self.modality_dropout = modality_dropout and (split == "train")

        with open(vocab_path) as f:
            vocab_data = json.load(f)
        self.token2idx: dict[str, int] = vocab_data["token2idx"]

        df       = pd.read_parquet(metadata_path)
        self.df  = df[df["split"] == split].reset_index(drop=True)

        self.weights: np.ndarray | None = None
        if split == "train":
            self.weights = self._compute_sample_weights()

    def _compute_sample_weights(self) -> np.ndarray:
        weights = np.ones(len(self.df), dtype=np.float32)
        if "functional_groups" not in self.df.columns:
            return weights
        rare = set(self.cfg.rare_fg_groups)
        for idx, row in self.df.iterrows():
            fgs      = _parse_peak_list(row.get("functional_groups", []))
            fg_names = {fg if isinstance(fg, str) else fg.get("name", "") for fg in fgs}
            if fg_names & rare:
                weights[idx] = self.cfg.oversample_factor
        return weights

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row    = self.df.iloc[idx]
        augment = self.augment
        cfg    = self.cfg

        # ── Load dense spectral arrays from chunk file ────────────────────────
        # chunk_file is stored as an absolute path by Phase A
        chunk_path    = Path(row["chunk_file"])
        chunk_row_idx = int(row["chunk_row_idx"])
        spectral      = self._load_chunk_row(chunk_path, chunk_row_idx)

        # ── Modality availability (with optional dropout) ─────────────────────
        available = np.ones(6, dtype=bool)
        if self.modality_dropout:
            drop = np.random.random(6) < cfg.modality_dropout_p
            if drop[1] and drop[2]:   # never drop both NMR modalities simultaneously
                drop[1] = False
            available = ~drop

        item: dict = {}

        # ── IR  (modality index 0) ────────────────────────────────────────────
        if available[0]:
            raw_ir = spectral.get("ir_spectra", None)
            if raw_ir is not None:
                # FIX (Issue 7): .copy() prevents non-writable numpy array warning
                ir = np.array(raw_ir, dtype=np.float32).copy()
                if augment:
                    ir = augment_ir(ir)
                item["ir"] = torch.from_numpy(ir)
            else:
                available[0] = False  # chunk had no IR column → treat as missing

        # ── 1H-NMR  (modality index 1) ───────────────────────────────────────
        if available[1]:
            h_peaks        = _parse_peak_list(row.get("h_nmr_peaks", []))
            tokens, mask   = tokenize_h_nmr(h_peaks, cfg.nmr_max_peaks, augment)
            item["h_nmr"]       = torch.from_numpy(tokens)
            item["h_nmr_mask"]  = torch.from_numpy(mask)

        # ── 13C-NMR  (modality index 2) ──────────────────────────────────────
        if available[2]:
            c_peaks        = _parse_peak_list(row.get("c_nmr_peaks", []))
            tokens, mask   = tokenize_c_nmr(c_peaks, cfg.nmr_max_peaks, augment)
            item["c_nmr"]       = torch.from_numpy(tokens)
            item["c_nmr_mask"]  = torch.from_numpy(mask)

        # ── HSQC  (modality index 3) ──────────────────────────────────────────
        if available[3]:
            hsqc_peaks = _parse_peak_list(row.get("hsqc_nmr_peaks", []))
            if len(hsqc_peaks) == 0:
                # 24 acyclic molecules have no H-C bonds → genuinely no HSQC peaks
                available[3] = False
            else:
                tokens, mask      = tokenize_hsqc(hsqc_peaks, cfg.hsqc_max_peaks)
                item["hsqc"]      = torch.from_numpy(tokens)
                item["hsqc_mask"] = torch.from_numpy(mask)

        # ── MS/MS positive  (modality index 4) ───────────────────────────────
        if available[4]:
            pos_modes = [
                _parse_peak_list(row.get(k, []))
                for k in ["msms_cfmid_positive_10ev", "msms_cfmid_positive_20ev", "msms_cfmid_positive_40ev"]
            ]
            tokens, mask          = tokenize_ms(pos_modes, cfg.ms_max_peaks_per_mode, augment)
            item["ms_pos"]        = torch.from_numpy(tokens)   # (3, max_peaks, 2)
            item["ms_pos_mask"]   = torch.from_numpy(mask)     # (3, max_peaks)

        # ── MS/MS negative  (modality index 5) ───────────────────────────────
        if available[5]:
            neg_modes = [
                _parse_peak_list(row.get(k, []))
                for k in ["msms_cfmid_negative_10ev", "msms_cfmid_negative_20ev", "msms_cfmid_negative_40ev"]
            ]
            tokens, mask          = tokenize_ms(neg_modes, cfg.ms_max_peaks_per_mode, augment)
            item["ms_neg"]        = torch.from_numpy(tokens)
            item["ms_neg_mask"]   = torch.from_numpy(mask)

        # FIX (Issue 1): Always emit all keys with zero/mask fallbacks so that
        # spectro_collate always sees a consistent set of keys across every sample
        # in a batch. Without this, batches where only some samples have HSQC (or
        # any other modality) produce tensors with fewer rows than batch size,
        # causing shape mismatches in model.encode().
        if "ir" not in item:
            item["ir"] = torch.zeros(cfg.ir_spectrum_len, dtype=torch.float32)

        if "h_nmr" not in item:
            item["h_nmr"]      = torch.zeros(cfg.nmr_max_peaks, 5, dtype=torch.float32)
            item["h_nmr_mask"] = torch.ones(cfg.nmr_max_peaks, dtype=torch.bool)

        if "c_nmr" not in item:
            item["c_nmr"]      = torch.zeros(cfg.nmr_max_peaks, 2, dtype=torch.float32)
            item["c_nmr_mask"] = torch.ones(cfg.nmr_max_peaks, dtype=torch.bool)

        if "hsqc" not in item:
            item["hsqc"]      = torch.zeros(cfg.hsqc_max_peaks, 3, dtype=torch.float32)
            item["hsqc_mask"] = torch.ones(cfg.hsqc_max_peaks, dtype=torch.bool)

        if "ms_pos" not in item:
            item["ms_pos"]      = torch.zeros(3, cfg.ms_max_peaks_per_mode, 2, dtype=torch.float32)
            item["ms_pos_mask"] = torch.ones(3, cfg.ms_max_peaks_per_mode, dtype=torch.bool)

        if "ms_neg" not in item:
            item["ms_neg"]      = torch.zeros(3, cfg.ms_max_peaks_per_mode, 2, dtype=torch.float32)
            item["ms_neg_mask"] = torch.ones(3, cfg.ms_max_peaks_per_mode, dtype=torch.bool)

        item["available"] = torch.from_numpy(available)

        # ── SELFIES target ────────────────────────────────────────────────────
        import selfies as sf
        selfies_str = str(row["selfies"])
        toks = list(sf.split_selfies(selfies_str))
        ids  = (
            [cfg.bos_idx]
            + [self.token2idx.get(t, cfg.unk_idx) for t in toks]
            + [cfg.eos_idx]
        )
        max_len = cfg.max_decoder_len + 1
        ids     = ids[:max_len]
        ids    += [cfg.pad_idx] * (max_len - len(ids))
        item["selfies_ids"] = torch.tensor(ids, dtype=torch.long)

        return item

    def _load_chunk_row(self, chunk_path: Path, row_idx: int) -> dict:
        """Load one row of dense spectral arrays from a parquet chunk file."""
        try:
            df = _read_chunk_cached(str(chunk_path))
            return df.iloc[row_idx].to_dict()
        except Exception:
            return {}


# ─── Collate function ─────────────────────────────────────────────────────────

def spectro_collate(batch: list[dict]) -> dict:
    """
    Stacks all tensors in the batch. Because __getitem__ now always emits every
    key (with zero/mask fallbacks for missing modalities), every sample in the
    batch has the same keys and the same fixed shapes — no padding is needed
    beyond what the fixed max_peaks sizes already provide.
    """
    out: dict = {}
    out["selfies_ids"] = torch.stack([b["selfies_ids"] for b in batch])
    out["available"]   = torch.stack([b["available"]   for b in batch])

    for key in ["ir"]:
        out[key] = torch.stack([b[key] for b in batch])

    for key in [
        "h_nmr",  "h_nmr_mask",
        "c_nmr",  "c_nmr_mask",
        "hsqc",   "hsqc_mask",
        "ms_pos", "ms_pos_mask",
        "ms_neg", "ms_neg_mask",
    ]:
        out[key] = torch.stack([b[key] for b in batch])

    # Cast masks back to bool after stacking (torch.stack preserves dtype,
    # but an explicit cast is defensive against any float creep).
    for mask_key in ["h_nmr_mask", "c_nmr_mask", "hsqc_mask", "ms_pos_mask", "ms_neg_mask"]:
        out[mask_key] = out[mask_key].bool()

    return out


# ─── DataLoader factory ───────────────────────────────────────────────────────

def make_dataloader(
    cfg: SpectroConfig,
    split: str,
    physical_batch_size: int,
    num_workers: int = 4,
) -> DataLoader:
    augment = (split == "train")

    ds = SpectroDataset(
        metadata_path=cfg.metadata_path,
        data_root=cfg.data_root,
        vocab_path=cfg.vocab_path,
        cfg=cfg,
        split=split,
        augment=augment,
        modality_dropout=(split == "train"),
    )

    sampler = None
    shuffle = False

    if split == "train" and ds.weights is not None:
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(ds.weights),
            num_samples=len(ds),
            replacement=True,
        )
    else:
        shuffle = (split == "train")

    return DataLoader(
        ds,
        batch_size=physical_batch_size,
        sampler=sampler,
        shuffle=(shuffle if sampler is None else False),
        num_workers=num_workers,
        collate_fn=spectro_collate,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )