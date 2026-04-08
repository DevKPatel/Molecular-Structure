"""
Cross-modal attention fusion and full SpectroModel.

Architecture flow:
  [IR, 1H-NMR, 13C-NMR, HSQC, MS+, MS-]
       ↓ (modality-specific encoders)
  [enc_IR, enc_1H, enc_13C, enc_HSQC, enc_MS+, enc_MS-]
       ↓ (cross-modal attention fusion → 6 CLS tokens)
  fused_repr  (B, 6, d_model)
       ↓ (autoregressive SELFIES decoder)
  SELFIES token logits
"""

import torch
import torch.nn as nn
from torch import Tensor

from .config import SpectroConfig
from .encoders import IREncoder, NMREncoder, HSQCEncoder, MSMSEncoder


# ─── Modality index constants ─────────────────────────────────────────────────
MOD_IR   = 0
MOD_1H   = 1
MOD_13C  = 2
MOD_HSQC = 3
MOD_MSP  = 4   # MS/MS positive
MOD_MSN  = 5   # MS/MS negative


# ─── Cross-Modal Attention Fusion ─────────────────────────────────────────────

class CrossModalFusion(nn.Module):
    """
    One learnable CLS token per modality attends (cross-attention) across all
    other modalities' encoder outputs, then a 2-layer FFN refines each token.

    This compresses each modality's variable-length encoder output to a single
    d_model vector before passing to the decoder. The decoder then attends over
    just 6 vectors rather than 1,200+ tokens.

    Input:  list of 6 tensors, each (B, T_i, d_model)
            corresponding mask list, each (B, T_i) bool (True = pad)
    Output: (B, 6, d_model) — one refined CLS vector per modality
    """

    def __init__(self, cfg: SpectroConfig) -> None:
        super().__init__()
        n = cfg.n_modalities   # 6

        # One learnable CLS token per modality
        self.cls_tokens = nn.Parameter(torch.randn(n, cfg.d_model))

        # Per-modality cross-attention: CLS_i attends over all encoder sequences
        # We stack the cross-attn layers; each CLS attends over the concatenated
        # sequences of all other modalities.
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.nhead,
            dropout=cfg.dropout,
            batch_first=True,
        )

        # Per-CLS-token 2-layer FFN (applied independently per modality)
        self.ffn = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.dim_feedforward),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.dim_feedforward, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )
        self.norm = nn.LayerNorm(cfg.d_model)

        self.n_fusion_layers = cfg.fusion_layers

    def forward(
        self,
        enc_seqs: list[Tensor],   # 6 × (B, T_i, d_model) — [MISSING] tensors are zeros
        enc_masks: list[Tensor],  # 6 × (B, T_i) — True = pad; [MISSING] slots are all-True
    ) -> Tensor:
        B = enc_seqs[0].size(0)
        n = len(enc_seqs)         # 6

        # Initialise CLS tokens for this batch: (B, n, d_model)
        cls = self.cls_tokens.unsqueeze(0).expand(B, -1, -1).clone()

        # Concatenate all encoder outputs into one key/value sequence
        # (B, sum(T_i), d_model)
        kv_seq = torch.cat(enc_seqs, dim=1)
        kv_mask = torch.cat(enc_masks, dim=1)  # (B, sum(T_i))

        for _ in range(self.n_fusion_layers):
            # Each CLS token queries across the full concatenated KV sequence
            # query: (B, n, d_model), key/value: (B, sum_T, d_model)
            attn_out, _ = self.cross_attn(
                query=cls,
                key=kv_seq,
                value=kv_seq,
                key_padding_mask=kv_mask,
            )
            cls = self.norm(cls + attn_out)
            cls = cls + self.ffn(cls)

        return cls  # (B, 6, d_model)


# ─── Full Spectroscopic Elucidation Model ─────────────────────────────────────

class SpectroModel(nn.Module):
    """
    Multimodal spectroscopic structure elucidation model.

    Encoder side:
        - IREncoder         → (B, 90, d_model)
        - NMREncoder (1H)   → (B, T_1h+1, d_model)
        - NMREncoder (13C)  → (B, T_13c+1, d_model)   [shared weights with 1H]
        - HSQCEncoder       → (B, T_hsqc+1, d_model)
        - MSMSEncoder (+)   → (B, 3*(T_ms+1), d_model)
        - MSMSEncoder (-)   → (B, 3*(T_ms+1), d_model) [shared weights with +]

    Note on MS encoder: MSMSEncoder handles ONE polarity (3 energy levels) at a
    time. It is called twice — once for ms_pos, once for ms_neg — with shared
    weights. This matches the dataset output of (B, 3, T, 2) per polarity.

    Fusion:
        - CrossModalFusion  → (B, 6, d_model)

    Decoder:
        - nn.TransformerDecoder over the 6-token memory
        - Output projection → (B, seq_len, vocab_size)

    Missing-modality handling:
        Any modality can be flagged as missing. Its encoder output is replaced
        by a single learned [MISSING] token and its attention mask marks all
        positions as padding — the CLS token for that modality receives no
        real information and the decoder learns to ignore it.
    """

    def __init__(self, cfg: SpectroConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # ── Encoders ─────────────────────────────────────────────────────────
        self.ir_encoder   = IREncoder(cfg)
        self.nmr_encoder  = NMREncoder(cfg)    # shared for 1H and 13C
        self.hsqc_encoder = HSQCEncoder(cfg)
        self.ms_encoder   = MSMSEncoder(cfg)   # shared for MS+ and MS-

        # ── Missing-modality placeholder ──────────────────────────────────────
        # One learned vector per modality, used when that modality is absent.
        # Shape: (n_modalities, 1, d_model) — T=1 single token
        self.missing_token = nn.Parameter(torch.randn(cfg.n_modalities, 1, cfg.d_model))

        # ── Fusion ────────────────────────────────────────────────────────────
        self.fusion = CrossModalFusion(cfg)

        # ── Decoder ───────────────────────────────────────────────────────────
        self.tgt_embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_idx)
        self.tgt_pe    = nn.Embedding(cfg.max_decoder_len + 2, cfg.d_model)  # learned positional

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder     = nn.TransformerDecoder(decoder_layer, num_layers=cfg.decoder_layers)
        self.output_proj = nn.Linear(cfg.d_model, cfg.vocab_size)

        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Re-init output projection with small weights to avoid large early logits
        nn.init.normal_(self.output_proj.weight, std=0.02)

    # ── Encoder forward ───────────────────────────────────────────────────────

    def encode(self, batch: dict) -> tuple[Tensor, Tensor]:
        """
        Run all modality encoders and fuse.

        batch keys (all always present after dataset fix, presence of real data
        communicated via the 'available' mask):
            ir:           (B, 1800)
            h_nmr:        (B, T, 5)
            h_nmr_mask:   (B, T) bool
            c_nmr:        (B, T, 2)
            c_nmr_mask:   (B, T) bool
            hsqc:         (B, T, 3)
            hsqc_mask:    (B, T) bool
            ms_pos:       (B, 3, T, 2)
            ms_pos_mask:  (B, 3, T) bool
            ms_neg:       (B, 3, T, 2)
            ms_neg_mask:  (B, 3, T) bool
            available:    (B, 6) bool — True = modality present for this sample

        Returns:
            memory:      (B, 6, d_model) — fused CLS tokens
            memory_mask: (B, 6) bool — True = modality was missing (for decoder cross-attn)
        """
        B = next(v for v in batch.values() if isinstance(v, Tensor) and v.dim() >= 1).size(0)
        device = self.missing_token.device

        available: Tensor = batch.get(
            "available",
            torch.ones(B, self.cfg.n_modalities, dtype=torch.bool, device=device),
        )  # (B, 6)

        enc_seqs: list[Tensor] = []
        enc_masks: list[Tensor] = []

        # Helper: substitute missing modality with learned placeholder
        def _maybe_missing(mod_idx: int, seq: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
            """
            For samples where available[:, mod_idx] is False, replace seq with
            the learned missing token and mark the mask as all-padding so the
            CLS for this modality receives no signal.
            """
            present = available[:, mod_idx]  # (B,) bool
            if present.all():
                return seq, mask

            # missing token: (B, 1, d_model)
            miss = self.missing_token[mod_idx].unsqueeze(0).expand(B, -1, -1)
            # missing mask: all True → CLS query gets no information from KV
            miss_mask = torch.ones(B, 1, dtype=torch.bool, device=device)

            # Per-sample selection (avoid in-place ops for autograd safety)
            pres_idx = present.nonzero(as_tuple=True)[0]
            miss_idx = (~present).nonzero(as_tuple=True)[0]

            out_seq  = seq.clone()
            out_mask = mask.clone()

            if miss_idx.numel() > 0:
                # Pad missing samples' seq to length 1 with the missing token
                out_seq  = torch.zeros(B, 1, self.cfg.d_model, device=device)
                out_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
                if pres_idx.numel() > 0:
                    # Present samples keep their full sequences; we pad to max T
                    T_full = seq.size(1)
                    pad_seq  = torch.zeros(B, T_full, self.cfg.d_model, device=device)
                    pad_mask = torch.ones(B, T_full, dtype=torch.bool, device=device)
                    pad_seq[pres_idx]  = seq[pres_idx]
                    pad_mask[pres_idx] = mask[pres_idx]
                    out_seq  = pad_seq
                    out_mask = pad_mask
                # Overwrite missing samples with the learned token
                out_seq[miss_idx]  = miss[miss_idx]
                out_mask[miss_idx] = miss_mask[miss_idx]

            return out_seq, out_mask

        # ── IR ────────────────────────────────────────────────────────────────
        if "ir" in batch:
            ir_enc = self.ir_encoder(batch["ir"])                  # (B, 90, d_model)
            ir_mask = torch.zeros(B, ir_enc.size(1), dtype=torch.bool, device=device)
        else:
            ir_enc  = torch.zeros(B, 1, self.cfg.d_model, device=device)
            ir_mask = torch.ones(B, 1, dtype=torch.bool, device=device)

        ir_enc, ir_mask = _maybe_missing(MOD_IR, ir_enc, ir_mask)
        enc_seqs.append(ir_enc);  enc_masks.append(ir_mask)

        # ── 1H-NMR ───────────────────────────────────────────────────────────
        if "h_nmr" in batch:
            h_enc = self.nmr_encoder(batch["h_nmr"], batch["h_nmr_mask"], modality_id=0)
            h_mask = torch.cat(
                [torch.zeros(B, 1, dtype=torch.bool, device=device), batch["h_nmr_mask"]], dim=1
            )
        else:
            h_enc  = torch.zeros(B, 1, self.cfg.d_model, device=device)
            h_mask = torch.ones(B, 1, dtype=torch.bool, device=device)

        h_enc, h_mask = _maybe_missing(MOD_1H, h_enc, h_mask)
        enc_seqs.append(h_enc);  enc_masks.append(h_mask)

        # ── 13C-NMR ──────────────────────────────────────────────────────────
        if "c_nmr" in batch:
            c_enc = self.nmr_encoder(batch["c_nmr"], batch["c_nmr_mask"], modality_id=1)
            c_mask = torch.cat(
                [torch.zeros(B, 1, dtype=torch.bool, device=device), batch["c_nmr_mask"]], dim=1
            )
        else:
            c_enc  = torch.zeros(B, 1, self.cfg.d_model, device=device)
            c_mask = torch.ones(B, 1, dtype=torch.bool, device=device)

        c_enc, c_mask = _maybe_missing(MOD_13C, c_enc, c_mask)
        enc_seqs.append(c_enc);  enc_masks.append(c_mask)

        # ── HSQC ─────────────────────────────────────────────────────────────
        if "hsqc" in batch:
            hsqc_enc = self.hsqc_encoder(batch["hsqc"], batch["hsqc_mask"])
            hsqc_mask = torch.cat(
                [torch.zeros(B, 1, dtype=torch.bool, device=device), batch["hsqc_mask"]], dim=1
            )
        else:
            hsqc_enc  = torch.zeros(B, 1, self.cfg.d_model, device=device)
            hsqc_mask = torch.ones(B, 1, dtype=torch.bool, device=device)

        hsqc_enc, hsqc_mask = _maybe_missing(MOD_HSQC, hsqc_enc, hsqc_mask)
        enc_seqs.append(hsqc_enc);  enc_masks.append(hsqc_mask)

        # ── MS/MS positive ────────────────────────────────────────────────────
        # FIX (Issues 2 & 3): ms_pos is (B, 3, T, 2) — 3 energy modes, not 6.
        # MSMSEncoder now expects N_ENERGY_MODES=3. Mask prefix is also 3, not 6.
        if "ms_pos" in batch:
            msp_enc = self.ms_encoder(batch["ms_pos"], batch["ms_pos_mask"])
            # batch["ms_pos_mask"]: (B, 3, T)
            # Each mode contributes one prefix token (never masked) + T peak tokens
            # Rebuild the full mask matching encoder output shape (B, 3*(T+1))
            T_ms = batch["ms_pos"].size(2)
            prefix_mask = torch.zeros(B, 3, 1, dtype=torch.bool, device=device)  # (B, 3, 1)
            # stack prefix + peak mask per mode then flatten
            msp_mask_full = torch.cat(
                [torch.zeros(B, 3, 1, dtype=torch.bool, device=device),
                batch["ms_pos_mask"]],
                dim=2
            ).reshape(B, -1)  # (B, 3*(T+1))
            # Simpler and equivalent: just prepend a False column per mode then flatten
            # ms_pos_mask: (B, 3, T) → per-mode prepend → (B, 3, T+1) → flatten → (B, 3*(T+1))
            msp_mask_full = torch.cat(
                [torch.zeros(B, 3, 1, dtype=torch.bool, device=device),
                 batch["ms_pos_mask"]], dim=2
            ).reshape(B, -1)  # (B, 3*(T+1))
        else:
            msp_enc       = torch.zeros(B, 1, self.cfg.d_model, device=device)
            msp_mask_full = torch.ones(B, 1, dtype=torch.bool, device=device)

        msp_enc, msp_mask_full = _maybe_missing(MOD_MSP, msp_enc, msp_mask_full)
        enc_seqs.append(msp_enc);  enc_masks.append(msp_mask_full)

        # ── MS/MS negative ────────────────────────────────────────────────────
        # FIX (Issues 2 & 3): same fix as MS+ above — 3 modes, prefix shape (B,3,1)
        if "ms_neg" in batch:
            msn_enc = self.ms_encoder(batch["ms_neg"], batch["ms_neg_mask"])
            msn_mask_full = torch.cat(
                [torch.zeros(B, 3, 1, dtype=torch.bool, device=device),
                 batch["ms_neg_mask"]], dim=2
            ).reshape(B, -1)  # (B, 3*(T+1))
        else:
            msn_enc       = torch.zeros(B, 1, self.cfg.d_model, device=device)
            msn_mask_full = torch.ones(B, 1, dtype=torch.bool, device=device)

        msn_enc, msn_mask_full = _maybe_missing(MOD_MSN, msn_enc, msn_mask_full)
        enc_seqs.append(msn_enc);  enc_masks.append(msn_mask_full)

        # ── Fusion ────────────────────────────────────────────────────────────
        memory = self.fusion(enc_seqs, enc_masks)  # (B, 6, d_model)

        # Memory key padding mask for decoder: missing modalities → True (ignored)
        # A modality is "fully missing" if its entire enc_mask is True
        memory_mask = torch.stack(
            [m.all(dim=1) for m in enc_masks], dim=1
        )  # (B, 6) bool

        return memory, memory_mask

    # ── Decoder forward ───────────────────────────────────────────────────────

    def decode(
        self,
        tgt_ids: Tensor,       # (B, T_tgt) — teacher-forced token ids
        memory: Tensor,        # (B, 6, d_model)
        memory_mask: Tensor,   # (B, 6) bool
    ) -> Tensor:
        B, T = tgt_ids.shape
        device = tgt_ids.device

        # Token + position embedding
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)  # (B, T)
        x = self.tgt_embed(tgt_ids) + self.tgt_pe(positions)  # (B, T, d_model)

        # Causal mask for autoregressive decoding
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)

        out = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=causal_mask,
            memory_key_padding_mask=memory_mask,
        )  # (B, T, d_model)

        return self.output_proj(out)  # (B, T, vocab_size)

    # ── Full forward pass (teacher-forced, for training) ──────────────────────

    def forward(self, batch: dict, tgt_ids: Tensor) -> Tensor:
        """
        batch: dict of spectral tensors (see encode() docstring)
        tgt_ids: (B, T) — [BOS, tok1, tok2, ..., tokN] (input to decoder)

        Returns logits: (B, T, vocab_size)
        """
        memory, memory_mask = self.encode(batch)
        return self.decode(tgt_ids, memory, memory_mask)

    # ── Greedy / beam search (inference) ─────────────────────────────────────

    @torch.inference_mode()
    def beam_search(
        self,
        batch: dict,
        beam_size: int | None = None,
        max_len: int | None = None,
    ) -> list[list[list[int]]]:
        """
        Beam search decoding. Returns list (batch) of list (beams) of token-id lists.
        Caller converts token-ids → SELFIES → SMILES via the vocab.

        This is a straightforward beam search; replace with OpenNMT-py's
        beam search for production use (handles length penalties etc.).
        """
        k = beam_size or self.cfg.beam_size
        max_len = max_len or self.cfg.max_decoder_len
        cfg = self.cfg
        device = next(self.parameters()).device

        memory, memory_mask = self.encode(batch)
        B = memory.size(0)

        results: list[list[list[int]]] = []

        for b in range(B):
            mem_b  = memory[b : b + 1]       # (1, 6, d_model)
            mask_b = memory_mask[b : b + 1]  # (1, 6)

            # beams: list of (score, token_ids)
            beams: list[tuple[float, list[int]]] = [(0.0, [cfg.bos_idx])]
            completed: list[tuple[float, list[int]]] = []

            for _ in range(max_len):
                if len(beams) == 0:
                    break

                all_candidates: list[tuple[float, list[int]]] = []
                for score, ids in beams:
                    tgt = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
                    logits = self.decode(tgt, mem_b, mask_b)  # (1, T, vocab)
                    log_probs = torch.log_softmax(logits[0, -1], dim=-1)  # (vocab,)
                    top_lp, top_tok = log_probs.topk(k)

                    for lp, tok in zip(top_lp.tolist(), top_tok.tolist()):
                        all_candidates.append((score + lp, ids + [tok]))

                # Prune to top-k
                all_candidates.sort(key=lambda x: x[0], reverse=True)
                beams = []
                for sc, ids in all_candidates[: k * 2]:
                    if ids[-1] == cfg.eos_idx:
                        completed.append((sc / len(ids), ids))  # length-normalised
                    else:
                        beams.append((sc, ids))
                    if len(beams) == k:
                        break

            # Fill remaining beams into completed
            completed.extend((sc / max(len(ids), 1), ids) for sc, ids in beams)
            completed.sort(key=lambda x: x[0], reverse=True)
            results.append([ids for _, ids in completed[:k]])

        return results