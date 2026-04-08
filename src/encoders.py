"""
Modality-specific input encoders for multimodal spectroscopic structure elucidation.

Each encoder ingests one spectral modality and outputs a sequence of d_model-dim
hidden vectors that feed into the cross-modal attention fusion layer.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import SpectroConfig


# ─── Positional encoding ──────────────────────────────────────────────────────

class SinusoidalPE(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani 2017)."""

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, d_model)
        return self.dropout(x + self.pe[:, : x.size(1)])


# ─── IR Encoder ───────────────────────────────────────────────────────────────

class IREncoder(nn.Module):
    """
    ViT-style patch projection for IR spectra.

    Input:  (B, 1800) — raw transmittance vector, 2 cm⁻¹ resolution
    Output: (B, 90, d_model) — 90 non-overlapping 20-pt patches
    """

    def __init__(self, cfg: SpectroConfig) -> None:
        super().__init__()
        self.patch_size = cfg.ir_patch_size     # 20 points
        self.n_patches = cfg.ir_n_patches       # 90

        # Learnable linear projection: 20 → d_model (identical to ViT patch embed)
        self.patch_proj = nn.Linear(cfg.ir_patch_size, cfg.d_model)

        self.pe = SinusoidalPE(cfg.d_model, max_len=cfg.ir_n_patches, dropout=cfg.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,   # pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.ir_encoder_layers)

    def forward(self, ir: Tensor) -> Tensor:
        # ir: (B, 1800)
        B = ir.size(0)
        # Reshape into patches: (B, 90, 20)
        patches = ir.view(B, self.n_patches, self.patch_size)
        # Project each patch to d_model: (B, 90, d_model)
        x = self.patch_proj(patches)
        x = self.pe(x)
        return self.transformer(x)  # (B, 90, d_model)


# ─── Shared NMR Encoder ───────────────────────────────────────────────────────

class NMREncoder(nn.Module):
    """
    Shared transformer encoder for 1H-NMR and 13C-NMR peak lists.

    Both modalities share weights; a learned modality-type embedding
    (prepended as position-0 token) differentiates them — analogous to
    BERT segment embeddings.

    1H-NMR peak tokens: [ppm_start, ppm_end, mult_type_embed, nH, integration]
      → 4 continuous + 1 categorical field per peak, then a separator
    13C-NMR peak tokens: [ppm_position, intensity]
      → 2 continuous fields per peak, then a separator

    Input:
        peaks:       (B, T, max_fields) — zero-padded peak tensor
        peak_mask:   (B, T) bool — True where padding (for attention mask)
        modality_id: int — 0 for 1H, 1 for 13C
    Output: (B, T+1, d_model) — includes prepended modality token
    """

    # Multiplicity types: s, d, t, q, m, dd, dt, ddd, td, qd, other
    N_MULT_TYPES = 12

    def __init__(self, cfg: SpectroConfig) -> None:
        super().__init__()
        self.d_model = cfg.d_model

        # Modality-type embeddings: index 0 = 1H, index 1 = 13C
        self.modality_embed = nn.Embedding(2, cfg.d_model)

        # Multiplicity embedding for 1H peaks (categorical field)
        self.mult_embed = nn.Embedding(self.N_MULT_TYPES + 1, 32, padding_idx=0)

        # Input projection for each modality
        # 1H: 4 continuous + 32-dim mult embed = 36 → d_model
        # 13C: 2 continuous → d_model
        self.proj_1h = nn.Linear(4 + 32, cfg.d_model)
        self.proj_13c = nn.Linear(2, cfg.d_model)

        self.pe = SinusoidalPE(cfg.d_model, max_len=cfg.nmr_max_peaks + 1, dropout=cfg.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.nmr_encoder_layers)

    def forward(
        self,
        peaks: Tensor,           # (B, T, fields) — fields=5 for 1H, 2 for 13C
        peak_mask: Tensor,       # (B, T) True = padding
        modality_id: int,        # 0 = 1H, 1 = 13C
    ) -> Tensor:
        B, T, _ = peaks.shape

        if modality_id == 0:  # 1H-NMR
            # peaks fields: [ppm_start, ppm_end, mult_type(int), nH, integration]
            cont = torch.cat([peaks[..., :2], peaks[..., 3:5]], dim=-1)  # (B, T, 4)
            mult = self.mult_embed(peaks[..., 2].long())                  # (B, T, 32)
            x = self.proj_1h(torch.cat([cont, mult], dim=-1))            # (B, T, d_model)
        else:  # 13C-NMR
            x = self.proj_13c(peaks[..., :2])                            # (B, T, d_model)

        # Prepend modality token: (B, 1, d_model)
        mod_token = self.modality_embed(
            torch.full((B, 1), modality_id, dtype=torch.long, device=peaks.device)
        )
        x = torch.cat([mod_token, x], dim=1)                 # (B, T+1, d_model)

        # Extend mask to cover the prepended modality token (never masked)
        full_mask = torch.cat(
            [torch.zeros(B, 1, dtype=torch.bool, device=peaks.device), peak_mask], dim=1
        )  # (B, T+1)

        x = self.pe(x)
        return self.transformer(x, src_key_padding_mask=full_mask)  # (B, T+1, d_model)


# ─── HSQC Encoder ─────────────────────────────────────────────────────────────

class HSQCEncoder(nn.Module):
    """
    Peak-list transformer for HSQC 2D NMR.

    We use the annotated peak list (x_ppm, y_ppm, integration) rather than the
    raw 512×512 matrix. This is memory-efficient (max 23 peaks observed in A7)
    and directly encodes the chemically meaningful cross-peaks.

    The 512×512 CNN route would cost 1,024 tokens per sample — at 4096-token
    effective batch size that severely limits batch diversity. Peak-list encodes
    the same information in ≤32 tokens.

    Input:
        peaks:     (B, T, 3) — [x_ppm, y_ppm, integration], zero-padded
        peak_mask: (B, T) bool — True where padding
    Output: (B, T+1, d_model) — includes prepended [HSQC] modality token
    """

    def __init__(self, cfg: SpectroConfig) -> None:
        super().__init__()
        self.proj = nn.Linear(cfg.hsqc_peak_fields, cfg.d_model)   # 3 → d_model
        self.modality_token = nn.Parameter(torch.randn(1, 1, cfg.d_model))

        self.pe = SinusoidalPE(cfg.d_model, max_len=cfg.hsqc_max_peaks + 1, dropout=cfg.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.hsqc_encoder_layers)

    def forward(self, peaks: Tensor, peak_mask: Tensor) -> Tensor:
        # peaks: (B, T, 3), peak_mask: (B, T)
        B, T, _ = peaks.shape
        x = self.proj(peaks)                                        # (B, T, d_model)

        mod = self.modality_token.expand(B, 1, -1)                  # (B, 1, d_model)
        x = torch.cat([mod, x], dim=1)                              # (B, T+1, d_model)

        full_mask = torch.cat(
            [torch.zeros(B, 1, dtype=torch.bool, device=peaks.device), peak_mask], dim=1
        )

        x = self.pe(x)
        return self.transformer(x, src_key_padding_mask=full_mask)  # (B, T+1, d_model)


# ─── MS/MS Encoder ────────────────────────────────────────────────────────────

class MSMSEncoder(nn.Module):
    """
    Peak-pair transformer for MS/MS spectra.

    # FIX (Issue 2): Changed N_ENERGY_MODES from 6 to 3.
    # The dataset produces separate (B, 3, T, 2) tensors for positive and
    # negative ion modes. This encoder is called TWICE in model.encode() —
    # once for ms_pos and once for ms_neg — each with 3 energy levels
    # (10 eV, 20 eV, 40 eV). Weights are shared between the two calls.
    #
    # Old design assumed a single (B, 6, T, 2) tensor combining both polarities,
    # which never matched what the dataset actually produced.

    Input:
        peaks:      (B, 3, T, 2) — [m/z, intensity] per peak, zero-padded
                    dim-1 order: 10eV, 20eV, 40eV  (for either pos or neg polarity)
        peak_mask:  (B, 3, T) bool — True where padding
    Output: (B, 3*(T+1), d_model)  — energy-mode prefix tokens + all peak tokens
    """

    # CHANGED: was 6 (combined pos+neg), now 3 (one polarity at a time)
    N_ENERGY_MODES = 3

    def __init__(self, cfg: SpectroConfig) -> None:
        super().__init__()
        # Project (m/z, intensity) pair to d_model
        self.peak_proj = nn.Linear(cfg.ms_peak_fields, cfg.d_model)

        # Learned energy-mode prefix embeddings (3 modes: 10eV, 20eV, 40eV)
        # CHANGED: was Embedding(6, ...) — now Embedding(3, ...)
        self.energy_mode_embed = nn.Embedding(self.N_ENERGY_MODES, cfg.d_model)

        self.pe = SinusoidalPE(
            cfg.d_model,
            max_len=self.N_ENERGY_MODES * (cfg.ms_max_peaks_per_mode + 1) + 1,
            dropout=cfg.dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.ms_encoder_layers)

    def forward(self, peaks: Tensor, peak_mask: Tensor) -> Tensor:
        # peaks: (B, 3, T, 2), peak_mask: (B, 3, T)
        B, M, T, _ = peaks.shape
        assert M == self.N_ENERGY_MODES, (
            f"MSMSEncoder expects (B, {self.N_ENERGY_MODES}, T, 2) but got dim-1={M}. "
            f"Pass pos and neg modes separately."
        )

        device = peaks.device
        mode_ids = torch.arange(M, device=device)  # (3,)

        seq_parts: list[Tensor] = []
        mask_parts: list[Tensor] = []

        for m in range(M):
            # Energy-mode prefix token: (B, 1, d_model)
            prefix = self.energy_mode_embed(
                mode_ids[m].expand(B)
            ).unsqueeze(1)
            # Peak tokens: (B, T, d_model)
            pk = self.peak_proj(peaks[:, m])

            # Concatenate prefix + peaks: (B, T+1, d_model)
            group = torch.cat([prefix, pk], dim=1)
            seq_parts.append(group)

            # Mask: prefix is never masked
            prefix_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
            mask_parts.append(torch.cat([prefix_mask, peak_mask[:, m]], dim=1))

        # Full sequence: (B, 3*(T+1), d_model)
        x = torch.cat(seq_parts, dim=1)
        full_mask = torch.cat(mask_parts, dim=1)  # (B, 3*(T+1))

        x = self.pe(x)
        return self.transformer(x, src_key_padding_mask=full_mask)  # (B, seq, d_model)