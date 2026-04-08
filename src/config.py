from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SpectroConfig:
    # ── Vocabulary (from A14) ─────────────────────────────────────────────────
    vocab_size: int = 111          # SELFIES token count including specials
    pad_idx: int = 0
    bos_idx: int = 1
    eos_idx: int = 2
    unk_idx: int = 3

    # ── Decoder sequence length (from A5: p99=62 + 20 buffer) ────────────────
    max_decoder_len: int = 82

    # ── Shared transformer hidden dim ─────────────────────────────────────────
    d_model: int = 512
    nhead: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.1

    # ── IR encoder (patch-projection ViT style) ───────────────────────────────
    ir_spectrum_len: int = 1800    # points (400–4000 cm⁻¹, 2 cm⁻¹ res)
    ir_patch_size: int = 20        # points per patch
    ir_n_patches: int = 90         # 1800 / 20
    ir_encoder_layers: int = 4

    # ── NMR encoder (shared for 1H and 13C) ──────────────────────────────────
    h_nmr_peak_fields: int = 5     # ppm_start, ppm_end, mult_type, nH, integration
    c_nmr_peak_fields: int = 2     # ppm_position, intensity
    nmr_encoder_layers: int = 4
    nmr_max_peaks: int = 64        # generous upper bound from A7 (max 1H=21, 13C=58)

    # ── HSQC encoder (CNN + transformer) ─────────────────────────────────────
    hsqc_matrix_size: int = 512    # 512×512 input
    hsqc_cnn_channels: tuple = field(default_factory=lambda: (32, 64, 128))
    hsqc_spatial_after_cnn: int = 32   # 512 / 2^4 = 32 (4 pooling layers of stride 2)
    hsqc_encoder_layers: int = 2
    # CNN output: (128, 32, 32) → flattened to 1024 tokens of 128-dim → projected to d_model
    # NOTE: using peak-list path for HSQC instead (memory-friendly); see HSQCEncoder

    # HSQC peak list fields: x_ppm, y_ppm, integration
    hsqc_peak_fields: int = 3
    hsqc_max_peaks: int = 32       # from A7: max=23, using 32 as safe upper bound

    # ── MS/MS encoder ─────────────────────────────────────────────────────────
    ms_peak_fields: int = 2        # m/z, intensity
    ms_n_energy_modes: int = 6     # pos10/20/40, neg10/20/40
    ms_encoder_layers: int = 4
    ms_max_peaks_per_mode: int = 64

    # ── Cross-modal fusion ────────────────────────────────────────────────────
    n_modalities: int = 6          # IR, 1H-NMR, 13C-NMR, HSQC, MS+, MS-
    fusion_layers: int = 2

    # ── Decoder ───────────────────────────────────────────────────────────────
    decoder_layers: int = 4
    beam_size: int = 10

    # ── Training ──────────────────────────────────────────────────────────────
    warmup_steps: int = 4000
    max_steps: int = 300_000
    effective_batch_tokens: int = 4096
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1

    # ── Modality dropout (missing-modality training) ──────────────────────────
    modality_dropout_p: float = 0.30   # per-modality, per-sample

    # ── Rare FG oversampling (from A10) ──────────────────────────────────────
    rare_fg_groups: tuple = field(default_factory=lambda: (
        "nitro", "boronic_acid", "phosphate", "lactone", "anhydride"
    ))
    oversample_factor: int = 5

    # ── Paths ─────────────────────────────────────────────────────────────────
    data_root: Path = Path("multimodal_spectroscopic_dataset")
    output_dir: Path = Path("outputs")
    vocab_path: Path = Path("audit_outputs/selfies_vocab.json")
    metadata_path: Path = Path("audit_outputs/dataset_metadata_with_chunks.parquet")
