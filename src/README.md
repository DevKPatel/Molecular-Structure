# Multimodal Spectroscopic Structure Elucidation
### Deep Learning Pipeline — Phase A Complete, Phase B Ready

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Execution Order — What to Run and When](#2-execution-order)
3. [What Phase A Found and What We Did About It](#3-phase-a-results-and-decisions)
4. [How the 24 Empty HSQC Molecules Are Handled](#4-the-24-empty-hsqc-molecules)
5. [What Each Phase B File Does](#5-phase-b-file-guide)
6. [Manual Changes You Must Make](#6-required-manual-changes)
7. [GPU Memory / Batch Size Reference](#7-gpu-memory-reference)
8. [Expected Training Milestones](#8-expected-training-milestones)
9. [Dependencies](#9-dependencies)

---

## 1. Project Structure

Your project root should look exactly like this. Everything under `model/` is
Phase B code delivered in this session. Everything under `audit_outputs/` was
produced by Phase A.

```
project_root/                          ← your working directory
│
├── multimodal_spectroscopic_dataset/  ← original dataset (downloaded separately)
│   ├── aligned_chunk_0.parquet        ← 245 chunk files (3,235 rows each)
│   ├── aligned_chunk_1.parquet
│   └── ...
│
├── audit_outputs/                     ← produced by Phase A scripts
│   ├── dataset_metadata_with_chunks.parquet   ← CRITICAL: scaffold-split index
│   ├── selfies_vocab.json                     ← CRITICAL: 111-token SELFIES vocab
│   ├── null_report.json
│   ├── tanimoto_diversity.png
│   ├── functional_group_imbalance.png
│   ├── heavy_atom_distribution.png
│   ├── length_distributions.png
│   └── sample_ir_spectra.png
│
├── model/                             ← Phase B: all model code lives here
│   ├── __init__.py
│   ├── config.py                      ← SpectroConfig dataclass (all hyperparameters)
│   ├── encoders.py                    ← IR, NMR, HSQC, MS/MS encoder modules
│   ├── model.py                       ← CrossModalFusion + SpectroModel
│   ├── dataset.py                     ← SpectroDataset, collate_fn, DataLoader factory
│   ├── train.py                       ← Stage 1 training loop
│   └── smoke_test.py                  ← shape/gradient verification (no real data needed)
│
├── outputs/                           ← created automatically during training
│   └── checkpoints/                   ← saved model checkpoints (best 5 by Top-1)
│
└── wandb/                             ← created automatically by wandb
```

> **Important:** `model/` must stay as a Python package (the `__init__.py` must
> be present). All scripts are run from `project_root/`, not from inside `model/`.

---

## 2. Execution Order

Run these in strict order. Do not skip a step and do not start the next step
until the previous one passes cleanly.

### Step 1 — Verify the Phase A outputs exist

Before touching any Phase B code, confirm these two files exist. They are the
only Phase A outputs that Phase B actually depends on:

```
audit_outputs/dataset_metadata_with_chunks.parquet
audit_outputs/selfies_vocab.json
```

If either is missing, re-run the relevant Phase A step (A13 for parquet, A14 for vocab).

---

### Step 2 — Run the smoke test (no GPU, no real data needed)

```bash
# From project_root/
python -m model.smoke_test
```

This creates synthetic random tensors and runs a complete forward pass, backward
pass, and beam search through every component. It takes about 10 seconds on CPU.
You should see:

```
Running smoke test on: cuda   (or cpu)
✓ IREncoder       output: (4, 90, 512)
✓ NMREncoder (1H) output: (4, 11, 512)
✓ NMREncoder (13C) output: (4, 16, 512)
✓ HSQCEncoder     output: (4, 9, 512)
✓ MSMSEncoder     output: (4, 63, 512)
Model parameters: ~45,000,000
✓ Forward pass    output: (4, 82, 111)
✓ Gradient flow   OK — all params received gradients
✓ Missing-modality path output: (4, 82, 111)
✓ Beam search     4 samples × 3 beams
✓ All Phase B smoke tests passed.
```

> **Note on MSMSEncoder output shape:** The smoke test will show `(4, 63, 512)`
> (= 3 modes × (T+1) tokens), not `(4, 126, 512)` as previously documented.
> This is correct — MSMSEncoder now handles one polarity (3 energy levels) at a
> time and is called twice (once for MS+, once for MS-).

**Do not proceed to Step 3 until this passes completely.**

---

### Step 3 — Verify dataset loading (one-time data check)

Before full training, confirm the dataset loads correctly with real data.
Run this in a Python shell from `project_root/`:

```python
import sys
sys.path.insert(0, ".")
from model.config import SpectroConfig
from model.dataset import SpectroDataset
cfg = SpectroConfig()
ds = SpectroDataset(
    metadata_path=cfg.metadata_path,
    data_root=cfg.data_root,
    vocab_path=cfg.vocab_path,
    cfg=cfg,
    split="train",
    augment=False,
    modality_dropout=False,
)
print(f"Dataset size: {len(ds)}")   # expect 710345

item = ds[0]
print("Keys:", sorted(item.keys()))
# Expect all 12 keys always present:
# available, c_nmr, c_nmr_mask, h_nmr, h_nmr_mask, hsqc, hsqc_mask,
# ir, ms_neg, ms_neg_mask, ms_pos, ms_pos_mask, selfies_ids

print("IR shape:",         item["ir"].shape)           # torch.Size([1800])
print("h_nmr shape:",      item["h_nmr"].shape)        # torch.Size([64, 5])
print("ms_pos shape:",     item["ms_pos"].shape)       # torch.Size([3, 64, 2])
print("selfies_ids shape:",item["selfies_ids"].shape)  # torch.Size([83])
print("available:",        item["available"])          # 6 booleans
```

All 12 keys must be present in every item. If `ms_pos` shows shape `(6, 64, 2)`
instead of `(3, 64, 2)`, you have an old version of `dataset.py` — replace it.

To also verify chunk loading manually (optional sanity check):

```python
import pandas as pd
from pathlib import Path

meta = pd.read_parquet("audit_outputs/dataset_metadata_with_chunks.parquet")
print(meta.columns.tolist())

# chunk_file stores the ABSOLUTE path written by Phase A — use it directly
chunk_path = Path(meta["chunk_file"].iloc[0])
print(chunk_path)          # e.g. D:\...\aligned_chunk_0.parquet
chunk = pd.read_parquet(chunk_path)
print(chunk.columns.tolist())
print(chunk.iloc[0]["ir_spectra"])
```

**See Section 6 (Manual Changes) for field-name checks based on what you find.**

---

### Step 4 — Stage 1 training, NMR-only curriculum

Following the curriculum plan, train NMR-only first. Set `available` to only
use modality indices 1 (1H) and 2 (13C) by temporarily editing `SpectroConfig`:

```python
# In config.py, temporarily set for NMR-only phase:
modality_dropout_p: float = 0.0   # no dropout during NMR-only phase
```

Then in `train.py`, add a `forced_available` mask to the batch before encoding:

```python
# In the training loop, before: logits = model(batch_dev, tgt_in)
# Force only NMR modalities:
B = tgt_in.size(0)
batch_dev["available"] = torch.zeros(B, 6, dtype=torch.bool, device=device)
batch_dev["available"][:, 1] = True   # 1H-NMR
batch_dev["available"][:, 2] = True   # 13C-NMR
```

Run training:

```bash
# A100 80GB:
python -m model.train --physical-batch 512 --accum-steps 8 --device cuda

# RTX 4090 24GB:
python -m model.train --physical-batch 128 --accum-steps 32 --device cuda

# 2x A6000 48GB (use torchrun for multi-GPU, or just one GPU with larger batch):
python -m model.train --physical-batch 256 --accum-steps 16 --device cuda

# A1000 8GB VRAM, Intel i7 128GB RAM:
py -3.11 -m model.train --physical-batch 12 --accum-steps 12 --workers 8 --device cuda --no-compile
```

Target before advancing: **validation Top-1 ≥ 60%** (should reach this around
step 80,000–120,000).

---

### Step 5 — Add MS/MS (curriculum stage 2)

Load the NMR-only checkpoint, then remove the `forced_available` override and
add modality indices 4 and 5 to the training loop. Train until Top-1 plateaus
(typically 15–20k additional steps).

---

### Step 6 — Add IR (curriculum stage 3)

Add modality index 0 (IR). Train until Top-1 plateaus.

---

### Step 7 — Add HSQC and full multimodal (curriculum stage 4)

Add modality index 3 (HSQC) and restore `modality_dropout_p = 0.30` in
`SpectroConfig`. This is now the full model with all 6 modalities and
random modality dropout active.

---

### Step 8 — Stage 2 and 3 (domain adaptation and fine-tuning)

These use the same `SpectroModel` architecture with frozen encoder weights
(Stage 2) or full unfreezing at 0.1× LR (Stage 3). Code for these stages
will be delivered separately.

---

## 3. Phase A Results and Decisions

Here is a plain-language account of what the Phase A audit found and how it
shaped every Phase B decision.

### Dataset size: 789,272 (not 794,403)

The raw dataset has 794,403 molecules. Phase A found **5,131 duplicate
canonical SMILES** (0.6%). After deduplication, the clean count is **789,272**.
This is the number you should cite in any writeup. All downstream splits use
the deduplicated set.

### Train / Val / Test split: 710,345 / 39,464 / 39,463

Generated by **Murcko scaffold-based splitting**, not random splitting. This is
the honest split — every molecule sharing a ring scaffold family is assigned
exclusively to one split. The maximum Tanimoto similarity between any test
molecule and any training molecule is **0.600**, well below the 0.85 danger
threshold. This means the test set contains genuinely novel scaffolds.

*Why this matters:* the paper reports 73.38% Top-1 accuracy on random splits.
On scaffold splits, you should expect 5–15% lower, around 60–68%. That lower
number is the real-world estimate. Do not be alarmed when scaffold-split
accuracy is lower than the paper — it is supposed to be.

### SELFIES vocabulary: 111 tokens

Built from the actual molecules in this dataset, including special tokens
`[PAD]=0, [BOS]=1, [EOS]=2, [UNK]=3`. The decoder embedding layer is exactly
111-dimensional output. This is already baked into `SpectroConfig.vocab_size = 111`.

100% SELFIES round-trip: every molecule converts to SELFIES and back to the
exact same canonical SMILES. Zero failures. This confirms the vocab is complete
for this dataset.

### Max decoder sequence length: 82

From the SELFIES token count distribution: p99 = 62 tokens, plus 20 buffer = 82.
The max observed SELFIES length is 75 tokens. `SpectroConfig.max_decoder_len = 82`
gives comfortable headroom with minimal padding waste.

### IR spectra: 1,800 points, 90 patches

All IR spectra are confirmed 1,800-point vectors at 2 cm⁻¹ resolution
(400–4000 cm⁻¹). Zero degenerate (all-zero) spectra. The ViT-style patch
encoder divides this into **90 non-overlapping patches of 20 points each**.
Each 20-point patch spans 40 cm⁻¹ — roughly one IR peak width — so each patch
token encodes one peak's worth of information.

### NMR peak list lengths

From A7: 1H-NMR peaks range 3–21 per molecule (mean 9.3), 13C-NMR peaks range
4–58 (mean 17.4). `SpectroConfig.nmr_max_peaks = 64` safely covers both.

### Dataset diversity: mean pairwise Tanimoto = 0.116

This is very low (more diverse = lower). No pair in the 5,000-sample check
exceeded 0.415. This is a genuinely chemically diverse dataset — the model
cannot succeed by memorizing a few scaffold families.

### Rare functional groups: 5 groups need oversampling

From A10, these five groups have estimated counts below 1,000 in the full
789,272-molecule dataset:
- `nitro` — estimated 0 molecules (0.0%) ← genuinely absent or near-absent
- `boronic_acid` — estimated 0 molecules (0.0%)
- `phosphate` — estimated ~31 molecules (0.004%)
- `lactone` — estimated ~126 molecules (0.016%)
- `anhydride` — estimated ~615 molecules (0.08%)

The `WeightedRandomSampler` in `dataset.py` applies a **5× oversampling factor**
to any training molecule containing one of these groups. This does not change
the dataset — it changes how often these molecules appear in a training epoch.

Note that `nitro` and `boronic_acid` appear to be effectively absent from this
dataset (0 in a 15,831-sample audit). The SMARTS patterns may not match how
these groups appear in the USPTO molecules, or they genuinely are absent. Either
way, oversampling with weight=5 on an empty set has no effect — it is safe.

### Heavy atom distribution

Peaks at 18–22 atoms, cleanly within the paper's [5, 35] range. Zero molecules
outside bounds. The distribution matches Figure 2A of the reference paper
exactly, confirming the data loading is correct.

---

## 4. The 24 Empty HSQC Molecules

**Short answer:** they are handled automatically. You do not need to do anything.

**What they are:** HSQC (Heteronuclear Single Quantum Coherence) is a 2D NMR
experiment that detects H-C bonds. It produces one cross-peak for every
directly bonded H-C pair in the molecule. If a molecule has **no H-C bonds**,
it has no HSQC peaks — the peak list is empty.

These 24 molecules are **acyclic molecules with no directly bonded hydrogens on
carbon**, which is a chemically valid situation (e.g., fully substituted carbons,
or molecules where every C is attached only to heteroatoms). They are not data
errors.

**What the code does with them:**

In `dataset.py`, `__getitem__`:

```python
if available[3]:                                    # if HSQC was supposed to be available
    hsqc_peaks = _parse_peak_list(row.get("hsqc_nmr_peaks", []))
    if len(hsqc_peaks) == 0:
        # Acyclic molecule — treat as genuinely missing, not as a data error
        available[3] = False                        # flip the availability flag to False
    else:
        tokens, mask = tokenize_hsqc(hsqc_peaks, self.cfg.hsqc_max_peaks)
        item["hsqc"]      = torch.from_numpy(tokens)
        item["hsqc_mask"] = torch.from_numpy(mask)
```

When `available[3]` is set to False, the model's `encode()` method in `model.py`
uses the learned `[MISSING]` token for the HSQC modality slot. The cross-modal
fusion CLS token for HSQC gets no real information, and the model learns during
training that the HSQC slot being missing does not mean failure — it means "this
molecule has no H-C correlations to show you."

Regardless of the HSQC availability flag, `__getitem__` always emits `hsqc` and
`hsqc_mask` keys (zero tensor + all-True mask for absent cases), so every sample
in a batch has a consistent key set and `spectro_collate` can do a simple stack.

---

## 5. Phase B File Guide

### `config.py` — SpectroConfig

A single Python dataclass containing every hyperparameter. **This is the one
file you change to run different experiments.** All other files import from it.

Key values derived directly from Phase A:
| Config field | Value | Source |
|---|---|---|
| `vocab_size` | 111 | A14: SELFIES vocab size |
| `max_decoder_len` | 82 | A5: p99 SELFIES tokens + 20 |
| `ir_n_patches` | 90 | A7: 1800 ÷ 20 |
| `nmr_max_peaks` | 64 | A7: 13C max=58, padded to 64 |
| `hsqc_max_peaks` | 32 | A7: HSQC max=23, padded to 32 |

---

### `encoders.py` — Four modality-specific encoders

**IREncoder** — Takes `(B, 1800)` raw transmittance vector. Reshapes to
`(B, 90, 20)` patches, applies a learnable linear projection to 512-dim
(identical to ViT patch embedding), adds sinusoidal positional encoding
so the model knows each patch's wavenumber position, then runs 4 pre-LN
transformer encoder layers. Output: `(B, 90, 512)`.

**NMREncoder** — Shared weights for both 1H-NMR and 13C-NMR. A prepended
learned modality-type embedding (index 0 = 1H, index 1 = 13C) differentiates
the two modalities, analogous to BERT's segment A/B embeddings. For 1H-NMR,
each peak is tokenized as 5 fields: `[ppm_start, ppm_end, mult_type, nH,
integration]` where `mult_type` is an integer that gets embedded through a
small embedding table (s=1, d=2, t=3, q=4, m=5, dd=6, etc.). For 13C-NMR,
each peak is just `[ppm_position, intensity]`. Output: `(B, T_peaks+1, 512)`.

**HSQCEncoder** — Takes the annotated peak list `(B, T, 3)` where each peak
is `[x_ppm, y_ppm, integration]`. A learned modality token is prepended.
2 transformer encoder layers. Output: `(B, T_peaks+1, 512)`. This uses the
peak list rather than the raw 512×512 matrix — cheaper and chemically cleaner.

**MSMSEncoder** — Takes `(B, 3, T, 2)` where dim-1 is 3 energy levels
(10 eV, 20 eV, 40 eV) for **one polarity**. Each energy level gets a learned
prefix token prepended to its peak sequence. The 3 groups are concatenated.
The encoder is called **twice** in `model.encode()` — once for `ms_pos` and
once for `ms_neg` — with **shared weights** between the two calls (the energy-
mode prefix embeddings index 0/1/2 are the same for both polarities, letting
the encoder focus on peak pattern rather than polarity). Output per call:
`(B, 3*(T+1), 512)`.

> **Why 3 modes, not 6?** The dataset stores positive and negative ion modes
> separately as `ms_pos` and `ms_neg`. Passing them combined as 6 modes would
> require merging them before the encoder and splitting the mask afterward —
> more complexity for no architectural benefit. Calling the shared encoder twice
> is simpler and equally expressive.

---

### `model.py` — CrossModalFusion + SpectroModel

**CrossModalFusion** — Six learnable CLS tokens (one per modality). Each CLS
token attends via cross-attention across the full concatenated key-value
sequence of all encoder outputs. This runs for `fusion_layers=2` rounds. The
result is 6 vectors of 512-dim each, representing each modality's globally
relevant summary. The decoder then attends over just these 6 vectors rather
than 1,200+ raw encoder tokens.

**SpectroModel** — Wires all encoders → fusion → a standard 4-layer pre-LN
transformer decoder → linear projection to vocab_size=111. The decoder uses
learned positional embeddings (not sinusoidal) because the SELFIES token
sequence is short (≤82 tokens) and positions are semantically meaningful.

The `encode()` method handles both the case where a modality key is absent
from the batch dict entirely (batch-wide absence) and the case where the
`available` mask marks specific samples as missing within a batch.

---

### `dataset.py` — SpectroDataset

Reads from `audit_outputs/dataset_metadata_with_chunks.parquet` (the scaffold-split
index produced by A13), lazy-loads dense spectral arrays from the chunk files,
and applies all augmentations and tokenizations.

**Key design:** `__getitem__` **always returns all 12 keys** for every sample.
Modalities that are unavailable (dropped by modality dropout, or genuinely
absent like HSQC for acyclic molecules) are represented by a zero tensor + an
all-True mask. The `available` boolean tensor communicates to the model which
modalities contain real data. This makes `spectro_collate` a simple `torch.stack`
across the batch with no conditional padding logic.

**Augmentations applied during training only (never val/test):**
- IR: Gaussian noise (SNR 40–60 dB) + random polynomial baseline drift (degree 1–3)
- 1H-NMR: random ±0.0–0.1 ppm shift per spectrum; 20% probability of dropping
  low-integration peaks (< 0.1)
- 13C-NMR: random ±0–2 ppm uniform shift per spectrum
- MS/MS: 50% probability of dropping peaks with relative intensity < 5%;
  random ±0.01 Da m/z perturbation

**Rare-FG oversampling:** `WeightedRandomSampler` gives 5× weight to any
training molecule containing nitro, boronic_acid, phosphate, lactone, or
anhydride. This requires that the metadata parquet has a `functional_groups`
column. If it does not (Phase A may not have saved this), the sampler falls
back to uniform sampling silently.

---

### `model/train.py` — Stage 1 training loop

The `train()` function runs the full Stage 1 pretraining with:
- Noam LR schedule (warmup 4,000 steps, then inverse-sqrt decay)
- bf16 autocast via `torch.autocast` (no GradScaler — bf16 does not need gradient scaling)
- Gradient accumulation (`accumulation_steps` physical batches before each optimizer step)
- Gradient clipping at norm 1.0
- Checkpoint saving every 5,000 steps (keeps best 5 by Top-1 accuracy)
- Evaluation every 2,000 steps (greedy decoding on 200 val batches)
- Early stopping with patience 20,000 steps, minimum 100,000 steps

Run directly as a module:

```bash
# python -m model.train --physical-batch 512 --accum-steps 8
py -3.11 -m model.train --physical-batch 4 --accum-steps 32 --workers 8 --device cuda --no-compile
```

---

### `smoke_test.py` — Shape and gradient verification

Creates random tensors matching every encoder's expected input format and runs
them through the full model. Checks output shapes, confirms all parameters
receive gradients, tests the missing-modality path, and runs beam search.
No real data, no GPU required (though it will use GPU if available).

> **If you have an old smoke_test.py** that passes `(B, 6, T, 2)` tensors to
> `MSMSEncoder`, update it to pass `(B, 3, T, 2)` — one polarity at a time.

---

## 6. Required Manual Changes

These are the places in the code where you **must verify or change** things to
match your specific setup.

---

### Change 1 — Verify column names for spectral arrays

Run the data check from Section 2 Step 3. The column names in your chunk files
for the dense spectral arrays must match what `dataset.py` expects. Check:

```python
import pandas as pd
from pathlib import Path

meta = pd.read_parquet("audit_outputs/dataset_metadata_with_chunks.parquet")
chunk_path = Path(meta["chunk_file"].iloc[0])   # absolute path — use directly
chunk = pd.read_parquet(chunk_path)
print(chunk.columns.tolist())
```

`dataset.py` currently expects the column name `"ir_spectra"`. If your chunk
files use a different name (e.g., `"ir"` or `"IR_spectrum"`), update this line
in `dataset.py` inside `__getitem__`:

```python
raw_ir = spectral.get("ir_spectra", None)   # ← change "ir_spectra" if needed
```

Note: the peak list columns (`h_nmr_peaks`, `c_nmr_peaks`, `hsqc_nmr_peaks`,
`msms_positive_*`, `msms_negative_*`) are read from the **metadata parquet**,
not from the chunk files. Those names are already verified correct by Phase A.

---

### Change 2 — Verify 1H-NMR peak field names

In `dataset.py`, the `tokenize_h_nmr` function reads peak dictionaries with
these field names: `start`/`ppm_start`, `end`/`ppm_end`, `multiplicity`,
`n_hydrogen`/`nH`, `integration`.

Print one actual 1H-NMR peak from your dataset to confirm field names match:

```python
import pandas as pd, ast
from pathlib import Path

meta = pd.read_parquet("audit_outputs/dataset_metadata_with_chunks.parquet")

# Get first sample
chunk_path = Path(meta["chunk_file"].iloc[0])
row_idx = int(meta["chunk_row_idx"].iloc[0])

chunk = pd.read_parquet(chunk_path)

peaks = chunk.iloc[row_idx]["h_nmr_peaks"]

# If stored as string → parse
if isinstance(peaks, str):
    peaks = ast.literal_eval(peaks)

print(peaks[0])
```

If the field names differ, update the `.get()` calls in `tokenize_h_nmr()`.

---

### Change 3 — Verify MS/MS peak field names

Same check for MS/MS peaks. The code expects `mz` or `m/z` and `intensity` or
`relative_intensity`. Print one MS/MS peak to confirm:

```python
import pandas as pd, ast
from pathlib import Path

meta = pd.read_parquet("audit_outputs/dataset_metadata_with_chunks.parquet")

# Get first sample
chunk_path = Path(meta["chunk_file"].iloc[0])
row_idx = int(meta["chunk_row_idx"].iloc[0])

chunk = pd.read_parquet(chunk_path)

# Pick one MS/MS mode (e.g., positive 10 eV)
peaks = chunk.iloc[row_idx]["msms_positive_10ev"]

# Parse if string
if isinstance(peaks, str):
    peaks = ast.literal_eval(peaks)

print(peaks[0])
```

Update `tokenize_ms()` in `dataset.py` if field names differ.

---

### Change 4 — Check whether the metadata parquet has a `selfies` column

The dataset currently reads `row["selfies"]` to get the SELFIES string. Phase A
step A4 performed the SELFIES conversion, but depending on how A13 saved the
parquet, the column may be named differently or may be absent.

Check:

```python
meta = pd.read_parquet("audit_outputs/dataset_metadata_with_chunks.parquet")
print(meta.columns.tolist())
```

Your Phase A terminal output showed these columns:
`['canonical_smiles', 'selfies', 'molecular_formula', 'scaffold', 'split',
'smiles_len', 'selfies_len', 'n_heavy', 'chunk_file', 'chunk_row_idx']`

The `selfies` column is present — no action needed. If for any reason it were
absent, you could add a conversion step in `__getitem__`:

```python
import selfies as sf
selfies_str = sf.encoder(row["canonical_smiles"])
```

---

### Change 5 — Update `data_root` and paths in `config.py` if needed

The default paths in `SpectroConfig` are relative to `project_root/`. If your
dataset directory has a different name, update these fields:

```python
# In config.py:
data_root:      Path = Path("multimodal_spectroscopic_dataset")  # ← your dataset folder name
output_dir:     Path = Path("outputs")
vocab_path:     Path = Path("audit_outputs/selfies_vocab.json")
metadata_path:  Path = Path("audit_outputs/dataset_metadata_with_chunks.parquet")
```

Your Phase A output shows the data lives at:
`D:\AU - Ahmedabad Uni\sem2\DL\DL projec\multimodal_spectroscopic_dataset`

The `chunk_file` column in the metadata already stores absolute paths, so
`data_root` is not used for chunk loading — it is only used if you add any
code that constructs paths from it. The relative paths above for `vocab_path`
and `metadata_path` work as long as you run from `project_root/`.

---

### Change 6 — wandb project name (optional but recommended)

In `train.py` at the bottom (the `__main__` block), change the wandb project
and run name to something meaningful for your experiment tracking:

```python
wandb.init(
    project="spectro-elucidation",          # ← matches your existing wandb project
    name="stage1_nmr_only",                 # ← change per curriculum stage
    config=cfg.__dict__,
)
```

Your Phase A already logged to `dev-p14-ahmedabad-university/spectro-elucidation`,
so keep the project name consistent.

---

## 7. GPU Memory Reference

Effective batch size must be 4,096 tokens in all cases. Adjust `--physical-batch`
and `--accum-steps` so their product = 4,096.

| GPU | VRAM | `--physical-batch` | `--accum-steps` | Effective batch |
|---|---|---|---|---|
| A100 80GB | 80 GB | 512 | 8 | 4,096 |
| A6000 48GB | 48 GB | 256 | 16 | 4,096 |
| RTX 4090 24GB | 24 GB | 128 | 32 | 4,096 |
| RTX 3090 24GB | 24 GB | 64 | 64 | 4,096 |

If you get OOM errors, halve `--physical-batch` and double `--accum-steps`.
bf16 is already enabled by default — this halves memory vs fp32.

---

## 8. Expected Training Milestones

These are approximate. Your actual numbers will depend on GPU and dataset IO speed.

| Curriculum Stage | Steps | Expected Val Top-1 | Action |
|---|---|---|---|
| NMR-only start | 0–10k | 5–15% | Normal warmup, loss should be falling |
| NMR-only mid | 10k–60k | 20–50% | Steady improvement expected |
| NMR-only target | ~100k | **≥ 60%** | Must hit this before adding MS/MS |
| Add MS/MS | 100k–120k | Slight dip then recovery | Expected — new modality disrupts briefly |
| Add IR | 120k–150k | Gradual improvement | IR alone is weak; adds marginal signal |
| Full multimodal | 150k–300k | **≥ 75%** target | All 6 modalities + modality dropout |

If NMR-only Top-1 is below 40% at step 60,000, stop and check: learning rate
schedule, scaffold split implementation, SELFIES tokenization, and data loading.
Do not add more modalities until NMR is working.

---

## 9. Dependencies

```bash
py -3.11 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers selfies rdkit wandb pandas pyarrow numpy tqdm
```

Python 3.10+ required (the code uses `X | Y` union type hints).

Optional for faster chunk loading:
```bash
pip install fastparquet   # faster parquet IO than pyarrow for random row access
```

Verify the install:
```bash
py -3.11 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
py -3.11 -c "import selfies; print(selfies.__version__)"
py -3.11 -c "from rdkit import Chem; print('RDKit OK')"
```
