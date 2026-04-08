# 🧪 Multimodal Molecular Structure Prediction  
### From Spectral Data → Molecular Structure (SELFIES / SMILES)

---

## 🚀 Overview

This project builds a **multimodal deep learning system** that predicts the molecular structure of organic compounds from spectroscopic data.

Given multiple spectral inputs:
- IR (Infrared Spectroscopy)
- ¹H-NMR (Proton NMR)
- ¹³C-NMR (Carbon NMR)
- HSQC (2D NMR)
- MS/MS (Mass Spectrometry)

👉 The model outputs a **SELFIES sequence**, which is converted into a valid molecular structure (SMILES).

---

## 🎯 Problem Statement

In chemistry, determining a molecule’s structure from spectral data is:
- Time-consuming  
- Expert-driven  
- Error-prone  

This project automates the process using **transformer-based deep learning**, enabling:
- Fast predictions  
- Scalable analysis  
- Robust handling of incomplete data  

---

## 🧠 Model Architecture

The system is a **multimodal encoder-decoder transformer**:

### 🔹 Encoders (6 modalities)
Each modality is processed using a specialized encoder:

| Modality | Input Type | Encoding Strategy |
|----------|-----------|------------------|
| IR | 1800-length signal | Patch-based transformer (ViT-style) |
| ¹H-NMR | Peak list | Transformer encoder |
| ¹³C-NMR | Peak list | Shared encoder with modality embedding |
| HSQC | 2D peak list | Transformer encoder |
| MS/MS+ | Peak lists (3 energies) | Transformer encoder |
| MS/MS- | Peak lists (3 energies) | Transformer encoder |

---

### 🔹 Cross-Modal Fusion
- Uses **6 learnable CLS tokens (one per modality)**
- Each token attends to **all modality outputs**
- Produces **6 compact summary vectors (B, 6, 512)**

---

### 🔹 Decoder
- Autoregressive transformer (4 layers)
- Generates **SELFIES tokens**
- Vocabulary size: **111 tokens**
- Max sequence length: **82**

---

## 📊 Dataset

- **Total molecules:** 789,255  
- **Train / Val / Test:** 90% / 5% / 5%  
- **Modalities:** 6  
- **Storage:** 245 parquet chunk files  

### Key Features:
- Canonical SMILES + SELFIES conversion  
- Scaffold-based splitting (ensures generalization)  
- Functional group analysis  
- Rare class imbalance handling  

---

## ⚙️ Training Strategy

| Component | Value |
|----------|------|
| Effective batch size | 4096 |
| Physical batch size | 512 |
| Gradient accumulation | 8 steps |
| Precision | BF16 |
| Optimizer | Adam |
| Scheduler | Noam (warmup + decay) |

---

### 🧪 Techniques Used

- Label smoothing (0.1)  
- Gradient clipping (norm = 1.0)  
- Modality dropout (30%)  
- Rare functional group oversampling (×5)  
- Mixed precision training (BF16)  

---

## 📈 Evaluation Metrics

| Metric | Description |
|--------|------------|
| Top-1 Accuracy | Exact molecule match |
| Validity Rate | % of valid generated molecules (~100%) |
| Tanimoto Similarity | Structural similarity (0–1 range) |

---

## 🏗️ Data Pipeline

- Lazy loading from parquet chunks  
- LRU cache for efficient disk access  
- On-the-fly data augmentation:
  - IR noise + baseline drift  
  - NMR peak shifts  
  - MS noise & dropout  

---

## 📁 Project Structure
├── config.py # Model + training configuration
├── encoders.py # All modality encoders
├── model.py # Fusion + full model
├── dataset.py # Data loading + augmentation
├── train.py # Training loop
├── selfies_vocab.json # Token vocabulary
├── audit_outputs/ # Dataset metadata


---

## 🔬 Key Innovations

- ✅ Multimodal learning across **6 spectroscopy types**
- ✅ Efficient **cross-modal fusion using CLS tokens**
- ✅ Robust to **missing modalities**
- ✅ Scalable to **~800k molecules**
- ✅ Uses **SELFIES for guaranteed valid outputs**

---

## 🧩 Challenges Solved

- Handling heterogeneous data (signals + peak lists)  
- Memory-efficient training on large datasets  
- Class imbalance (rare functional groups)  
- Variable-length inputs across modalities  
- Missing data in real-world scenarios  

---

## 🔮 Future Work

- Add full multimodal training (Stage 2)  
- Improve Top-1 accuracy with larger models  
- Deploy as an API for chemists  
- Integrate real experimental datasets  

---

## 🛠️ Tech Stack

- Python  
- PyTorch  
- Transformers  
- RDKit  
- SELFIES  
- NumPy / Pandas  

---

## 🤝 Contributions

This is an individual deep learning project focused on:
- System design  
- Multimodal modeling  
- Large-scale training  

---

## 📌 Summary

This project demonstrates how **deep learning + transformers** can solve a complex real-world scientific problem:

> Predicting molecular structure from raw spectroscopic data.

---

## ⭐ If you found this useful

Give the repo a ⭐ and feel free to contribute!
