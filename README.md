# CPRE Fluoroscopy Image Classification — Snapshot Ensemble

Automatic classification of fluoroscopic CPRE (Endoscopic Retrograde Cholangiopancreatography) images using a Snapshot Ensemble of EfficientNet models, achieving **F1-macro 0.9012** and **AUC-ROC 0.9804** on the test set.

> Academic project · Aprendizagem Profunda (AP) · University of Minho · 2025/2026

---

## Results

| Model | F1-macro (Test) |
|---|---|
| Baseline (literature) | 0.7380 |
| EfficientNet-B5 (seed 42) | 0.8841 |
| EfficientNet-B5 (seed 7) | 0.8723 |
| tf_EfficientNet-B7 | 0.8589 |
| EfficientNet-B6 | 0.8061 |
| **Snapshot Ensemble + TTA** | **0.9012** |

**AUC-ROC (macro):** 0.9804 · **Accuracy:** 89.8% · **Δ vs baseline:** +0.1632

---

## Dataset

[MIQR-CC Dataset](https://figshare.com/articles/dataset/MIQR-CC_Dataset/31079236) — 1 568 fluoroscopic CPRE images across 4 classes:

| Class | Total | % |
|---|---|---|
| Lithiasis | 726 | 46.3% |
| Stricture | 392 | 25.0% |
| Normal | 299 | 19.1% |
| Biliary Leaks | 151 | 9.6% |

Split: 70% train / 15% val / 15% test. Class imbalance handled via `WeightedRandomSampler` and weighted `CrossEntropyLoss`.

---

## Approach

### Ensemble
Four EfficientNet models trained independently and combined via weighted averaging:

| Model | Epochs | Seed | Ensemble Weight |
|---|---|---|---|
| EfficientNet-B5 | 120 | 42 | 0.30 |
| EfficientNet-B5 | 120 | 7 | 0.30 |
| tf_EfficientNet-B7 | 80 | 42 | 0.05 |
| EfficientNet-B6 | 80 | 42 | 0.35 |

Weights tuned via grid search on the validation set.

### Key Techniques
- **CLAHE preprocessing** — adaptive histogram equalisation to enhance ductal structures
- **Two-phase fine-tuning** — frozen backbone initially, then full unfreezing with lower LR
- **Test-Time Augmentation (TTA)** — multi-crop inference for more robust predictions
- **Grad-CAM** — visual interpretability of model attention regions
- **Overfitting analysis** — all four models show Train-Val F1 gap > 0.08; ensemble generalises better than any individual model

### Hyperparameters
| Parameter | Value | Rationale |
|---|---|---|
| `IMG_SIZE` | 384 | Optimal for EfficientNet B5–B7; preserves ductal detail |
| `BATCH_SIZE` | 16 | Balance between GPU memory (T4 16GB) and gradient stability |
| Loss | CrossEntropyLoss + class weights | Compensates for class imbalance |

---

## Repository Structure

```
├── trabalhoap.ipynb     # Full pipeline: EDA → training → ensemble → Grad-CAM
├── requirements.txt     # Python dependencies
├── AP_Relatório.pdf     # Full project report (Portuguese)
└── README.md
```

---

## How to Run

**Environment:** Kaggle (GPU T4 x2, Python 3.10+) — recommended for reproducibility.

**1. Download the dataset**

Download the [MIQR-CC Dataset](https://figshare.com/articles/dataset/MIQR-CC_Dataset/31079236) and update the paths in the notebook:

```python
BASE_DIR    = '/kaggle/working'
DATASET_DIR = '/path/to/MIQR-CC-Dataset'
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Run the notebook**

Open `trabalhoap.ipynb` and run all cells in order. Training the full ensemble takes approximately 8–10 hours on dual T4 GPUs.

---

## Error Analysis

On the 236-image test set, 24 misclassifications (10.2%):

| Real → Predicted | Count | Clinical Risk |
|---|---|---|
| Normal → Lithiasis | 7 | Low |
| Stricture → Lithiasis | 7 | **Critical** |
| Lithiasis → Normal | 5 | Low |
| Biliary Leaks → Normal | 2 | **Critical** |
| Others | 3 | — |

Stricture and Biliary Leaks false negatives represent the highest clinical risk and are the primary area for future improvement.

---

## Report

The full project report is available at [`AP_Relatório.pdf`](./AP_Relatório.pdf).

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)

`timm` · `pytorch-grad-cam` · `opencv` · `scikit-learn` · `matplotlib`
