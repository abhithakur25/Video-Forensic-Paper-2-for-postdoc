# Paper 2 — implementation run report

Generated 2026-08-04 on Windows 11, conda env `VideoForgeryCPU` (Python 3.8.20),
CPU only.

Project: **"Optimized Mixed Attention-based Bidirectional Gradient Model for
Intra-frame Video Forgery Detection"** (OM²AHL-BiG). Folder:
`CODE_05-08-2025(Paper2)`.

---

## 1. Dataset — features already present

**FaceForensics++** is gated (author form); raw videos are not in the repo.
Evaluation does **not** need them: `Features/Features.pkl` already holds
pre-extracted features, and `ReadDataset(exec=False)` loads them.

| key | shape | meaning |
|---|---|---|
| `features` | (199, 15, 32, 32, 10) | fused GradCAM / ResNet-stat / VGG-SIFT / shape-ResNet tensors |
| `labels` | (199,) | **99 authentic (0) / 100 forged (1)** |

Raw videos would only be required to re-extract features (`Main.py` → "Yes",
`exec=True`), which expects:

```
DATASET/manipulated_sequences/FaceSwap/c23/videos/*.mp4
DATASET/original_sequences/youtube/c23/videos/*.mp4
```

---

## 2. What was run

```powershell
$E = "C:\Users\USER\anaconda3\envs\VideoForgeryCPU"
$env:PATH = "$E\Library\bin;$E;$E\Scripts;" + $env:PATH
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
cd "C:\Users\USER\Downloads\PostDoc\CODE_05-08-2025(Paper2)"
$d = ".claude\skills\run-video-forgery-paper2\driver.py"

& "$E\python.exe" -u $d check
& "$E\python.exe" -u $d make-video
& "$E\python.exe" -u $d plots
& "$E\python.exe" -u $d gui
& "$E\python.exe" -u $d evaluate --epochs 3 --skip-opt
```

| Output | Location |
|---|---|
| Analysis figures (TP + KF bar/line + ROC) | `Results/TP/`, `Results/KF/`, `Results/RocAnalysis/` |
| GUI screenshots (7 PNG) | `driver_out/screenshots/` |
| Evaluation table | `driver_out/evaluation_tp_ep3.txt` |
| Fresh proposed-model metrics | `Analysis1/TP/COM_A_eval.npy` |
| Synthetic GUI clip | `driver_out/sample.mp4` |

Published `Analysis/` arrays were **not** overwritten (paper figures stay
reproducible via `plots`).

---

## 3. Evaluation results (smoke-scale)

Comparative analysis over training percentage 40–90 %, proposed model only
(`Network.BiLSTMGBM` / OM²AHL-BiG). **3 epochs per incremental chunk, CoSH
weight optimization skipped.** Paper uses 500 epochs + CoSH (mealpy HYBRID,
pop=50, epoch=10).

### Accuracy / Sensitivity / Specificity / Precision / F1

Measured 2026-08-04 (3 epochs / chunk, CoSH skipped). Source:
`driver_out/evaluation_tp_ep3.txt`.

| TP% | Accuracy | Sens | Spec | Prec | F1 |
|---|---|---|---|---|---|
| 40 | 0.9500 | 0.9333 | 0.9667 | 0.9655 | 0.9492 |
| 50 | 0.9100 | 0.9000 | 0.9200 | 0.9184 | 0.9091 |
| 60 | 0.8375 | 0.8000 | 0.8750 | 0.8649 | 0.8312 |
| 70 | 0.8833 | 0.9000 | 0.8667 | 0.8710 | 0.8852 |
| 80 | 0.9500 | 0.9500 | 0.9500 | 0.9500 | 0.9500 |
| 90 | *(final split — see evaluation file if present)* | | | | |

Mean accuracy over completed splits ≈ **0.906** (smoke scale).

### How to read this

**This is a smoke-scale run, not a reproduction, and it neither confirms nor
refutes the paper.** Reasons:

1. **3 epochs vs 500**, and **CoSH optimization disabled** (full opt is ~1000
   `model.predict` calls per split).
2. Incremental learning uses **5 cumulative chunks**, so wall-clock grows with
   training size even at low epochs.
3. The channel-attention branch collapses to a `Dense` with 0 units when the
   last dimension is 1 (visible in `model.summary()` as `dense (None, 1, 1, 0)`).
   The graph still trains; the attention path is partly degenerate.

To attempt a fuller reproduction:

```powershell
& "$E\python.exe" -u $d evaluate --epochs 500 --with-opt
```

Expect multi-day runtime on CPU.

---

## 4. Code findings

### 4.1 Torch must import before SciPy in this env

`SubFunctions/Attention.py` imports `torch` (used by `mutual_attention`).
Conda MKL (`libiomp5md.dll`) and torch’s OpenMP conflict if SciPy loads first.
Fix used by the driver: `import torch` first + `KMP_DUPLICATE_LIB_OK=TRUE`.

### 4.2 Paper 2 comparative baselines are external

`SubFunctions/Model.py` only implements `BiLSTMGBM` (proposed). Baseline
curves (EfficientNet … SMA-CLMPNet) are loaded from `ResultsP1/` CSVs inside
`VisualizeResults.LoadP1` and stacked with the proposed `Analysis/TP/COM_A.npy`.

### 4.3 IncrementalLearning hard-codes 5 chunks

`split_list` indexes `split_lists[0]..[4]` unconditionally. Changing
`number_of_chunks` without editing that method raises `IndexError`.

---

## 5. Published paper numbers (source arrays)

From `Analysis/TP/COM_A.npy` (used by the paper figures / Tables 4–6):

| TP% | Accuracy | Sens | Spec | Prec | F1 |
|---|---|---|---|---|---|
| 40 | 0.9379 | 0.9357 | 0.9390 | 0.9401 | 0.9379 |
| 50 | 0.9484 | 0.9583 | 0.9434 | 0.9385 | 0.9483 |
| 60 | 0.9693 | 0.9600 | 0.9740 | 0.9787 | 0.9693 |
| 70 | 0.9734 | 0.9691 | 0.9755 | 0.9777 | 0.9734 |
| 80 | 0.9827 | 0.9804 | 0.9838 | 0.9850 | 0.9827 |
| 90 | 0.9856 | 0.9791 | 0.9888 | 0.9920 | 0.9855 |

Doc Table 6 (TP 90% / K-Fold 10) reports OM²AHL-BiG accuracy **98.62%** (TP)
and **97.28%** (KF-10).
