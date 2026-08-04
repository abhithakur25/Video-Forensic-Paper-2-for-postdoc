# Video Forensic Paper 2 for Postdoc

**Title:** Optimized Mixed Attention-based Bidirectional Gradient Model for Intra-frame Video Forgery Detection (**OM²AHL-BiG**)

**Repository:** private postdoc codebase for Paper 2 (`CODE_05-08-2025(Paper2)`).

This repository contains the full Python implementation, pre-extracted FaceForensics++ features, analysis arrays, result figures, a customtkinter GUI, and an agent/CLI **driver** that can check the environment, regenerate plots, walk the GUI, and re-run training/evaluation without interactive popups.

---

## 1. Project overview

Intra-frame video forgery (FaceSwap-style manipulations inside individual frames) is detected with a hybrid pipeline:

1. **Preprocessing** — adaptive key-frame selection + Viola–Jones ROI (face).
2. **Feature extraction** — Grad-CAM heatmaps, ResNet-101 statistical descriptors (mean / std / var / skew / kurtosis), hybrid VGG-16 + SIFT, shape-descriptor ResNet features.
3. **Classifier** — multilevel mixed attention + Bi-LSTM, hybrid incremental learning, CoSH (Coati–Sea Horse hybrid) weight refinement, final **Gradient Boosting Machine (GBM)** head.

Paper doc in this folder: `Neha Dhiman - Paper 2 (Premium) (1).docx`.

Run details and measured smoke-evaluation numbers: **`RUN_REPORT.md`**.

---

## 2. Repository layout (what each path is for)

| Path | Role |
|---|---|
| `Main.py` | Interactive entry: Yes = full train+plots (~48 h), No = plots from `Analysis/*.npy` |
| `GUI.py` | Desktop GUI (customtkinter): load video → preprocess → GradCAM → ResNet-stat → VGG/SIFT → shape/flow |
| `SubFunctions/GetData.py` | `ReadDataset`: extract features from `DATASET/` **or** load `Features/Features.pkl` |
| `SubFunctions/GetPreprocessing.py` | Key-frame + face ROI preprocessing |
| `SubFunctions/GetFeatures.py` | GradCAM / ResNet-stat / VGG-SIFT / shape-ResNet feature builders (loads ResNet101 + VGG16 at import) |
| `SubFunctions/GradCAM.py` | Grad-CAM helper used by GUI and features |
| `SubFunctions/Model.py` | **`Network.BiLSTMGBM`** — proposed OM²AHL-BiG model (~1.1 M params) |
| `SubFunctions/Attention.py` | Mutual cross-attention (torch), sparse self-attention, channel + zero attention (Keras) |
| `SubFunctions/IncrementalLearning.py` | 5-chunk cumulative incremental training sets |
| `SubFunctions/Optimization.py` | CoSH / mealpy `HYBRID` weight optimization after BiLSTM training |
| `SubFunctions/Analysis.py` | `TPAnalysis` (train % 40–90) and `KFAnalysis` (k = 6…10) loops |
| `SubFunctions/Evaluate.py` | Confusion-matrix metrics: ACC, SEN, SPE, PRE, F1 (+ TPR/FPR) |
| `SubFunctions/VisualizeResults.py` | Bar/line/ROC/statistical plots; stacks Paper-1 baselines from `ResultsP1/` |
| `mealpy/` | Vendored metaheuristic library (includes `Proposed.HYBRID` = CoSH) |
| `Features/Features.pkl` | Pre-extracted tensors `(199, 15, 32, 32, 10)` + labels |
| `Analysis/TP|KF/*.npy` | **Published** metrics used by paper figures |
| `Analysis1/` | Re-run outputs (does not overwrite `Analysis/`) |
| `Results/` | Generated figures + CSVs (comparative, performance, ROC, time, etc.) |
| `ResultsP1/` | Paper-1 comparative CSV baselines pulled into Paper-2 plots |
| `driver_out/` | Driver outputs: sample video, GUI screenshots, evaluation text tables |
| `.claude/skills/run-video-forgery-paper2/` | **`driver.py` + `SKILL.md`** — non-interactive automation |

---

## 3. Environment

### Hardware (paper)

- Windows 11, ≥16 GB RAM, ≥100 GB free disk, multi-core CPU.

### Software (verified working)

Use the conda env **`VideoForgeryCPU`** (Python **3.8.20**). Do **not** use a newer system Python (e.g. 3.14) — TensorFlow 2.10 / numpy 1.21 are not available there.

```powershell
$E = "C:\Users\USER\anaconda3\envs\VideoForgeryCPU"
$env:PATH = "$E\Library\bin;$E;$E\Scripts;" + $env:PATH
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
```

**`$E\Library\bin` on PATH is mandatory** — without it, `skimage`/`scipy.linalg` can hard-crash the interpreter (no traceback).

**`KMP_DUPLICATE_LIB_OK=TRUE` and `import torch` before SciPy** are required for the full-analysis path (`Attention.py` imports torch).

### Packages

```bash
pip install -r requirements.txt
```

Important pins / deviations:

| Package | Note |
|---|---|
| `tensorflow==2.10.0`, `keras==2.10.0` | Project target |
| `scikit-image==0.19.3` | Code uses `greycomatrix` / `greycoprops` (renamed in 0.20+) |
| `customtkinter==5.1.3` | 6.x API differs |
| `PySimpleGUI==4.60.5.1` | Free 4.x; `4.60.5` is gone from PyPI |
| `torch==1.13.1` | Needed only for mutual-attention weight path / full analysis |

First import of `GetFeatures` / `GUI` downloads ~733 MB of Keras weights (ResNet101 + VGG16) into `~/.keras/models/`.

---

## 4. How execution was created (step by step)

### Step A — Environment check

```powershell
cd <this-repo>
& "$E\python.exe" -u ".claude\skills\run-video-forgery-paper2\driver.py" check
```

**Files used:** `driver.py` → imports `numpy`, `scipy`, `cv2`, `tensorflow`, `keras`, `skimage`, `customtkinter`, `PySimpleGUI`, `peakutils`, `seaborn`, `sklearn`, `pandas`, `tqdm`, `PIL`; verifies vendored `mealpy/`; reports Keras weight cache.

Expected tail: `CHECK OK`.

### Step B — Synthetic GUI video (no FaceForensics++ videos in repo)

```powershell
& "$E\python.exe" -u ".claude\skills\run-video-forgery-paper2\driver.py" make-video
```

**Files used:** sample frames under `Results/ImageResults/Input/*.jpg` → OpenCV `VideoWriter` → `driver_out/sample.mp4`.

### Step C — Regenerate analysis figures from published arrays

```powershell
& "$E\python.exe" -u ".claude\skills\run-video-forgery-paper2\driver.py" plots
```

**Files used:**

- Input metrics: `Analysis/TP/*.npy`, `Analysis/KF/*.npy`, `ResultsP1/**/Comp_Analysis/Bar/*.csv`
- Code: `SubFunctions/VisualizeResults.py` → class `PlotResults(show=False, save=True)`
- Output: PNGs + CSVs under `Results/TP/`, `Results/KF/`, `Results/RocAnalysis/`

This is exactly `Main.py`’s **"No"** branch (plots only), without the blocking `popup_yes_no` and without `plt.show()`.

### Step D — Drive the GUI stage by stage + screenshots

```powershell
& "$E\python.exe" -u ".claude\skills\run-video-forgery-paper2\driver.py" gui
```

**Files used:**

| Stage | GUI method | Primary code |
|---|---|---|
| Select Video | `select_data_event` | `GUI.py` (file dialog patched to `driver_out/sample.mp4`) |
| Preprocessing | `preprocessing_event` | `GetPreprocessing` + Haar cascade `Temp/haarcascade_frontalface_alt2.xml` |
| GradCAM | `get_gradcam` | `GradCAM.py` |
| ResNet statistical | `get_resnetstat` | ResNet101 + mean/var/std/skew/kurtosis |
| VGG / SIFT | `get_vgg` | VGG16 + SIFT |
| Shape / flow | `get_flow` | Shape descriptor + ResNet101 |

Screenshots: `driver_out/screenshots/01-launched.png` … `07-flow.png`.

### Step E — Re-train and evaluate the proposed model (smoke scale)

```powershell
& "$E\python.exe" -u ".claude\skills\run-video-forgery-paper2\driver.py" evaluate --epochs 3 --skip-opt
```

**Files used (execution chain):**

1. `SubFunctions/GetData.py` → `ReadDataset(exec=False)` loads `Features/Features.pkl`
2. `SubFunctions/Analysis.py` → `train_test_split` (class-balanced split at TP = 0.4…0.9)
3. `SubFunctions/Model.py` → `Network.BiLSTMGBM`:
   - reshape features to `(N, 15*32, 32*10)` = `(N, 480, 320)`
   - BiLSTM(100) → BiLSTM(128) → multilevel attention → mixed attention → BiLSTM(128) → Dense → softmax
   - `IncrementalLearning` (5 cumulative chunks) × `epochs` each
   - optional CoSH (`Optimization.py` + `mealpy/Proposed.py`) — **skipped** with `--skip-opt`
   - feature layer → `GradientBoostingClassifier(n_estimators=100)`
4. `SubFunctions/Evaluate.py` → `Evaluation_Metrics` → ACC, SEN, SPE, PRE, F1
5. Saves `Analysis1/TP/COM_A_eval.npy` + `driver_out/evaluation_tp_ep3.txt`

Optional K-fold:

```powershell
& "$E\python.exe" -u ".claude\skills\run-video-forgery-paper2\driver.py" evaluate --kfold --epochs 3 --folds-per-k 2 --skip-opt
```

### Step F — Full paper pipeline (human / long run)

```powershell
& "$E\python.exe" Main.py
# Popup "Yes" → full Comparative + Performance + ROC + KF (paper: epochs=500, ~48 h)
# Popup "No"  → plots only from Analysis/
```

Requires FaceForensics++ under `DATASET/` only if you choose re-extraction (`exec=True`).

---

## 5. Driver command summary

```text
driver.py check          # env + imports + weight cache
driver.py make-video     # driver_out/sample.mp4
driver.py plots          # regenerate Results/ figures headlessly
driver.py gui            # 6-stage GUI walkthrough + screenshots
driver.py evaluate       # BiLSTMGBM TP sweep or --kfold
driver.py all            # check → make-video → plots → gui
```

Always pass `-u` (unbuffered) so crashes still flush logs.

---

## 6. Dataset

- **FaceForensics++** FaceSwap @ c23 + original youtube @ c23  
  Request access: https://www.niessnerlab.org/projects/roessler2019faceforensicspp.html  
- Shipped features: **199 videos**, balanced ~99/100 authentic/forged.

---

## 7. Metrics (paper-side published snapshot)

Source: `Analysis/TP/COM_A.npy` / doc Tables 4–6 (OM²AHL-BiG).

| Setting | Accuracy (reported) |
|---|---|
| Training percentage 90% (best) | ~98.3–98.6% |
| K-fold = 10 (best) | ~97.3–97.7% |

Smoke-scale re-runs (3 epochs, no CoSH) are intentionally lower and are documented in **`RUN_REPORT.md`** / `driver_out/evaluation_tp_ep3.txt`. They validate that the pipeline **executes end-to-end**, not that paper numbers are reproduced.

---

## 8. Large files

| File | ~Size | Notes |
|---|---|---|
| `Features/Features.pkl` | ~233 MB | Required for evaluation without raw videos (Git LFS recommended) |
| `Results/ImageResults/` | ~231 MB | Sample frames for paper figures / GUI make-video |
| `Results/Results.rar` | ~212 MB | Archive duplicate — gitignored |

---

## 9. Citation / paper

See `Neha Dhiman - Paper 2 (Premium) (1).docx` for the full manuscript (abstract, system model, OM²AHL-BiG, CoSH, results Tables 4–6, ROC, confusion matrix, convergence, compute time).

---

## 10. Web application (Render)

Browser demo of the pipeline (ForensiQ-style dark UI):

| Path | Role |
|---|---|
| `webapp/app.py` | Flask API + stage pipeline (OpenCV, Render-safe) |
| `webapp/templates/index.html` | Detection / Results / About pages |
| `webapp/render.yaml` | Render.com service definition |
| `webapp/requirements.txt` | Lightweight deps (no TensorFlow) |

```powershell
cd webapp
pip install -r requirements.txt
python app.py
# http://127.0.0.1:8080
```

On Render: root directory `webapp`, start `gunicorn app:app`.  
Full BiLSTM-GBM training stays offline (`driver.py evaluate`); the web app visualizes the same stages and shows paper + smoke metrics.

Design references used for the UI: ForensiQ detector on Render and AT SCHOOL.IN project hub (dark cards, drop-zone upload, blue accent, results tables).

## 11. License / access

This repository is **private**. Redistribute only with the owner’s permission. FaceForensics++ remains under its own license and access policy.
