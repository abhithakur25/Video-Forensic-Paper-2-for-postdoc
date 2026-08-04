---
name: run-video-forgery-paper2
description: Build, run, and drive the Paper 2 video forgery detection project (intra-frame forgery, multilevel attention Bi-LSTM/GBM). Use when asked to start or launch this app, run the GUI, regenerate the analysis plots/figures, take a screenshot of the UI, verify a change works in the real app, or smoke-test the feature-extraction pipeline.
---

Paper 2 ("Intra-frame video forgery detection using multilevel attention enabled
hybrid Bi-LSTM GBM") is a Windows Python 3.8 research codebase with two surfaces:
a customtkinter desktop GUI (`GUI.py`) and a plot-regeneration script (`Main.py`).
**Both block on modal prompts and cannot be driven from a shell.** Drive them with
`.claude/skills/run-video-forgery-paper2/driver.py`, which bypasses the prompts,
walks the GUI stage by stage, and writes PNG screenshots.

All paths below are relative to `CODE_05-08-2025(Paper2)/`.
This is Windows + PowerShell, not Linux — there is no xvfb and none is needed;
the GUI renders on the real desktop.

## Prerequisites

Use the existing conda env `VideoForgeryCPU` (Python 3.8.20) — it already matches
`requirements.txt`. Every command below assumes these two lines first:

```powershell
$E = "C:\Users\USER\anaconda3\envs\VideoForgeryCPU"
$env:PATH = "$E\Library\bin;$E;$E\Scripts;" + $env:PATH
```

**The `$E\Library\bin` entry is mandatory, not cosmetic** — see Gotchas.
`conda activate VideoForgeryCPU` sets the same thing and works too.

The interpreter is `$E\python.exe`. Do **not** use the `python` on PATH: that is
3.14.6, and neither TensorFlow 2.10 nor numpy 1.21.6 exists for it.

### Rebuilding the env from scratch

> Unlike everything else in this file, this block was **not executed** in the
> session that wrote it — this network's TLS interception broke both `pip` and
> `conda` downloads partway through (see Troubleshooting). It is reconstructed
> from the versions actually present in the working env, so treat the pins as
> verified and the commands as untested.

```powershell
conda create -n VideoForgeryCPU python=3.8 -y
conda activate VideoForgeryCPU
conda install -y numpy=1.21.6 scipy=1.7.3 pandas=1.3.4 matplotlib=3.5.3 `
    seaborn=0.12.2 scikit-learn=1.0.2 scikit-image=0.19.3 -c conda-forge
pip install tensorflow==2.10.0 keras==2.10.0 opencv-python==4.8.0.76 `
    Pillow==9.5.0 termcolor peakutils==1.3.5 tqdm customtkinter==5.1.3 `
    "PySimpleGUI==4.60.5.1"
```

Deviations from `requirements.txt` that are required, not optional:

- `PySimpleGUI==4.60.5` **does not install** — it was pulled from PyPI when the
  project relicensed. Use `4.60.5.1`, the last free 4.x. It provides
  `popup_yes_no`, which is all `Main.py` uses.
- `pip install customtkinter` (unpinned) gives 6.0.0, whose API differs. Pin `5.1.3`.
- `scikit-image` must stay on 0.19.x: the code calls `greycomatrix`/`greycoprops`
  (`SubFunctions/GetFeatures.py:16`), renamed to `gray*` in 0.20 and removed later.
- `torch==1.13.1` is in `requirements.txt` but **cannot be installed usefully here**
  — see Gotchas. Nothing in this skill's paths needs it.

## Run (agent path)

Everything goes through the driver. Run it from the project root:

```powershell
$E = "C:\Users\USER\anaconda3\envs\VideoForgeryCPU"
$env:PATH = "$E\Library\bin;$E;$E\Scripts;" + $env:PATH
cd "C:\Users\USER\Downloads\PostDoc\CODE_05-08-2025(Paper2)"
& "$E\python.exe" -u ".claude\skills\run-video-forgery-paper2\driver.py" check
```

Always pass `-u`. Without it a hard crash (see Gotchas) discards buffered stdout
and the failure looks like silence.

`check` verifies the interpreter, every import, the vendored `mealpy`, and the
keras weight cache. Expected tail:

```
[driver]   torch UNAVAILABLE (OSError) - expected in this env; only the full-analysis path needs it
[driver]   vendored ./mealpy present
[driver]   cached resnet101_weights_tf_dim_ordering_tf_kernels.h5 (180 MB)
[driver]   cached vgg16_weights_tf_dim_ordering_tf_kernels.h5 (553 MB)
[driver] CHECK OK
```

### Drive the GUI and take screenshots

```powershell
& "$E\python.exe" -u ".claude\skills\run-video-forgery-paper2\driver.py" make-video
& "$E\python.exe" -u ".claude\skills\run-video-forgery-paper2\driver.py" gui
```

`make-video` is required once — neither repo ships a `DATASET/` or any video, and
the GUI's first button needs one. It builds `driver_out/sample.mp4` from the 75
sample frames in `Results/ImageResults/Input/`.

`gui` builds the real `App()`, then calls each handler in order, pumping Tk events
and screenshotting after each. Screenshots land in `driver_out/screenshots/`:

| file | stage | `App` method |
|---|---|---|
| `01-launched.png` | window as opened | — |
| `02-select.png` | Select Video | `select_data_event` |
| `03-preprocess.png` | Preprocessing (haar face ROI) | `preprocessing_event` |
| `04-gradcam.png` | GradCAM heatmap | `get_gradcam` |
| `05-resnet.png` | ResNet + mean/var/std/skew/kurtosis | `get_resnetstat` |
| `06-vgg.png` | SIFT + VGG16 | `get_vgg` |
| `07-flow.png` | Shape descriptor + ResNet101 | `get_flow` |

Whole run is ~25 s once weights are cached. Subsets: `--stages select,preprocess,gradcam`.
Another video: `--video path\to\clip.mp4`.

**Look at the PNGs afterwards.** A stage can "pass" and still render nothing.

### Regenerate the analysis figures

```powershell
& "$E\python.exe" -u ".claude\skills\run-video-forgery-paper2\driver.py" plots
```

This is `Main.py`'s "No" branch (plots from pre-evaluated `Analysis/*.npy`) with
the modal prompt removed and `matplotlib` forced to Agg. Writes **41** figures
plus CSVs under `Results/` (TP + KF comparative/performance bar and line graphs,
and `Results/RocAnalysis/Graph_roc.png`). Verified output: a 9-method ROC plot
(EfficientNet … OM²AHL-BiG). Timing varies — observed 57 s here and up to ~2 min
on the sibling project; allow a couple of minutes.

`check`, `make-video`, `plots` then `gui` in one go: `driver.py all`.

## Run (human path)

```powershell
& "$E\python.exe" GUI.py     # opens the window; drive it by hand, close to exit
& "$E\python.exe" Main.py    # modal Yes/No popup, then ~40 blocking plt.show() windows
```

Both are interactive-only and unusable for automation. `Main.py`'s "Yes" branch is
the full 48-hour training run; it needs a `DATASET/` (FaceForensics++) that is not
in the repo, and a working torch.

## Gotchas

- **`import skimage` hard-crashes the interpreter if `$E\Library\bin` is not on
  PATH.** Exit code `-1066598273` (0xC0000409, "stack buffer overrun"), no
  traceback. `-X faulthandler` traces it to `skimage/color/colorconv.py:396` →
  `scipy.linalg.inv` → Windows exception `0xc06d007f`, a **DLL delay-load
  failure**: conda's MKL lives in `Library\bin`, and invoking `python.exe` by
  absolute path (rather than `conda activate`) leaves it off PATH. The driver's
  `check` calls `scipy.linalg.inv` first as a canary.

- **torch and this env's scipy cannot coexist in one process.** Both ship a
  `libiomp5md.dll` — conda's is a 157 KB stub, torch's the real 1.9 MB Intel
  OpenMP — and the Windows loader gives whichever loads first to both.
  scipy first → `import torch` raises `OSError [WinError 182]` on `shm.dll`;
  torch first → `scipy.linalg.inv` hard-crashes. PATH order, `os.add_dll_directory`,
  `KMP_DUPLICATE_LIB_OK` and preloading torch's OpenMP were all tried; none work.
  The real fix is a conda-built torch (`conda install -c pytorch pytorch=1.13.1
  cpuonly`), **which could not be installed here** — conda and pip both hit
  `CERTIFICATE_VERIFY_FAILED` / connection resets through this network's TLS
  interception. So torch is currently unavailable.

- **The driver sidesteps torch by never running `SubFunctions/__init__.py`.**
  That file does `from .Analysis import ...` → `Model` → `Attention` → `torch`,
  so *any* `SubFunctions` import drags torch in — including `GUI.py`'s innocuous
  `from SubFunctions.GradCAM import GradCAM`. `driver.subfunctions_lite()`
  registers `SubFunctions` as a bare package (a module object with `__path__`)
  so submodule imports resolve without executing `__init__`. Safe because
  `GradCAM.py` and `VisualizeResults.py` import nothing else from the package.
  It also cuts GUI import from minutes to ~10 s. The full-analysis path does
  need the real `__init__`; `--full-package` opts back in.

- **Importing this project downloads 733 MB of keras weights.**
  `SubFunctions/GetFeatures.py:13-14` and `GUI.py:15-16` call `ResNet101()` and
  `VGG16()` **at module scope**, so a bare `import` triggers it (resnet101 180 MB,
  vgg16 553 MB → `~/.keras/models`), with a progress bar that floods stdout.
  `get_gradcam` additionally pulls MobileNetV2 (14 MB) on first use. Cached after
  the first run; `driver.py check` reports which are present.

- **Screenshots: neither Tk coordinates nor a DPI ratio give the right rectangle.**
  `winfo_rootx/rooty/width/height` are Tk-space; `PIL.ImageGrab` is physical
  pixels; and customtkinter applies a *second* scaling factor of its own. Scaling
  the Tk box by screen-size ratio captured the desktop and taskbar instead of the
  window. What works: `SetProcessDpiAwareness(2)` **before** tkinter is imported,
  plus Win32 `GetWindowRect` on the toplevel HWND — reached via
  `GetAncestor(root.winfo_id(), GA_ROOT)`, since `winfo_id()` is a child window.

- **Tk never repaints unless you pump it.** The driver never calls `mainloop()`,
  so without manual `update_idletasks()`/`update()` loops every screenshot is a
  blank window.

- **`GUI.exit_event` is a `@staticmethod` referencing a module global `app`**
  that only exists under `if __name__ == "__main__"`. The driver assigns
  `GUI.app = app` after constructing it, or Exit raises `NameError`.

- **`SubFunctions` prints emoji through termcolor**, which raises
  `UnicodeEncodeError` on a cp1252 console and kills the run partway. The driver
  reconfigures stdout to utf-8/replace.

- **`PlotResults()` defaults to `show=True`**, i.e. ~40 blocking `plt.show()`
  windows. `PlotResults(show=False, save=True)` writes files instead — that flag
  pair is the whole difference between the human and agent paths.

- **CWD must be the project root.** `Temp\themes\rose.json`, `Analysis\TP\*.npy`
  and the vendored `./mealpy` are all resolved relative to CWD. The driver
  `chdir`s itself, so it can be invoked from anywhere.

- **In PowerShell, `$?` lies about native exit codes.** `python -c "import seaborn"`
  returned 0 but wrote a warning to stderr, and `$?` went `False`. Check
  `$LASTEXITCODE`.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Exit `-1066598273` or `3`, no traceback | `Library\bin` missing from PATH (skimage/scipy), or torch got imported. Re-run `driver.py check`. |
| `OSError [WinError 182] ... shm.dll` | torch vs conda MKL clash. Expected; ignore unless you need the full-analysis path. |
| Silent failure, no output at all | Missing `-u`; buffered stdout is lost on a hard crash. |
| `ERROR: Could not find a version that satisfies the requirement PySimpleGUI==4.60.5` | Use `4.60.5.1`. |
| `ImportError: cannot import name 'greycomatrix'` | scikit-image ≥ 0.20. Pin `0.19.3`. |
| `pip`/`conda` → `CERTIFICATE_VERIFY_FAILED, self-signed certificate in chain` | TLS-intercepting proxy. Export the Windows trusted roots to a PEM and pass `pip --cert <pem>`; this fixed verification (downloads may still reset). |
| GUI screenshot shows desktop/taskbar instead of the window | DPI awareness not set before importing tkinter, or Tk coords used instead of `GetWindowRect`. |
| Screenshots blank | Tk events not pumped between actions. |
| `driver.py gui` → "no video at ...; run 'make-video' first" | Run the `make-video` command. |
| First run appears to hang with a progress bar | Downloading the 733 MB of keras weights. Let it finish once. |
