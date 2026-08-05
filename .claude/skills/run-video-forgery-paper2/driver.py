#!/usr/bin/env python
"""
Agent-facing driver for the video-forgery-detection research code.

The app has two surfaces and neither can be driven from a plain shell:

  * GUI.py  - a customtkinter desktop app whose only entry point is
              app.mainloop(), and whose "Select Video" button opens a
              blocking native file dialog.
  * Main.py - blocks on a PySimpleGUI popup_yes_no(), then calls
              PlotResults() with show=True, which blocks on plt.show()
              once per figure (~40 figures).

This driver bypasses both blocking prompts and drives the real code.

Commands:
  check        env sanity: imports, keras weight cache, vendored mealpy
  make-video   synthesize driver_out/sample.mp4 from Results/ImageResults/Input
  plots        regenerate every analysis figure headlessly into Results/
  gui          walk the GUI through all 6 stages, screenshot each
  all          check -> make-video -> plots -> gui

Run it with the project directory as CWD, or from anywhere - it chdir's to
the project root itself (relative paths like "Temp\\themes\\rose.json" and
the vendored ./mealpy package require it).
"""
import argparse
import ctypes
import os
import sys
import time
import traceback
from pathlib import Path

# <unit>/.claude/skills/run-*/driver.py  ->  <unit>
PROJECT = Path(__file__).resolve().parents[3]
OUT = PROJECT / "driver_out"
SHOTS = OUT / "screenshots"


def log(msg):
    sys.stdout.write(f"[driver] {msg}\n")
    sys.stdout.flush()


def subfunctions_lite():
    """Register `SubFunctions` as a bare package so that submodule imports
    work WITHOUT executing SubFunctions/__init__.py.

    __init__.py does `from .Analysis import ...` -> Model -> Attention -> torch,
    and in this conda env torch cannot be imported in the same process as the
    conda-forge MKL build of scipy (see SKILL.md "Gotchas"). Nothing the GUI or
    the plotting code needs lives behind that import: GUI.py only wants
    SubFunctions.GradCAM (+ SubFunctions.LDZP on Paper1) and the plots path only
    wants SubFunctions.VisualizeResults - none of which import anything else
    from the package. Skipping __init__ also avoids ~3 min of Analysis/Model
    /mealpy import time.

    The full 48-hour ReadDataset/TPAnalysis/KFAnalysis path DOES need the real
    __init__ (and therefore a working torch); use --full-package for that.
    """
    import types
    if "SubFunctions" in sys.modules:
        return
    pkg = types.ModuleType("SubFunctions")
    pkg.__path__ = [str(PROJECT / "SubFunctions")]
    pkg.__package__ = "SubFunctions"
    sys.modules["SubFunctions"] = pkg
    log("SubFunctions registered without running __init__.py (torch bypass)")


def setup():
    """chdir to project root and make stdout tolerate the code's emoji output."""
    os.chdir(PROJECT)
    sys.path.insert(0, str(PROJECT))
    OUT.mkdir(exist_ok=True)
    SHOTS.mkdir(exist_ok=True)
    # SubFunctions prints emoji via termcolor; cp1252 consoles raise
    # UnicodeEncodeError and kill the run partway through.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


# --------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------
def cmd_check(args):
    ok = True
    log(f"project root : {PROJECT}")
    log(f"python       : {sys.version.split()[0]} @ {sys.executable}")

    # scipy.linalg is the canary: if the conda env's Library\bin is missing
    # from PATH, this hard-crashes the interpreter (0xC0000409) instead of
    # raising, and takes skimage down with it.
    import numpy as np
    import scipy.linalg
    scipy.linalg.inv(np.eye(3))
    log("scipy.linalg LAPACK delay-load OK (Library\\bin is on PATH)")

    for mod in ("cv2", "tensorflow", "keras", "skimage", "customtkinter",
                "PySimpleGUI", "peakutils", "seaborn", "sklearn",
                "pandas", "tqdm", "PIL"):
        try:
            m = __import__(mod)
            log(f"  import {mod:<14} {getattr(m, '__version__', '?')}")
        except Exception as e:
            ok = False
            log(f"  IMPORT FAILED {mod}: {type(e).__name__}: {e}")

    # torch is expected to fail here: it cannot share a process with the
    # conda-forge MKL build of scipy. Not fatal - nothing this driver runs
    # needs it (and Paper1 never imports it at all).
    uses_torch = any(
        "import torch" in p.read_text(encoding="utf-8", errors="replace")
        for p in (PROJECT / "SubFunctions").glob("*.py"))
    try:
        import torch
        log(f"  import {'torch':<14} {torch.__version__}")
    except Exception as e:
        if uses_torch:
            log(f"  torch UNAVAILABLE ({type(e).__name__}) - expected in this "
                f"env; only the full-analysis path needs it")
        else:
            log(f"  torch UNAVAILABLE ({type(e).__name__}) - this project never "
                f"imports torch, so it does not matter")

    from skimage.feature import greycomatrix  # noqa: F401  (0.19.x spelling)
    log("  skimage.feature.greycomatrix present (needs scikit-image 0.19.x)")

    if not (PROJECT / "mealpy" / "__init__.py").exists():
        ok = False
        log("  MISSING vendored ./mealpy")
    else:
        log("  vendored ./mealpy present")

    cache = Path.home() / ".keras" / "models"
    for w in ("resnet101_weights_tf_dim_ordering_tf_kernels.h5",
              "vgg16_weights_tf_dim_ordering_tf_kernels.h5"):
        p = cache / w
        if p.exists():
            log(f"  cached {w} ({p.stat().st_size / 1e6:.0f} MB)")
        else:
            log(f"  NOT CACHED {w} - first GUI/import run will download it")

    log("CHECK OK" if ok else "CHECK FAILED")
    return 0 if ok else 1


# --------------------------------------------------------------------------
# make-video
# --------------------------------------------------------------------------
def cmd_make_video(args):
    """The repo ships no DATASET/ and no video, but GUI.select_data_event
    needs one. Build a short clip from the sample frames in Results/."""
    import cv2
    import numpy as np

    dst = OUT / "sample.mp4"
    src = sorted((PROJECT / "Results" / "ImageResults" / "Input").glob("*.jpg"))
    frames = []
    for p in src[:30]:
        im = cv2.imread(str(p))
        if im is not None:
            frames.append(cv2.resize(im, (256, 256)))
    if not frames:
        log("no sample frames found; generating synthetic ones")
        for i in range(30):
            f = np.full((256, 256, 3), 40, np.uint8)
            cv2.circle(f, (60 + 4 * i, 128), 40, (200, 180, 160), -1)
            frames.append(f)

    vw = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), 10, (256, 256))
    for f in frames:
        vw.write(f)
    vw.release()
    log(f"wrote {dst} ({len(frames)} frames, {dst.stat().st_size} bytes)")
    return 0


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------
def cmd_plots(args):
    """Main.py's else-branch, minus the two blocking prompts.

    Main.py does PlotResults() -> show=True -> plt.show() per figure. Under
    Agg with show=False/save=True the same figures land in Results/ instead.
    """
    import matplotlib
    matplotlib.use("Agg")

    if getattr(args, "full_package", False):
        from SubFunctions import PlotResults
    else:
        subfunctions_lite()
        from SubFunctions.VisualizeResults import PlotResults

    before = {p: p.stat().st_mtime for p in PROJECT.glob("Results/**/*.png")}
    pl = PlotResults(show=False, save=True)
    t0 = time.time()
    pl.TPAnalysisResult()
    pl.KFAnalysisResult()
    dt = time.time() - t0

    after = list(PROJECT.glob("Results/**/*.png"))
    fresh = [p for p in after if p not in before or p.stat().st_mtime > before[p]]
    log(f"plots done in {dt:.1f}s - {len(fresh)} figures written/updated "
        f"({len(after)} PNGs total under Results/)")
    for p in sorted(fresh)[:8]:
        log(f"    {p.relative_to(PROJECT)}")
    if len(fresh) > 8:
        log(f"    ... and {len(fresh) - 8} more")
    return 0 if fresh else 1


# --------------------------------------------------------------------------
# gui
# --------------------------------------------------------------------------
STAGES = [
    ("select", "select_data_event", "Select Video"),
    ("preprocess", "preprocessing_event", "Preprocessing"),
    ("gradcam", "get_gradcam", "GradCAM"),
    ("resnet", "get_resnetstat", "Resnet Statistical"),
    ("vgg", "get_vgg", "VGG / SIFT"),
    ("flow", "get_flow", "Optical Flow / Shape-Resnet"),
]


class Shooter:
    """Screengrab of the Tk window.

    Getting the capture rectangle right on Windows is fiddly: Tk's
    winfo_rootx/rooty/width/height are in Tk's own coordinate space, while
    PIL.ImageGrab works in physical screen pixels. On a scaled display they
    disagree, and customtkinter adds a *second* scaling factor of its own on
    top - so deriving a ratio from the screen size overshoots and you capture
    the desktop around the window (verified: got the taskbar).

    Win32 GetWindowRect on the toplevel HWND sidesteps all of it and returns
    exactly the rectangle ImageGrab needs - provided the process is DPI-aware,
    which setup_dpi() guarantees.
    """

    def __init__(self, root):
        from PIL import ImageGrab
        self.grab = ImageGrab.grab
        self.root = root
        self.n = 0
        # winfo_id() is the Tk child window; walk up to the real toplevel.
        self.hwnd = ctypes.windll.user32.GetAncestor(root.winfo_id(), 2)  # GA_ROOT
        log(f"toplevel hwnd {self.hwnd}, rect {self.rect()}")

    def rect(self):
        class RECT(ctypes.Structure):
            _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                        ("right", ctypes.c_long), ("bottom", ctypes.c_long)]
        r = RECT()
        if self.hwnd and ctypes.windll.user32.GetWindowRect(self.hwnd, ctypes.byref(r)):
            return (r.left, r.top, r.right, r.bottom)
        x, y = self.root.winfo_rootx(), self.root.winfo_rooty()
        return (x, y, x + self.root.winfo_width(), y + self.root.winfo_height())

    def pump(self, seconds=0.4):
        """Tk only repaints inside mainloop(); we never call mainloop(), so
        the window stays blank unless events are pumped by hand."""
        end = time.time() + seconds
        while time.time() < end:
            self.root.update_idletasks()
            self.root.update()
            time.sleep(0.02)

    def shot(self, name):
        self.n += 1
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.pump(0.6)
        img = self.grab(bbox=self.rect(), all_screens=True)
        dst = SHOTS / f"{self.n:02d}-{name}.png"
        img.save(dst)
        log(f"  screenshot {dst.relative_to(PROJECT)}  {img.size}")
        return dst


def setup_dpi():
    """Must run BEFORE tkinter/customtkinter is imported. Without it the
    process is DPI-virtualised, Win32 window rects come back in logical units,
    and screenshots capture the wrong region on a scaled display."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor v2
        log("process DPI awareness set (per-monitor)")
    except Exception:
        ctypes.windll.user32.SetProcessDPIAware()
        log("process DPI awareness set (system)")


def cmd_gui(args):
    setup_dpi()
    video = Path(args.video) if args.video else OUT / "sample.mp4"
    if not video.exists():
        log(f"no video at {video}; run 'make-video' first")
        return 1

    wanted = args.stages.split(",") if args.stages else [s[0] for s in STAGES]

    # Patch the blocking native file picker before GUI.py can call it.
    import tkinter.filedialog as fd
    fd.askopenfilename = lambda *a, **k: str(video)

    if not getattr(args, "full_package", False):
        subfunctions_lite()

    log("importing GUI (builds ResNet101 + VGG16 at module scope, ~30-90s)...")
    t0 = time.time()
    import GUI
    log(f"GUI imported in {time.time() - t0:.0f}s")

    app = GUI.App()
    GUI.app = app  # GUI.exit_event() is a staticmethod referencing global `app`
    sh = Shooter(app)
    sh.pump(0.8)
    sh.shot("launched")

    failures = []
    for key, method, label in STAGES:
        if key not in wanted:
            continue
        log(f"stage '{key}' -> App.{method}()  [{label}]")
        t = time.time()
        try:
            getattr(app, method)()
            sh.pump(0.5)
            sh.shot(key)
            log(f"  ok in {time.time() - t:.1f}s")
        except Exception as e:
            failures.append((key, f"{type(e).__name__}: {e}"))
            log(f"  FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            try:
                sh.shot(f"{key}-FAILED")
            except Exception:
                pass

    app.destroy()
    log(f"screenshots in {SHOTS.relative_to(PROJECT)}")
    if failures:
        log("FAILED STAGES: " + ", ".join(f"{k} ({m})" for k, m in failures))
        return 1
    log("GUI WALKTHROUGH OK")
    return 0


# --------------------------------------------------------------------------
# evaluate  (Paper 2 proposed model only — OM2AHL-BiG / BiLSTMGBM)
# --------------------------------------------------------------------------
def _prepare_eval_runtime(args):
    """Torch must load BEFORE scipy/MKL, and KMP_DUPLICATE_LIB_OK must be set.

    Model.BiLSTMGBM always imports Optimization and calls CoSH (mealpy HYBRID,
    epoch=10, pop=50) which is ~1000 model.predict calls. --skip-opt replaces
    it with a no-op so a smoke-scale sweep finishes in ~30-40 min instead of days.
    """
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import torch  # noqa: F401  — must precede scipy/tensorflow in this env
    log(f"torch {torch.__version__} loaded first (OpenMP order OK)")

    import SubFunctions.Model as ModelMod

    if getattr(args, "skip_opt", True):
        class _NoOpt:
            def __init__(self, model, x_test, y_test):
                self.model = model

            def main_update_hyperparameters(self):
                log("  CoSH optimization SKIPPED (--skip-opt)")
                return self.model

        ModelMod.Optimization = _NoOpt

    try:
        import keras.utils as ku
        ku.plot_model = lambda *a, **k: None
    except Exception:
        pass


def cmd_evaluate(args):
    """Train + evaluate OM2AHL-BiG (Network.BiLSTMGBM) on Features/Features.pkl.

    Mirrors TPAnalysis.ComparativeAnalysis for the proposed model only (Paper 2
    ships a single Network method; comparison baselines come from ResultsP1).
    """
    _prepare_eval_runtime(args)
    import numpy as np
    from SubFunctions.GetData import ReadDataset
    from SubFunctions.Analysis import train_test_split
    from SubFunctions.Model import Network
    from SubFunctions.Evaluate import Evaluation_Metrics
    from sklearn.model_selection import KFold

    t0 = time.time()
    data = ReadDataset(exec=False).read_data()
    lab = np.asarray(data["labels"])
    feats = np.asarray(data["features"])
    u, c = np.unique(lab, return_counts=True)
    log(f"features {feats.shape}  labels {dict(zip(u.tolist(), c.tolist()))}  "
        f"(0=original, 1=manipulated)  loaded in {time.time()-t0:.1f}s")

    epochs = int(args.epochs)
    pcts = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = []
    t_all = time.time()

    if getattr(args, "kfold", False):
        ks = [6, 7, 8, 9, 10]
        n = len(lab)
        grid = []
        for k in ks:
            kf = KFold(n_splits=k, random_state=1, shuffle=True)
            fold_ms = []
            for j, (tr, te) in enumerate(kf.split(np.arange(n))):
                if j >= int(args.folds_per_k):
                    break
                x_tr, x_te = feats[tr], feats[te]
                y_tr, y_te = lab[tr].astype(int), lab[te].astype(int)
                log(f"===== k={k} fold {j+1}/{min(k, args.folds_per_k)} "
                    f"train={len(tr)} test={len(te)} =====")
                t = time.time()
                try:
                    net = Network(x_train=x_tr, x_test=x_te, y_train=y_tr,
                                  y_test=y_te, epochs=epochs)
                    pred = net.BiLSTMGBM(epochs=epochs)
                    m = [float(x) for x in Evaluation_Metrics(y_te, np.asarray(pred))]
                    fold_ms.append(m)
                    log(f"  acc={m[0]:.4f} f1={m[4]:.4f} ({time.time()-t:.0f}s)")
                except Exception as e:
                    fold_ms.append([float("nan")] * 5)
                    log(f"  FAILED {type(e).__name__}: {e}")
            grid.append(np.nanmean(np.asarray(fold_ms, float), axis=0)
                        if fold_ms else [float("nan")] * 5)
            log(f"  -> k={k} mean acc={grid[-1][0]:.4f}")
        arr = np.asarray(grid, dtype=float)
        outdir = PROJECT / "Analysis1" / "KF"
        outdir.mkdir(parents=True, exist_ok=True)
        np.save(outdir / "COM_A_eval.npy", arr)
        mode = "kfold"
        col_labels = [f"k={k}" for k in ks]
    else:
        for tp in pcts:
            d = train_test_split(data, train_size=tp)
            log(f"===== TP={tp} train={d[0].shape[0]} test={d[1].shape[0]} "
                f"epochs={epochs} =====")
            t = time.time()
            try:
                net = Network(x_train=d[0], x_test=d[1], y_train=d[2],
                              y_test=d[3], epochs=epochs)
                pred = net.BiLSTMGBM(epochs=epochs)
                m = [float(x) for x in Evaluation_Metrics(
                    np.asarray(d[3]), np.asarray(pred))]
                results.append(m)
                log(f"  ACC={m[0]:.4f} SEN={m[1]:.4f} SPE={m[2]:.4f} "
                    f"PRE={m[3]:.4f} F1={m[4]:.4f}  ({time.time()-t:.0f}s)")
            except Exception as e:
                results.append([float("nan")] * 5)
                log(f"  FAILED {type(e).__name__}: {e}")
                traceback.print_exc()
        arr = np.asarray(results, dtype=float)
        outdir = PROJECT / "Analysis1" / "TP"
        outdir.mkdir(parents=True, exist_ok=True)
        np.save(outdir / "COM_A_eval.npy", arr)
        mode = "tp"
        col_labels = [f"{int(p*100)}%" for p in pcts]

    # Human-readable table
    metric_names = ["Accuracy", "Sensitivity", "Specificity", "Precision", "F1"]
    lines = [
        "Paper 2 (OM2AHL-BiG / BiLSTMGBM) evaluation",
        f"features : Features/Features.pkl shape={feats.shape}",
        f"samples  : {len(lab)} videos, class balance "
        f"{dict(zip(u.tolist(), c.tolist()))}",
        f"epochs   : {epochs}   (paper uses 500)",
        f"CoSH opt : {'skipped' if getattr(args, 'skip_opt', True) else 'enabled'}",
        f"mode     : {mode}",
        f"elapsed  : {time.time()-t_all:.0f}s",
        "",
    ]
    hdr = f"{'Split':<10}" + "".join(f"{m:>12}" for m in metric_names)
    lines += [hdr, "-" * len(hdr)]
    for i, lab_ in enumerate(col_labels):
        row = arr[i]
        lines.append(f"{lab_:<10}" + "".join(
            f"{(row[j] if row[j] == row[j] else float('nan')):>12.4f}"
            for j in range(5)))

    # Append published Analysis numbers when TP sweep
    pub = PROJECT / "Analysis" / ("KF" if mode == "kfold" else "TP") / "COM_A.npy"
    if pub.exists() and mode == "tp":
        paper = np.load(pub)
        lines += ["", "Published Analysis/TP/COM_A.npy (paper figures source):",
                  hdr, "-" * len(hdr)]
        for i, lab_ in enumerate(col_labels):
            if i < len(paper):
                row = paper[i]
                lines.append(f"{lab_:<10}" + "".join(
                    f"{float(row[j]):>12.4f}" for j in range(5)))

    text = "\n".join(lines) + "\n"
    print("\n" + text)
    tag = f"evaluation_{mode}_ep{epochs}.txt"
    dst = OUT / tag
    dst.write_text(text, encoding="utf-8")
    log(f"wrote {dst.relative_to(PROJECT)}  and Analysis1/"
        f"{'KF' if mode == 'kfold' else 'TP'}/COM_A_eval.npy")
    return 0


# --------------------------------------------------------------------------
# evaluate-multi  — real train/predict for ≥3 models on Paper 2 features
# --------------------------------------------------------------------------
def cmd_evaluate_multi(args):
    """Train and score multiple classifiers on Features.pkl (same splits).

    Comparison models actually fit on Paper 2 tensors — they do NOT load
    ResultsP1 CSVs. Includes EfficientNetV2B0 as the latest backbone variant
    available under TF 2.10 keras.applications.
    """
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    # torch before scipy for OM2AHL-BiG path
    try:
        import torch
        log(f"torch {torch.__version__} (for OM2AHL-BiG attention path)")
    except Exception as e:
        log(f"torch UNAVAILABLE ({type(e).__name__}) — OM2AHL-BiG may fail")

    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    from SubFunctions.GetData import ReadDataset
    from SubFunctions.Analysis import train_test_split
    from SubFunctions.Evaluate import Evaluation_Metrics
    from SubFunctions.MultiModel import (
        MODEL_REGISTRY, LATEST_BACKBONE, LATEST_BACKBONE_REASON, run_model,
    )

    epochs = int(args.epochs)
    pcts = [float(x) for x in args.train_pcts.split(",")]
    wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in wanted:
        if m not in MODEL_REGISTRY:
            log(f"unknown model {m!r}; known={list(MODEL_REGISTRY)}")
            return 1

    t0 = time.time()
    data = ReadDataset(exec=False).read_data()
    lab = np.asarray(data["labels"])
    feats = np.asarray(data["features"])
    u, c = np.unique(lab, return_counts=True)
    log(f"features {feats.shape}  labels {dict(zip(u.tolist(), c.tolist()))}")
    log(f"LATEST_BACKBONE={LATEST_BACKBONE}  reason={LATEST_BACKBONE_REASON}")
    log(f"tf={tf.__version__} keras={keras.__version__} epochs={epochs} "
        f"models={wanted} train_pcts={pcts}")

    # grid[model][split_i] = [ACC,SEN,SPE,PRE,F1] or nan
    grid = {m: [] for m in wanted}
    t_all = time.time()

    for si, tp in enumerate(pcts):
        d = train_test_split(data, train_size=tp)
        x_tr, x_te, y_tr, y_te = d[0], d[1], np.asarray(d[2]), np.asarray(d[3])
        log(f"===== split {si+1}/{len(pcts)} TP={tp} "
            f"train={len(y_tr)} test={len(y_te)} =====")
        for name in wanted:
            t = time.time()
            try:
                kwargs = {}
                if name == "OM2AHL-BiG":
                    kwargs["skip_opt"] = not getattr(args, "with_opt", False)
                pred = run_model(name, x_tr, y_tr, x_te, y_te,
                                 epochs=epochs, **kwargs)
                pred = np.asarray(pred).reshape(-1)
                if len(pred) != len(y_te):
                    raise RuntimeError(
                        f"pred length {len(pred)} != test labels {len(y_te)}")
                m = [float(x) for x in Evaluation_Metrics(y_te, pred)]
                grid[name].append(m)
                log(f"  {name:<16} ACC={m[0]:.4f} SEN={m[1]:.4f} SPE={m[2]:.4f} "
                    f"PRE={m[3]:.4f} F1={m[4]:.4f}  ({time.time()-t:.0f}s)")
            except Exception as e:
                grid[name].append([float("nan")] * 5)
                log(f"  {name:<16} FAILED {type(e).__name__}: {e}")
                traceback.print_exc()

    # save npy: one file per model, rows=splits
    outdir = PROJECT / "Analysis1" / "TP"
    outdir.mkdir(parents=True, exist_ok=True)
    letter = {m: chr(ord("A") + i) for i, m in enumerate(wanted)}
    for name in wanted:
        arr = np.asarray(grid[name], dtype=float)
        np.save(outdir / f"MULTI_{letter[name]}_{name.replace('/', '_')}.npy", arr)

    # side-by-side text table
    metric_names = ["Accuracy", "Sensitivity", "Specificity", "Precision", "F1"]
    lines = [
        "Paper 2 multi-model evaluation (real train/predict on Features.pkl)",
        f"features : {feats.shape}",
        f"samples  : {len(lab)}  balance {dict(zip(u.tolist(), c.tolist()))}",
        f"epochs   : {epochs}",
        f"train_pcts: {pcts}",
        f"models   : {wanted}",
        f"latest   : {LATEST_BACKBONE} — {LATEST_BACKBONE_REASON}",
        f"tf/keras : {tf.__version__} / {keras.__version__}",
        f"elapsed  : {time.time()-t_all:.0f}s",
        f"loaded_in: {t0 and time.time()-t0:.0f}s total wall incl. load",
        "",
    ]
    for mi, metric in enumerate(metric_names):
        hdr = f"{metric:<14}" + "".join(f"{int(p*100):>10}%" for p in pcts)
        lines += [hdr, "-" * len(hdr)]
        for name in wanted:
            cells = []
            for i in range(len(pcts)):
                v = grid[name][i][mi]
                cells.append(f"{v:>11.4f}" if v == v else f"{'FAILED':>11}")
            lines.append(f"{name:<14}" + "".join(cells))
        lines.append("")

    # failures summary
    lines.append("Per-model status:")
    for name in wanted:
        fails = sum(1 for row in grid[name] if any(v != v for v in row))
        ok = len(pcts) - fails
        lines.append(f"  {name}: {ok}/{len(pcts)} splits OK"
                     + (f", {fails} FAILED" if fails else ""))

    text = "\n".join(lines) + "\n"
    print("\n" + text)
    tag = f"evaluation_multi_ep{epochs}.txt"
    dst = OUT / tag
    dst.write_text(text, encoding="utf-8")
    # also CSV for easy comparison
    import csv
    csv_path = OUT / f"evaluation_multi_ep{epochs}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "train_pct", "ACC", "SEN", "SPE", "PRE", "F1", "status"])
        for name in wanted:
            for i, tp in enumerate(pcts):
                row = grid[name][i]
                ok = all(v == v for v in row)
                w.writerow([name, int(tp * 100),
                            *[f"{v:.6f}" if v == v else "" for v in row],
                            "OK" if ok else "FAILED"])
    log(f"wrote {dst.relative_to(PROJECT)} and {csv_path.relative_to(PROJECT)}")
    # require ≥1 numeric metric for ≥3 models
    ok_models = [m for m in wanted
                 if any(all(v == v for v in row) for row in grid[m])]
    if len(ok_models) < 3:
        log(f"WARNING: only {len(ok_models)} models produced numeric metrics "
            f"({ok_models}); need ≥3")
        return 1
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    full = dict(action="store_true", dest="full_package",
                help="run SubFunctions/__init__.py for real (needs a working torch)")
    sub.add_parser("check")
    sub.add_parser("make-video")
    p = sub.add_parser("plots")
    p.add_argument("--full-package", **full)
    g = sub.add_parser("gui")
    g.add_argument("--video", help="path to an .mp4/.avi to feed Select Video")
    g.add_argument("--stages", help="comma list: " + ",".join(s[0] for s in STAGES))
    g.add_argument("--full-package", **full)
    ev = sub.add_parser("evaluate",
                        help="train/eval OM2AHL-BiG on Features.pkl (TP sweep or k-fold)")
    ev.add_argument("--epochs", type=int, default=3,
                    help="training epochs per incremental chunk (paper=500; default 3)")
    ev.add_argument("--kfold", action="store_true",
                    help="K-fold (k=6..10) instead of training-percentage sweep")
    ev.add_argument("--folds-per-k", type=int, default=2,
                    help="how many folds of each k to run (default 2)")
    ev.add_argument("--skip-opt", action="store_true", default=True,
                    help="skip CoSH mealpy weight optimization (default on)")
    ev.add_argument("--with-opt", action="store_true",
                    help="enable CoSH optimization (very slow)")
    em = sub.add_parser(
        "evaluate-multi",
        help="multi-model train+eval on Features.pkl (DCNN, EfficientNetV2B0, "
             "MobileNetV2, OM2AHL-BiG)")
    em.add_argument("--epochs", type=int, default=3,
                    help="epochs per model (default 3; paper 500)")
    em.add_argument("--train-pcts", default="0.8,0.9",
                    help="comma training fractions (default 0.8,0.9 for speed)")
    em.add_argument("--models",
                    default="DCNN,EfficientNetV2B0,MobileNetV2,OM2AHL-BiG",
                    help="comma model names from MultiModel.MODEL_REGISTRY")
    em.add_argument("--with-opt", action="store_true",
                    help="enable CoSH for OM2AHL-BiG only (slow)")
    a = sub.add_parser("all")
    a.add_argument("--video")
    a.add_argument("--stages")
    a.add_argument("--full-package", **full)
    args = ap.parse_args()
    if getattr(args, "with_opt", False):
        args.skip_opt = False

    setup()
    if args.cmd == "all":
        for fn in (cmd_check, cmd_make_video, cmd_plots):
            rc = fn(args)
            if rc:
                return rc
        return cmd_gui(args)
    return {"check": cmd_check, "make-video": cmd_make_video,
            "plots": cmd_plots, "gui": cmd_gui,
            "evaluate": cmd_evaluate,
            "evaluate-multi": cmd_evaluate_multi}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
