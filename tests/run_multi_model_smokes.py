#!/usr/bin/env python
"""Runnable smoke suite (no pytest required) for multi-model train/predict.

Exit 0 only if adapters + DCNN + EfficientNetV2B0 + MobileNetV2 each produce
predictions whose length matches the test labels.
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def main():
    import numpy as np

    try:
        import torch  # noqa: F401
        print("torch", torch.__version__)
    except Exception as e:
        print("torch skip", type(e).__name__, e)

    with open(ROOT / "Features" / "Features.pkl", "rb") as fh:
        data = pickle.load(fh)
    x = np.asarray(data["features"])
    y = np.asarray(data["labels"]).astype(int)
    idx0 = np.where(y == 0)[0][:16]
    idx1 = np.where(y == 1)[0][:16]
    tr = np.concatenate([idx0[:12], idx1[:12]])
    te = np.concatenate([idx0[12:16], idx1[12:16]])
    x_tr, y_tr, x_te, y_te = x[tr], y[tr], x[te], y[te]
    print("split", x_tr.shape, x_te.shape)

    from SubFunctions.feature_adapters import as_seq, as_spatial, as_rgb_image
    assert as_seq(x_tr).shape == (24, 480, 320)
    assert as_spatial(x_tr).shape == (24, 32, 32, 10)
    assert as_rgb_image(x_tr, size=32).shape == (24, 32, 32, 3)
    print("adapters OK")

    from SubFunctions.MultiModel import (
        train_predict_dcnn,
        train_predict_efficientnetv2,
        train_predict_mobilenetv2,
        LATEST_BACKBONE,
    )
    assert LATEST_BACKBONE == "EfficientNetV2B0"

    p1 = train_predict_dcnn(x_tr, y_tr, x_te, y_te, epochs=1, batch_size=8)
    assert len(p1) == len(y_te), (len(p1), len(y_te))
    print("DCNN OK", p1)

    p2 = train_predict_efficientnetv2(
        x_tr, y_tr, x_te, y_te, epochs=1, batch_size=4, img_size=32)
    assert len(p2) == len(y_te)
    print("EfficientNetV2B0 OK", p2)

    p3 = train_predict_mobilenetv2(
        x_tr, y_tr, x_te, y_te, epochs=1, batch_size=4, img_size=32)
    assert len(p3) == len(y_te)
    print("MobileNetV2 OK", p3)

    print("ALL MULTI-MODEL SMOKES PASSED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        raise SystemExit(1)
