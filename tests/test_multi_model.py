"""Smoke tests for multi-model train/predict on Paper 2 feature tensors.

These call the real MultiModel train_predict functions (not stubs) on a tiny
subset of Features/Features.pkl so CI/local runs stay short.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


@pytest.fixture(scope="module")
def tiny_split():
    """Load Features.pkl and return a small balanced train/test set."""
    import pickle

    pkl = ROOT / "Features" / "Features.pkl"
    assert pkl.exists(), f"missing {pkl}"
    with open(pkl, "rb") as fh:
        data = pickle.load(fh)
    x = np.asarray(data["features"])
    y = np.asarray(data["labels"]).astype(int)
    assert x.ndim == 5 and x.shape[0] == len(y)
    # take 12 authentic + 12 forged for train, 4+4 for test
    idx0 = np.where(y == 0)[0][:16]
    idx1 = np.where(y == 1)[0][:16]
    tr = np.concatenate([idx0[:12], idx1[:12]])
    te = np.concatenate([idx0[12:16], idx1[12:16]])
    return x[tr], y[tr], x[te], y[te]


def test_feature_adapters_shapes(tiny_split):
    from SubFunctions.feature_adapters import as_seq, as_spatial, as_rgb_image

    x_tr, _, _, _ = tiny_split
    seq = as_seq(x_tr)
    assert seq.shape == (x_tr.shape[0], 15 * 32, 32 * 10)
    sp = as_spatial(x_tr)
    assert sp.shape == (x_tr.shape[0], 32, 32, 10)
    rgb = as_rgb_image(x_tr, size=32)
    assert rgb.shape == (x_tr.shape[0], 32, 32, 3)


def test_dcnn_pred_length_matches_test(tiny_split):
    from SubFunctions.MultiModel import train_predict_dcnn

    x_tr, y_tr, x_te, y_te = tiny_split
    pred = train_predict_dcnn(x_tr, y_tr, x_te, y_te, epochs=1, batch_size=8)
    assert len(pred) == len(y_te)
    assert set(np.unique(pred)).issubset({0, 1})


def test_efficientnetv2_pred_length_matches_test(tiny_split):
    """Latest backbone path — real EfficientNetV2B0 fit on projected features."""
    from SubFunctions.MultiModel import train_predict_efficientnetv2, LATEST_BACKBONE

    assert LATEST_BACKBONE == "EfficientNetV2B0"
    x_tr, y_tr, x_te, y_te = tiny_split
    pred = train_predict_efficientnetv2(
        x_tr, y_tr, x_te, y_te, epochs=1, batch_size=4, img_size=32)
    assert len(pred) == len(y_te)
    assert set(np.unique(pred)).issubset({0, 1})


def test_two_models_same_split_both_produce_preds(tiny_split):
    """Two different model paths train on identical split indices."""
    from SubFunctions.MultiModel import train_predict_dcnn, train_predict_mobilenetv2

    x_tr, y_tr, x_te, y_te = tiny_split
    p1 = train_predict_dcnn(x_tr, y_tr, x_te, y_te, epochs=1, batch_size=8)
    p2 = train_predict_mobilenetv2(
        x_tr, y_tr, x_te, y_te, epochs=1, batch_size=4, img_size=32)
    assert len(p1) == len(y_te) == len(p2)
