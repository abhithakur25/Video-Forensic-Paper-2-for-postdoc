"""Reshape Paper 2 feature tensors for different classifier input layouts.

Shipped Features/Features.pkl layout:
    features: (N, T=15, H=32, W=32, C=10)
    labels:   (N,)
"""
from __future__ import annotations

import numpy as np


def as_seq(x: np.ndarray) -> np.ndarray:
    """(N,15,32,32,10) -> (N, 15*32, 32*10) for BiLSTM (same as Model.BiLSTMGBM)."""
    x = np.asarray(x)
    if x.ndim != 5:
        raise ValueError(f"as_seq expects rank-5, got {x.shape}")
    return x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3] * x.shape[4])


def as_spatial(x: np.ndarray) -> np.ndarray:
    """Mean over time -> (N, 32, 32, 10) for small Conv2D classifiers."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 5:
        raise ValueError(f"as_spatial expects rank-5, got {x.shape}")
    return np.mean(x, axis=1)


def as_rgb_image(x: np.ndarray, size: int = 64) -> np.ndarray:
    """Project features to 3-channel images for Keras Application backbones.

    Takes time-mean volume, maps first 3 of 10 channels to RGB-like planes,
    resizes to `size` x `size` with bilinear (cv2).
    """
    import cv2

    vol = as_spatial(x)  # (N,32,32,10)
    # pick 3 informative channel indices: mean/energy-ish planes 0,3,6
    ch = vol[:, :, :, [0, 3, 6]]
    # per-sample min-max normalize to [0, 255]
    out = np.zeros((ch.shape[0], size, size, 3), dtype=np.float32)
    for i in range(ch.shape[0]):
        plane = ch[i]
        lo = plane.min(axis=(0, 1), keepdims=True)
        hi = plane.max(axis=(0, 1), keepdims=True)
        denom = np.maximum(hi - lo, 1e-6)
        normed = (plane - lo) / denom
        img = (normed * 255.0).astype(np.uint8)
        resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
        out[i] = resized.astype(np.float32)
    return out


def as_flat(x: np.ndarray) -> np.ndarray:
    """Flatten for classical heads: (N, 15*32*32*10)."""
    x = np.asarray(x, dtype=np.float32)
    return x.reshape(x.shape[0], -1)
