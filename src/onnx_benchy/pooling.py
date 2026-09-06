"""Pooling + L2 normalization over ONNX outputs (numpy)."""

from __future__ import annotations

import numpy as np


def apply_pooling(
    hidden: np.ndarray, mask: np.ndarray, mode: str
) -> np.ndarray:
    """Pool [batch, seq, hidden] -> [batch, hidden]. Mode 'none' passes through."""
    if mode == "none":
        if hidden.ndim != 2:
            raise ValueError(
                f"--pooling none needs a rank-2 (already pooled) output, "
                f"got rank {hidden.ndim}"
            )
        return hidden
    if hidden.ndim != 3:
        raise ValueError(
            f"--pooling {mode} needs a rank-3 (per-token) output, "
            f"got rank {hidden.ndim}; use --pooling none"
        )
    m = mask.astype(hidden.dtype)[..., None]  # [B, S, 1]
    if mode == "cls":
        return hidden[:, 0]
    if mode == "mean":
        summed = (hidden * m).sum(axis=1)
        counts = m.sum(axis=1).clip(min=1e-9)
        return summed / counts
    if mode == "max":
        masked = np.where(m > 0, hidden, np.full_like(hidden, -np.inf))
        return masked.max(axis=1)
    if mode == "lasttoken":
        idx = m.sum(axis=1).astype(np.int64).squeeze(-1) - 1  # last non-pad
        idx = np.clip(idx, 0, hidden.shape[1] - 1)
        return hidden[np.arange(hidden.shape[0]), idx]
    raise ValueError(f"unknown pooling mode {mode!r}")


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True).clip(min=eps)
    return x / norms


def pool_and_norm(
    output: np.ndarray, mask: np.ndarray, pooling: str, normalize: bool
) -> np.ndarray:
    emb = apply_pooling(output, mask, pooling)
    if normalize:
        emb = l2_normalize(emb)
    return emb
