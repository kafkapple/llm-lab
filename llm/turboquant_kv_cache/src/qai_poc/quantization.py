"""Educational group-wise symmetric quantization."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QuantizedArray:
    """Integer codes and per-group scales."""

    codes: np.ndarray
    scales: np.ndarray
    shape: tuple[int, ...]
    bits: int
    group_size: int


def quantize(x: np.ndarray, bits: int, group_size: int = 128) -> QuantizedArray:
    """Quantize a floating array with symmetric per-group scaling."""
    if bits not in (2, 4, 8):
        raise ValueError("bits must be one of 2, 4, 8")
    if group_size < 1:
        raise ValueError("group_size must be positive")
    flat = np.asarray(x, dtype=np.float32).ravel()
    padding = (-flat.size) % group_size
    padded = np.pad(flat, (0, padding)).reshape(-1, group_size)
    limit = 2 ** (bits - 1) - 1
    scales = np.max(np.abs(padded), axis=1, keepdims=True) / limit
    scales[scales == 0] = 1.0
    codes = np.clip(np.rint(padded / scales), -limit, limit).astype(np.int8)
    return QuantizedArray(codes, scales.astype(np.float32), x.shape, bits, group_size)


def dequantize(q: QuantizedArray) -> np.ndarray:
    """Restore approximate floating values from quantized codes."""
    size = int(np.prod(q.shape))
    restored = (q.codes.astype(np.float32) * q.scales).ravel()[:size]
    return restored.reshape(q.shape)


def metrics(x: np.ndarray, restored: np.ndarray, bits: int) -> dict[str, float]:
    """Return reconstruction and ideal storage metrics."""
    x_flat = x.ravel().astype(np.float64)
    y_flat = restored.ravel().astype(np.float64)
    denom = np.linalg.norm(x_flat) * np.linalg.norm(y_flat)
    return {
        "bits": bits,
        "mse": float(np.mean((x_flat - y_flat) ** 2)),
        "cosine": float(np.dot(x_flat, y_flat) / denom),
        "ideal_compression": 32.0 / bits,
    }
