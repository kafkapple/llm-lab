"""Small, auditable TurboQuant-MSE teaching reference.

This is an algorithm-level approximation, not Google's production implementation.
It uses a seeded orthogonal rotation and a normal-distribution Lloyd-Max codebook.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class RotatedQuantized:
    codes: np.ndarray
    levels: np.ndarray
    norms: np.ndarray
    rotation: np.ndarray
    original_dim: int
    bits: int


@lru_cache(maxsize=8)
def _lloyd_max_normal(bits: int, *, samples: int = 200_000) -> np.ndarray:
    """Deterministic scalar codebook for an approximately N(0, 1) coordinate."""
    if bits not in range(1, 9):
        raise ValueError("bits must be in [1, 8]")
    rng = np.random.default_rng(0)
    values = np.sort(rng.normal(size=samples))
    edges = np.linspace(0, samples, 2**bits + 1, dtype=int)
    levels = np.array(
        [values[edges[i] : edges[i + 1]].mean() for i in range(2**bits)]
    )
    for _ in range(20):
        thresholds = (levels[:-1] + levels[1:]) / 2
        labels = np.searchsorted(thresholds, values)
        updated = np.array(
            [values[labels == i].mean() for i in range(len(levels))]
        )
        if np.allclose(levels, updated):
            break
        levels = updated
    return levels.astype(np.float32)


def random_rotation(dim: int, seed: int) -> np.ndarray:
    """Return a reproducible Haar-like orthogonal matrix for a teaching-scale PoC."""
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(dim, dim)))
    q *= np.sign(np.diag(r))
    return q.astype(np.float32)


def quantize_mse(x: np.ndarray, bits: int, *, seed: int = 0) -> RotatedQuantized:
    """Normalize, rotate, and scalar-quantize rows of x."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("x must have shape [vectors, dimension]")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    safe_norms = np.maximum(norms, np.finfo(np.float32).tiny)
    rotation = random_rotation(x.shape[1], seed)
    rotated = (x / safe_norms) @ rotation
    scaled = rotated * np.sqrt(x.shape[1])
    levels = _lloyd_max_normal(bits)
    thresholds = (levels[:-1] + levels[1:]) / 2
    codes = np.searchsorted(thresholds, scaled).astype(np.uint8)
    return RotatedQuantized(
        codes, levels, norms.astype(np.float32), rotation, x.shape[1], bits
    )


def dequantize_mse(q: RotatedQuantized) -> np.ndarray:
    scaled = q.levels[q.codes]
    unit = (scaled / np.sqrt(q.original_dim)) @ q.rotation.T
    return (unit * q.norms).astype(np.float32)


def effective_bits_per_coordinate(q: RotatedQuantized) -> float:
    """Serialized teaching estimate including per-vector FP32 norms.

    The shared rotation/codebook are amortized and deliberately excluded. Production
    benchmarking must also count packing alignment, allocator, and workspace bytes.
    """
    payload = q.codes.size * q.bits
    norms = q.norms.size * 32
    return (payload + norms) / q.codes.size


def distortion_metrics(x: np.ndarray, reconstructed: np.ndarray) -> dict[str, float]:
    error = x - reconstructed
    denom = np.sum(x * x, axis=1)
    nmse = np.mean(np.sum(error * error, axis=1) / np.maximum(denom, 1e-12))
    cosine = np.sum(x * reconstructed, axis=1) / np.maximum(
        np.linalg.norm(x, axis=1) * np.linalg.norm(reconstructed, axis=1), 1e-12
    )
    return {"normalized_mse": float(nmse), "mean_cosine": float(cosine.mean())}


def inner_product_metrics(
    queries: np.ndarray, keys: np.ndarray, reconstructed_keys: np.ndarray
) -> dict[str, float]:
    reference = queries @ keys.T
    estimate = queries @ reconstructed_keys.T
    error = estimate - reference
    return {
        "ip_bias": float(error.mean()),
        "ip_rmse": float(np.sqrt(np.mean(error * error))),
    }
