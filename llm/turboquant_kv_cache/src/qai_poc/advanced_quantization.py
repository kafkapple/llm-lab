"""Quantization utilities beyond the introductory fake-quantization example."""

from dataclasses import dataclass

import numpy as np

from .quantization import dequantize, quantize


@dataclass(frozen=True)
class StorageEstimate:
    """Idealized packed storage including scale metadata."""

    payload_bytes: int
    scale_bytes: int
    total_bytes: int
    compression_ratio: float


def storage_estimate(
    num_values: int, bits: int, group_size: int, scale_bits: int = 32
) -> StorageEstimate:
    """Estimate packed weight storage relative to FP32."""
    if num_values < 1 or group_size < 1:
        raise ValueError("num_values and group_size must be positive")
    groups = (num_values + group_size - 1) // group_size
    payload = (num_values * bits + 7) // 8
    scales = (groups * scale_bits + 7) // 8
    total = payload + scales
    return StorageEstimate(payload, scales, total, num_values * 4 / total)


def pack_codes(codes: np.ndarray, bits: int) -> np.ndarray:
    """Pack unsigned low-bit codes into bytes."""
    if bits not in (2, 4, 8):
        raise ValueError("bits must be one of 2, 4, 8")
    values = np.asarray(codes, dtype=np.uint8).ravel()
    per_byte = 8 // bits
    padded = np.pad(values, (0, (-values.size) % per_byte))
    packed = np.zeros(padded.size // per_byte, dtype=np.uint8)
    mask = (1 << bits) - 1
    for offset in range(per_byte):
        packed |= (padded[offset::per_byte] & mask) << (offset * bits)
    return packed


def unpack_codes(packed: np.ndarray, bits: int, count: int) -> np.ndarray:
    """Unpack bytes produced by :func:`pack_codes`."""
    per_byte = 8 // bits
    mask = (1 << bits) - 1
    output = np.empty(np.asarray(packed).size * per_byte, dtype=np.uint8)
    for offset in range(per_byte):
        output[offset::per_byte] = (packed >> (offset * bits)) & mask
    return output[:count]


def activation_aware_weight(
    weight: np.ndarray,
    calibration: np.ndarray,
    bits: int = 4,
    group_size: int = 128,
    alpha: float = 0.5,
) -> np.ndarray:
    """Return an AWQ-inspired effective weight using activation channel scales.

    This teaches AWQ's equivalent-scaling idea; it is not the official search
    algorithm or CUDA kernel.
    """
    if weight.shape[1] != calibration.shape[1]:
        raise ValueError("weight and calibration feature dimensions must match")
    importance = np.mean(np.abs(calibration), axis=0) + 1e-6
    channel_scale = importance**alpha
    channel_scale /= np.exp(np.mean(np.log(channel_scale)))
    scaled_weight = weight * channel_scale[None, :]
    restored = dequantize(quantize(scaled_weight, bits, group_size))
    return restored / channel_scale[None, :]


def output_mse(
    calibration: np.ndarray, weight: np.ndarray, restored: np.ndarray
) -> float:
    """Measure layer-output error instead of weight-only error."""
    reference = calibration @ weight.T
    prediction = calibration @ restored.T
    return float(np.mean((reference - prediction) ** 2))
