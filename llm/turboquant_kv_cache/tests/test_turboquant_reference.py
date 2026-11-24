import numpy as np

from qai_poc.turboquant_reference import (
    dequantize_mse,
    distortion_metrics,
    effective_bits_per_coordinate,
    quantize_mse,
)


def test_reference_roundtrip_and_effective_bits():
    x = np.random.default_rng(4).normal(size=(12, 16)).astype(np.float32)
    q = quantize_mse(x, 4, seed=7)
    restored = dequantize_mse(q)
    assert restored.shape == x.shape
    assert q.codes.dtype == np.uint8
    assert effective_bits_per_coordinate(q) == 6.0


def test_more_bits_reduce_teaching_reference_distortion():
    x = np.random.default_rng(8).normal(size=(64, 16)).astype(np.float32)
    errors = []
    for bits in (2, 3, 4):
        restored = dequantize_mse(quantize_mse(x, bits, seed=2))
        errors.append(distortion_metrics(x, restored)["normalized_mse"])
    assert errors[0] > errors[1] > errors[2]
