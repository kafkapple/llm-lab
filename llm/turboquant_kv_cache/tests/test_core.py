import numpy as np
import torch

from qai_poc.contextual import ContextEncoder
from qai_poc.data2vec import ContextTransformer, PointwiseBaseline, make_modality_batch
from qai_poc.advanced_quantization import pack_codes, storage_estimate, unpack_codes
from qai_poc.quantization import dequantize, quantize
from qai_poc.sequence import FNetModel, SelectiveSSM, TransformerModel
from qai_poc.ssm import SelectiveSSMModel
from qai_poc.real_data import byte_sequences
from qai_poc.real_eval import quantize_linear_weights


def test_quantization_shape_and_order():
    x = np.random.default_rng(0).normal(size=(17, 19)).astype(np.float32)
    errors = [np.mean((x - dequantize(quantize(x, bits))) ** 2) for bits in (2, 4, 8)]
    assert errors[0] > errors[1] > errors[2]
    assert dequantize(quantize(x, 4)).shape == x.shape


def test_model_shapes():
    tokens = torch.randint(0, 16, (2, 8))
    for model in (
        TransformerModel(16, 16, 8),
        FNetModel(16, 16),
        SelectiveSSM(16, 16),
    ):
        assert model(tokens).shape == (2, 8, 16)
    assert ContextEncoder(4, 8)(torch.randn(2, 8, 4)).shape == (2, 8, 8)


def test_packing_and_storage():
    codes = np.arange(16, dtype=np.uint8) % 4
    assert np.array_equal(unpack_codes(pack_codes(codes, 2), 2, len(codes)), codes)
    assert storage_estimate(1024, 4, 128).compression_ratio < 8


def test_advanced_model_shapes():
    tokens = torch.randint(0, 16, (2, 8))
    assert SelectiveSSMModel(16, 16, 4)(tokens).shape == (2, 8, 16)
    inputs = make_modality_batch("text", 2, 8, 4, torch.device("cpu"))
    assert ContextTransformer(4, 16, 2, 1)(inputs).shape == (2, 8, 16)
    assert PointwiseBaseline(4, 16)(inputs).shape == (2, 8, 16)


def test_real_data_and_model_quantization():
    assert byte_sequences(["real text"], 8).shape == (1, 9)
    model = torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.ReLU())
    restored = quantize_linear_weights(model, 4, 8)
    assert restored(torch.randn(2, 8)).shape == (2, 4)
    assert not torch.equal(model[0].weight, restored[0].weight)
