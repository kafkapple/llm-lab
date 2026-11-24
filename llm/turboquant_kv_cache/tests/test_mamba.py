import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Mamba CUDA test")
def test_official_mamba_forward_backward():
    pytest.importorskip("mamba_ssm")
    from mamba_ssm import Mamba

    inputs = torch.randn(2, 32, 32, device="cuda", requires_grad=True)
    model = Mamba(d_model=32, d_state=8, d_conv=4, expand=2).cuda()
    outputs = model(inputs)
    outputs.square().mean().backward()
    assert outputs.shape == inputs.shape
    assert inputs.grad is not None
