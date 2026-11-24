"""State-space progression from linear recurrence to selective scan."""

import torch
from torch import nn


class SelectiveScan(nn.Module):
    """Input-dependent diagonal SSM using the Mamba discretization idea."""

    def __init__(self, width: int, state_size: int = 16):
        super().__init__()
        self.state_size = state_size
        self.log_a = nn.Parameter(torch.log(torch.arange(1, state_size + 1).float()))
        self.delta = nn.Linear(width, width)
        self.b = nn.Linear(width, state_size)
        self.c = nn.Linear(width, state_size)
        self.skip = nn.Parameter(torch.ones(width))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply a sequential selective scan with input-dependent Δ, B and C."""
        batch, _, width = inputs.shape
        state = inputs.new_zeros(batch, width, self.state_size)
        a = -torch.exp(self.log_a)[None, None, :]
        outputs = []
        for current in inputs.unbind(1):
            delta = nn.functional.softplus(self.delta(current))[..., None]
            transition = torch.exp(delta * a)
            b = self.b(current)[:, None, :]
            state = transition * state + delta * b * current[..., None]
            c = self.c(current)[:, None, :]
            outputs.append((state * c).sum(-1) + self.skip * current)
        return torch.stack(outputs, 1)


class SelectiveSSMModel(nn.Module):
    """Language-model wrapper around the educational selective scan."""

    def __init__(self, vocab: int, width: int, state_size: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(vocab, width)
        self.input_projection = nn.Linear(width, width * 2)
        self.scan = SelectiveScan(width, state_size)
        self.output_projection = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Predict tokens with gated convolution-free selective state mixing."""
        projected, gate = self.input_projection(self.embedding(tokens)).chunk(2, -1)
        hidden = self.scan(projected) * nn.functional.silu(gate)
        return self.head(self.norm(self.output_projection(hidden)))


class OfficialMambaModel(nn.Module):
    """Thin wrapper around the official state-spaces/mamba block."""

    def __init__(self, vocab: int, width: int, state_size: int = 16):
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except ImportError as error:
            raise ImportError("Install mamba-ssm to use OfficialMambaModel") from error
        self.embedding = nn.Embedding(vocab, width)
        self.mamba = Mamba(d_model=width, d_state=state_size, d_conv=4, expand=2)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Predict tokens with the official optimized Mamba block."""
        return self.head(self.norm(self.mamba(self.embedding(tokens))))
