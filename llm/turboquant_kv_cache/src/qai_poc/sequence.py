"""Minimal sequence-model baselines for a copy-memory task."""

import torch
from torch import nn


class TransformerModel(nn.Module):
    """Small Transformer encoder baseline."""

    def __init__(self, vocab: int, width: int, length: int):
        super().__init__()
        self.token = nn.Embedding(vocab, width)
        self.position = nn.Parameter(torch.randn(1, length, width) * 0.01)
        layer = nn.TransformerEncoderLayer(width, 4, width * 2, batch_first=True)
        self.body = nn.TransformerEncoder(layer, 2)
        self.head = nn.Linear(width, vocab)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Predict a token at every sequence position."""
        return self.head(self.body(self.token(tokens) + self.position[:, : tokens.size(1)]))


class FNetModel(nn.Module):
    """Token mixer using Fourier transforms instead of attention."""

    def __init__(self, vocab: int, width: int):
        super().__init__()
        self.token = nn.Embedding(vocab, width)
        self.ffn = nn.Sequential(
            nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width)
        )
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Mix sequence and feature axes with a real-valued 2-D FFT."""
        hidden = self.token(tokens)
        mixed = torch.fft.fft(torch.fft.fft(hidden, dim=1), dim=2).real
        return self.head(self.norm(hidden + mixed + self.ffn(mixed)))


class SelectiveSSM(nn.Module):
    """Educational input-gated recurrent state model, not production Mamba."""

    def __init__(self, vocab: int, width: int):
        super().__init__()
        self.token = nn.Embedding(vocab, width)
        self.gate = nn.Linear(width, width)
        self.candidate = nn.Linear(width, width)
        self.head = nn.Linear(width, vocab)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Scan linearly while input-dependent gates select state updates."""
        embedded = self.token(tokens)
        state = torch.zeros_like(embedded[:, 0])
        outputs = []
        for current in embedded.unbind(dim=1):
            gate = torch.sigmoid(self.gate(current))
            proposal = torch.tanh(self.candidate(current))
            state = gate * state + (1 - gate) * proposal
            outputs.append(state)
        return self.head(torch.stack(outputs, dim=1))


def copy_batch(
    batch_size: int, length: int, vocab: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create sequences whose second half repeats the first half."""
    half = length // 2
    first = torch.randint(1, vocab, (batch_size, half), device=device)
    tokens = torch.cat((first, first), dim=1)
    return tokens[:, :-1], tokens[:, 1:]
