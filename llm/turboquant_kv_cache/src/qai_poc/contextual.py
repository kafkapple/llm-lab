"""Tiny masked teacher-student contextual representation experiment."""

import torch
from torch import nn


class ContextEncoder(nn.Module):
    """Encode a sequence into contextual latent vectors."""

    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one contextual vector per position."""
        hidden = self.net[:2](x)
        return self.net[2](hidden.transpose(1, 2)).transpose(1, 2)


def make_sine_batch(
    batch_size: int, length: int, input_dim: int, device: torch.device
) -> torch.Tensor:
    """Create structured multi-channel sequences without external data."""
    time = torch.linspace(0, 2 * torch.pi, length, device=device)
    frequency = torch.rand(batch_size, 1, input_dim, device=device) * 3 + 0.5
    phase = torch.rand(batch_size, 1, input_dim, device=device) * 2 * torch.pi
    return torch.sin(time[None, :, None] * frequency + phase)


def train_step(
    student: nn.Module,
    teacher: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: torch.Tensor,
    mask_ratio: float = 0.5,
    ema: float = 0.999,
) -> float:
    """Predict teacher latents at masked positions and update the EMA teacher."""
    mask = torch.rand(batch.shape[:2], device=batch.device) < mask_ratio
    masked = batch.masked_fill(mask[..., None], 0)
    with torch.no_grad():
        target = teacher(batch)
    prediction = student(masked)
    loss = (prediction[mask] - target[mask]).square().mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        for teacher_p, student_p in zip(teacher.parameters(), student.parameters()):
            teacher_p.mul_(ema).add_(student_p, alpha=1 - ema)
    return float(loss.detach())
