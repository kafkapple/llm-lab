"""A compact data2vec-style Transformer teacher and multimodal toy data."""

import torch
from torch import nn


class ContextTransformer(nn.Module):
    """Return an average of the last contextual Transformer layers."""

    def __init__(
        self, input_dim: int, width: int = 64, depth: int = 4, target_layers: int = 2
    ):
        super().__init__()
        self.projection = nn.Linear(input_dim, width)
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(width, 4, width * 2, batch_first=True)
            for _ in range(depth)
        )
        self.target_layers = target_layers
        self.norm = nn.LayerNorm(width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode inputs and average normalized top-layer representations."""
        hidden = self.projection(inputs)
        outputs = []
        for layer in self.layers:
            hidden = layer(hidden)
            outputs.append(self.norm(hidden))
        return torch.stack(outputs[-self.target_layers :]).mean(0)


class PointwiseBaseline(nn.Module):
    """Non-contextual baseline that cannot mix neighboring positions."""

    def __init__(self, input_dim: int, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, width), nn.GELU(), nn.Linear(width, width)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode each position independently."""
        return self.net(inputs)


def make_modality_batch(
    modality: str,
    batch_size: int,
    length: int,
    input_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Create speech-, vision-, or text-like structured float sequences."""
    if modality == "speech":
        time = torch.linspace(0, 4 * torch.pi, length, device=device)
        frequency = torch.rand(batch_size, 1, input_dim, device=device) * 2 + 0.5
        return torch.sin(time[None, :, None] * frequency)
    if modality == "vision":
        side = int(length**0.5)
        if side * side != length:
            raise ValueError("vision length must be a square")
        grid = torch.linspace(-1, 1, side, device=device)
        y, x = torch.meshgrid(grid, grid, indexing="ij")
        center = torch.rand(batch_size, 2, device=device) - 0.5
        radius = (x[None] - center[:, 0, None, None]) ** 2
        radius += (y[None] - center[:, 1, None, None]) ** 2
        image = torch.exp(-radius[..., None] * torch.linspace(1, 3, input_dim, device=device))
        return image.reshape(batch_size, length, input_dim)
    if modality == "text":
        states = torch.randint(0, input_dim, (batch_size, 1), device=device)
        tokens = [states]
        for _ in range(length - 1):
            states = (states + torch.randint(0, 3, states.shape, device=device)) % input_dim
            tokens.append(states)
        return nn.functional.one_hot(torch.cat(tokens, 1), input_dim).float()
    raise ValueError("modality must be speech, vision, or text")


def span_mask(
    batch_size: int,
    length: int,
    ratio: float,
    span: int,
    device: torch.device,
) -> torch.Tensor:
    """Create contiguous masks approximating data2vec block masking."""
    mask = torch.zeros(batch_size, length, dtype=torch.bool, device=device)
    starts = torch.randint(
        0,
        max(1, length - span + 1),
        (batch_size, max(1, int(length * ratio / span))),
        device=device,
    )
    for row, row_starts in enumerate(starts):
        for start in row_starts:
            mask[row, start : start + span] = True
    return mask


def representation_step(
    student: nn.Module,
    teacher: nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    mask_ratio: float = 0.5,
    span: int = 4,
    ema: float | None = 0.999,
) -> float:
    """Train a masked student against contextual teacher representations."""
    mask = span_mask(inputs.size(0), inputs.size(1), mask_ratio, span, inputs.device)
    masked = inputs.masked_fill(mask[..., None], 0)
    with torch.no_grad():
        targets = teacher(inputs)
    predictions = student(masked)
    loss = (predictions[mask] - targets[mask]).square().mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if ema is not None:
        with torch.no_grad():
            for teacher_p, student_p in zip(teacher.parameters(), student.parameters()):
                teacher_p.mul_(ema).add_(student_p, alpha=1 - ema)
    return float(loss.detach())
