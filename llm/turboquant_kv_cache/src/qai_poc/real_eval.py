"""Shared evaluation helpers for real-data notebooks."""

import copy

import numpy as np
import torch
from torch import nn

from .quantization import dequantize, quantize


def quantize_linear_weights(model: nn.Module, bits: int, group_size: int = 128):
    """Return a fake-quantized copy of every Linear layer."""
    restored = copy.deepcopy(model)
    with torch.no_grad():
        for module in restored.modules():
            if isinstance(module, nn.Linear):
                array = module.weight.detach().float().cpu().numpy()
                values = dequantize(quantize(array, bits, group_size))
                module.weight.copy_(torch.from_numpy(values))
    return restored


@torch.no_grad()
def classification_accuracy(model, batches, device: torch.device) -> float:
    """Evaluate Hugging Face sequence classifiers."""
    if not getattr(model, "is_loaded_in_4bit", False):
        model.to(device)
    model.eval()
    correct = total = 0
    for encoded, labels in batches:
        predicted = model(**{k: v.to(device) for k, v in encoded.items()}).logits.argmax(-1)
        correct += int((predicted.cpu() == labels).sum())
        total += len(labels)
    return correct / total


def linear_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    train_count: int,
    classes: int,
    seed: int,
    steps: int = 150,
) -> float:
    """Train one linear classifier and report held-out accuracy."""
    torch.manual_seed(seed)
    probe = nn.Linear(features.size(1), classes, device=features.device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=0.03)
    for _ in range(steps):
        loss = nn.functional.cross_entropy(probe(features[:train_count]), labels[:train_count])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return float((probe(features[train_count:]).argmax(-1) == labels[train_count:]).float().mean())


def mean_std(values) -> tuple[float, float]:
    """Return population mean and standard deviation."""
    return float(np.mean(values)), float(np.std(values))
