"""Small, deterministic adapters for public text, image, and speech datasets."""

from io import BytesIO
from itertools import islice

import numpy as np
import torch


def take_stream(dataset: str, split: str, count: int, config: str | None = None):
    """Take a fixed prefix from a Hugging Face streaming dataset."""
    from datasets import load_dataset

    return list(islice(load_dataset(dataset, config, split=split, streaming=True), count))


def byte_sequences(texts: list[str], length: int = 128) -> torch.Tensor:
    """Encode real UTF-8 text without a learned tokenizer."""
    rows = []
    for text in texts:
        values = list(text.encode("utf-8")[: length + 1])
        rows.append(values + [0] * (length + 1 - len(values)))
    return torch.tensor(rows, dtype=torch.long)


def image_patches(images, patch: int = 8) -> torch.Tensor:
    """Convert PIL images to RGB patch sequences."""
    rows = []
    for image in images:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        height, width, channels = array.shape
        patches = array.reshape(height // patch, patch, width // patch, patch, channels)
        rows.append(patches.transpose(0, 2, 1, 3, 4).reshape(-1, patch * patch * channels))
    return torch.tensor(np.stack(rows))


def audio_features(audio_items, frames: int = 32, bins: int = 32) -> torch.Tensor:
    """Decode dataset audio bytes and return fixed-size log-spectrum frames."""
    import soundfile as sf

    rows = []
    target_samples = frames * 256
    for item in audio_items:
        source = BytesIO(item["bytes"]) if item.get("bytes") else item["path"]
        waveform, _ = sf.read(source, dtype="float32")
        waveform = waveform.mean(1) if waveform.ndim == 2 else waveform
        positions = np.linspace(0, max(0, len(waveform) - 1), target_samples)
        resampled = np.interp(positions, np.arange(len(waveform)), waveform)
        spectra = np.abs(np.fft.rfft(resampled.reshape(frames, 256), axis=1))[:, :bins]
        rows.append(np.log1p(spectra))
    return torch.tensor(np.stack(rows), dtype=torch.float32)
