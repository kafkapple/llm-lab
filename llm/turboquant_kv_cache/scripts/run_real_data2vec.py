#!/usr/bin/env python
"""Run compact data2vec-style learning and probes on three real modalities."""

from io import BytesIO
from pathlib import Path
import copy
import sys

import matplotlib.pyplot as plt
import pandas as pd
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qai_poc.data2vec import ContextTransformer, representation_step
from qai_poc.real_data import audio_features, byte_sequences, image_patches
from qai_poc.real_eval import linear_probe

ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGURES = ROOT / "results", ROOT / "figures"


def load_modalities(count=360):
    text_ds = load_dataset("stanfordnlp/sst2", split="train", streaming=True)
    vision_ds = load_dataset("uoft-cs/cifar10", split="train", streaming=True)
    speech_ds = load_dataset("PolyAI/minds14", "en-US", split="train", streaming=True)
    text = list(text_ds.shuffle(seed=7, buffer_size=2000).take(count))
    vision = list(vision_ds.shuffle(seed=7, buffer_size=2000).take(count))
    speech = list(
        speech_ds.cast_column("audio", Audio(decode=False))
        .shuffle(seed=7, buffer_size=2000)
        .take(count)
    )
    text_tokens = byte_sequences([row["sentence"] for row in text], 63)
    text_x = nn.functional.one_hot(text_tokens[:, :-1], 256).float()
    return {
        "text": (text_x, torch.tensor([row["label"] for row in text]), 2),
        "vision": (
            image_patches([row["img"] for row in vision]),
            torch.tensor([row["label"] for row in vision]),
            10,
        ),
        "speech": (
            audio_features([row["audio"] for row in speech]),
            torch.tensor([row["intent_class"] for row in speech]),
            14,
        ),
    }


def run_one(name, inputs, labels, classes, seed, device):
    torch.manual_seed(seed)
    inputs, labels = inputs.to(device), labels.to(device)
    student = ContextTransformer(inputs.size(-1), 64, 3, 2).to(device)
    teacher = copy.deepcopy(student)
    optimizer = torch.optim.AdamW(student.parameters(), lr=2e-3)
    losses = []
    generator = torch.Generator().manual_seed(seed)
    for _ in range(100):
        index = torch.randint(0, 288, (32,), generator=generator)
        losses.append(representation_step(student, teacher, optimizer, inputs[index]))
    with torch.no_grad():
        learned = teacher(inputs).mean(1)
        raw = inputs.mean(1)
    return [
        {
            "modality": name,
            "representation": kind,
            "seed": seed,
            "accuracy": linear_probe(features, labels, 288, classes, seed),
            "ssl_initial": losses[0],
            "ssl_final": sum(losses[-10:]) / 10,
            "train_samples": 288,
            "test_samples": 72,
        }
        for kind, features in (("raw_mean", raw), ("data2vec", learned))
    ]


def main():
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rows = []
    for name, (inputs, labels, classes) in load_modalities().items():
        for seed in (0, 1, 2):
            rows.extend(run_one(name, inputs, labels, classes, seed, device))
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "real_data2vec_probes.csv", index=False)
    summary = frame.groupby(["modality", "representation"]).accuracy.agg(["mean", "std"])
    summary.to_csv(RESULTS / "real_data2vec_summary.csv")
    chart = summary["mean"].unstack().plot.bar(yerr=summary["std"].unstack(), figsize=(7, 4))
    chart.set(ylabel="Held-out linear-probe accuracy", ylim=(0, 1))
    chart.figure.tight_layout()
    chart.figure.savefig(FIGURES / "real_data2vec_probes.png", dpi=150)
    print(summary)


if __name__ == "__main__":
    main()
