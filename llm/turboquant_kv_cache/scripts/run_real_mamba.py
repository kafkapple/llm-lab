#!/usr/bin/env python
"""Compare official Mamba and Transformer on real WikiText-2 bytes."""

from pathlib import Path
import math
import sys
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
from datasets import load_dataset
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qai_poc.real_data import byte_sequences
from qai_poc.sequence import TransformerModel
from qai_poc.ssm import OfficialMambaModel

ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGURES = ROOT / "results", ROOT / "figures"


def corpus(split, count, length):
    rows = load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1", split=split, streaming=True
    ).filter(lambda row: len(row["text"]) > 30).take(count)
    return byte_sequences([row["text"] for row in rows], length)


def run(model_class, train, valid, seed, device):
    torch.manual_seed(seed)
    model = (
        model_class(256, 48, train.size(1) - 1)
        if model_class is TransformerModel
        else model_class(256, 64, 16)
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    generator = torch.Generator().manual_seed(seed)
    warmup = train[:32].to(device)
    nn.functional.cross_entropy(
        model(warmup[:, :-1]).reshape(-1, 256), warmup[:, 1:].reshape(-1)
    ).backward()
    optimizer.zero_grad()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    losses, start = [], time.perf_counter()
    for _ in range(200):
        index = torch.randint(0, len(train), (32,), generator=generator)
        batch = train[index].to(device)
        loss = nn.functional.cross_entropy(
            model(batch[:, :-1]).reshape(-1, 256), batch[:, 1:].reshape(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    with torch.no_grad():
        scores = []
        for batch in valid.split(32):
            batch = batch.to(device)
            scores.append(float(nn.functional.cross_entropy(
                model(batch[:, :-1]).reshape(-1, 256), batch[:, 1:].reshape(-1)
            )))
    loss = sum(scores) / len(scores)
    return {
        "model": "Transformer" if model_class is TransformerModel else "Mamba",
        "seed": seed,
        "valid_loss": loss,
        "perplexity": math.exp(min(loss, 20)),
        "tokens_per_second": 200 * 32 * (train.size(1) - 1) / elapsed,
        "peak_gpu_mb": torch.cuda.max_memory_allocated() / 1e6,
        "parameters": sum(p.numel() for p in model.parameters()),
        "final_train_loss": sum(losses[-20:]) / 20,
    }


def main():
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    device = torch.device("cuda:0")
    train, valid = corpus("train", 2048, 128), corpus("validation", 512, 128)
    rows = [
        run(model, train, valid, seed, device)
        for seed in (0, 1, 2)
        for model in (TransformerModel, OfficialMambaModel)
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "real_wikitext_mamba.csv", index=False)
    summary = frame.groupby("model")[["perplexity", "tokens_per_second"]].agg(["mean", "std"])
    summary.to_csv(RESULTS / "real_wikitext_mamba_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for column, axis in zip(("perplexity", "tokens_per_second"), axes):
        values = frame.groupby("model")[column]
        axis.bar(values.mean().index, values.mean(), yerr=values.std(), capsize=4)
        axis.set_ylabel(column.replace("_", " "))
    fig.tight_layout()
    fig.savefig(FIGURES / "real_wikitext_mamba.png", dpi=150)
    print(summary)


if __name__ == "__main__":
    main()
