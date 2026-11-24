#!/usr/bin/env python
"""Evaluate low-bit RTN on a real fine-tuned Transformer and SST-2."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qai_poc.real_eval import classification_accuracy, quantize_linear_weights

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"


def main():
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    rows = list(load_dataset("stanfordnlp/sst2", split="validation", streaming=True).take(512))
    batches = []
    for start in range(0, len(rows), 32):
        part = rows[start : start + 32]
        encoded = tokenizer(
            [row["sentence"] for row in part],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        batches.append((encoded, torch.tensor([row["label"] for row in part])))

    base = AutoModelForSequenceClassification.from_pretrained(MODEL)
    total_values = sum(parameter.numel() for parameter in base.parameters())
    linear_values = sum(
        module.weight.numel()
        for module in base.modules()
        if isinstance(module, torch.nn.Linear)
    )
    output = []
    for bits in (32, 8, 4, 2):
        model = base if bits == 32 else quantize_linear_weights(base, bits)
        torch.cuda.reset_peak_memory_stats()
        accuracy = classification_accuracy(model, batches, device)
        output.append(
            {
                "method": "fp32" if bits == 32 else f"rtn_fake_{bits}bit",
                "bits": bits,
                "accuracy": accuracy,
                "estimated_model_mb": (
                    linear_values * bits / 8 + (total_values - linear_values) * 4
                )
                / 1e6,
                "peak_gpu_mb": torch.cuda.max_memory_allocated() / 1e6,
                "samples": len(rows),
                "model": MODEL,
            }
        )
        model.cpu()
        if bits != 32:
            del model
        torch.cuda.empty_cache()

    packed = AutoModelForSequenceClassification.from_pretrained(
        MODEL,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        device_map={"": 0},
    )
    torch.cuda.reset_peak_memory_stats()
    output.append(
        {
            "method": "bnb_nf4_packed",
            "bits": 4,
            "accuracy": classification_accuracy(packed, batches, device),
            "estimated_model_mb": packed.get_memory_footprint() / 1e6,
            "peak_gpu_mb": torch.cuda.max_memory_allocated() / 1e6,
            "samples": len(rows),
            "model": MODEL,
        }
    )
    frame = pd.DataFrame(output)
    frame.to_csv(RESULTS / "real_quantization_sst2.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].bar(frame.method, frame.accuracy)
    axes[0].tick_params(axis="x", rotation=55)
    axes[0].set(ylabel="SST-2 accuracy", ylim=(0, 1))
    axes[1].scatter(frame.estimated_model_mb, frame.accuracy)
    for row in frame.itertuples():
        axes[1].annotate(row.method, (row.estimated_model_mb, row.accuracy), fontsize=7)
    axes[1].set(xlabel="Estimated model footprint (MB)", ylabel="Accuracy")
    fig.tight_layout()
    fig.savefig(FIGURES / "real_quantization_sst2.png", dpi=150)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
