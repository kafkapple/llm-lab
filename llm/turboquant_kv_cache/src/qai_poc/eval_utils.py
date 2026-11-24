"""Small, dependency-free helpers shared by confirmatory benchmarks."""

import hashlib
import json
import random
import re
import string
import tomllib
from collections import Counter
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: dict) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def load_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, answer: str) -> float:
    pred = normalize_answer(prediction).split()
    gold = normalize_answer(answer).split()
    overlap = sum((Counter(pred) & Counter(gold)).values())
    if not pred or not gold:
        return float(pred == gold)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def bootstrap_ci(
    values: list[float], samples: int, seed: int
) -> tuple[float, float]:
    if not values:
        raise ValueError("bootstrap requires at least one value")
    rng = random.Random(seed)
    means = sorted(
        sum(rng.choice(values) for _ in values) / len(values)
        for _ in range(samples)
    )
    return means[round((samples - 1) * 0.025)], means[
        round((samples - 1) * 0.975)
    ]
