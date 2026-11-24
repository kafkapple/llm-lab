"""Validate and summarize paired confirmatory quality results."""

import argparse
import csv
import json
from pathlib import Path

from qai_poc.eval_utils import bootstrap_ci


METHODS = ("auto", "fp8_e4m3", "turboquant_k8v4", "turboquant_4bit_nc")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payloads = {
        method: json.loads((args.result_dir / f"{method}.json").read_text())
        for method in METHODS
    }
    baseline = payloads["auto"]
    expected_hash = baseline["config_hash"]
    expected_ids = [row["id"] for row in baseline["records"]]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("baseline contains duplicate IDs")
    metric = baseline["config"]["metric"]
    baseline_ok = baseline["mean_f1"] >= metric["baseline_floor"]
    rows = []
    for method in METHODS:
        payload = payloads[method]
        ids = [row["id"] for row in payload["records"]]
        if payload["config_hash"] != expected_hash or ids != expected_ids:
            raise ValueError(f"unpaired or config-drifted result: {method}")
        base = {row["id"]: row["f1"] for row in baseline["records"]}
        scores = {row["id"]: row["f1"] for row in payload["records"]}
        deltas = [scores[key] - base[key] for key in expected_ids]
        low, high = bootstrap_ci(
            deltas, metric["bootstrap_samples"], 20260728
        )
        rows.append({
            "kv_dtype": method,
            "n": len(deltas),
            "mean_f1": payload["mean_f1"],
            "delta_vs_bf16": sum(deltas) / len(deltas),
            "ci95_low": low,
            "ci95_high": high,
            "baseline_floor_pass": baseline_ok,
            "non_inferior": (
                baseline_ok and low >= metric["non_inferiority_margin"]
            ),
            "truncated_count": payload["truncated_count"],
            "length_finished_count": payload["length_finished_count"],
            "elapsed_seconds": payload["elapsed_seconds"],
            "config_hash": payload["config_hash"],
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=rows[0], lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
