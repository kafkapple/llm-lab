"""Aggregate serving metrics across independent server restarts."""

import argparse
import csv
import json
import statistics
from pathlib import Path

from qai_poc.eval_utils import bootstrap_ci


METRICS = ("request_throughput", "p99_ttft_ms", "p99_tpot_ms", "p99_e2el_ms")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payloads = [
        json.loads(path.read_text())
        for path in args.result_dir.glob("r*_*.json")
        if not path.name.endswith("schedule.json")
    ]
    if not payloads:
        raise ValueError("no serving results")
    hashes = {item["confirmatory"]["config_hash"] for item in payloads}
    if len(hashes) != 1:
        raise ValueError("serving configuration drift detected")
    groups = {}
    for item in payloads:
        meta = item["confirmatory"]
        workloads = {
            workload["name"]: workload
            for workload in meta["config"]["workload"]
        }
        if meta["workload"] not in workloads:
            raise ValueError(f"unknown workload: {meta['workload']}")
        if (
            item["completed"] != workloads[meta["workload"]]["num_prompts"]
            or item["failed"]
        ):
            raise ValueError("failed or incomplete serving run")
        groups.setdefault((meta["workload"], meta["kv_dtype"]), []).append(item)
    rows = []
    for (workload, method), items in sorted(groups.items()):
        repeats = {item["confirmatory"]["repeat"] for item in items}
        expected = items[0]["confirmatory"]["config"]["experiment"]["restarts"]
        if len(items) != expected or len(repeats) != expected:
            raise ValueError(f"missing independent restart: {workload}/{method}")
        row = {
            "workload": workload,
            "kv_dtype": method,
            "restarts": len(items),
            "jit_contaminated_runs": sum(
                item["confirmatory"]["jit_during_measurement"] for item in items
            ),
            "config_hash": items[0]["confirmatory"]["config_hash"],
        }
        for metric in METRICS:
            values = [item[metric] for item in items]
            low, high = bootstrap_ci(values, 10000, 20260728)
            row.update({
                f"{metric}_mean": statistics.mean(values),
                f"{metric}_min": min(values),
                f"{metric}_max": max(values),
                f"{metric}_ci95_low": low,
                f"{metric}_ci95_high": high,
            })
        rows.append(row)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=rows[0], lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
