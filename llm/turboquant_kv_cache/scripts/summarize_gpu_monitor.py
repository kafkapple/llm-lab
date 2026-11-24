"""Summarize the target GPU rows from monitor_gpu.sh output."""

import argparse
import csv
from pathlib import Path


FIELDS = ("memory_used_mib", "utilization_pct", "temperature_c", "power_w")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    with args.input.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                if int(row["index"].strip()) != args.gpu:
                    continue
                values = {
                    field: float(row[field].strip())
                    for field in FIELDS
                }
            except (AttributeError, KeyError, TypeError, ValueError):
                continue
            rows.append({"timestamp": row["timestamp"].strip(), **values})
    if not rows:
        raise ValueError(f"no complete rows for GPU {args.gpu}")
    summary = {"gpu": args.gpu, "samples": len(rows)}
    for field in FIELDS:
        values = [row[field] for row in rows]
        summary[f"{field}_min"] = min(values)
        summary[f"{field}_mean"] = sum(values) / len(values)
        summary[f"{field}_max"] = max(values)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary)
        writer.writeheader()
        writer.writerow(summary)


if __name__ == "__main__":
    main()
