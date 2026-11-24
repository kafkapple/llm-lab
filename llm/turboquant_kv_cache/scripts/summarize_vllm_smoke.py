"""Summarize vLLM smoke JSON and capacity lines into one CSV."""

import argparse
import csv
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--latency-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.result_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        log = path.with_suffix(".log").read_text()
        capacity = re.search(r"GPU KV cache size: ([\d,]+) tokens", log)
        concurrency = re.search(
            r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x", log
        )
        records = payload["records"]
        latency = (
            (args.latency_dir / f"{payload['kv_dtype']}.log").read_text()
            if args.latency_dir else ""
        )
        def latency_value(label: str):
            match = re.search(rf"{label}: ([\d.]+) seconds", latency)
            return float(match.group(1)) if match else ""
        rows.append({
            "model": payload["model"],
            "revision": payload.get("revision", ""),
            "kv_dtype": payload["kv_dtype"],
            "kv_tokens": int(capacity.group(1).replace(",", "")),
            "max_length_concurrency": float(concurrency.group(1)),
            "load_seconds": payload["load_seconds"],
            "generate_seconds_smoke": payload["generate_seconds"],
            "arithmetic_starts_42": records[0]["text"].startswith("42"),
            "capital_contains_paris": "Paris" in records[1]["text"],
            "needle_pass": payload["needle_pass"],
            "latency_mean_s": latency_value("Avg latency"),
            "latency_p50_s": latency_value("50% percentile latency"),
            "latency_p99_s": latency_value("99% percentile latency"),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
