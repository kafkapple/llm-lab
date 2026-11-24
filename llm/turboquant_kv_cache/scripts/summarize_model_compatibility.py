"""Summarize the locked 1B-family TurboQuant compatibility smoke."""

import argparse
import csv
import json
import re
import tomllib
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    rows = []
    for model in config["model"]:
        prefix = model["family"].lower()
        for dtype in config["experiment"]["kv_dtypes"]:
            suffix = "auto" if dtype == "auto" else "k8v4"
            stem = args.result_dir / f"{prefix}_{suffix}"
            log = stem.with_suffix(".log").read_text()
            capacity = re.search(r"GPU KV cache size: ([\d,]+) tokens", log)
            payload_path = stem.with_suffix(".json")
            if payload_path.exists():
                payload = json.loads(payload_path.read_text())
                texts = [record["text"] for record in payload["records"]]
                status = "executed"
                failure = ""
            else:
                texts = ["", "", ""]
                status = "unsupported"
                errors = re.findall(r"ValueError: (.+)", log)
                failure = errors[-1] if errors else "engine initialization failed"
            rows.append({
                "family": model["family"],
                "model": model["id"],
                "parameters": model["parameters"],
                "revision": model["revision"],
                "kv_dtype": dtype,
                "status": status,
                "kv_tokens": (
                    int(capacity.group(1).replace(",", "")) if capacity else ""
                ),
                "arithmetic_pass": texts[0].strip().startswith("42"),
                "capital_pass": "Paris" in texts[1],
                "needle_pass": "ORANGE-7319" in texts[2],
                "failure_reason": failure,
            })
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
