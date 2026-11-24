"""Dump top-K per-token logprobs for one KV dtype/skip-layers config.

Companion to run_vllm_quality.py — same config/data contract, but records
logprob distributions instead of F1, for compute_kl.py to compare against
a reference (native dtype) run. Reuses render_prompt from run_vllm_quality.
"""

import argparse
import json
import random
import time
from pathlib import Path

from qai_poc.eval_utils import canonical_hash, load_toml, sha256_file
from run_vllm_quality import render_prompt
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--kv-dtype", required=True)
    parser.add_argument("--kv-skip-layers", default="")
    parser.add_argument("--top-k", type=int, default=40, help="matches oobabooga localbench methodology (F5)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    skip_layers = [s for s in args.kv_skip_layers.split(",") if s]
    config = load_toml(args.config)
    exp, provenance = config["experiment"], config["provenance"]
    if sha256_file(args.data) != provenance["data_sha256"]:
        raise ValueError("LongBench data checksum mismatch")
    rows = [json.loads(line) for line in args.data.read_text().splitlines()]
    sample = random.Random(exp["sample_seed"]).sample(rows, exp["sample_size"])
    tokenizer = AutoTokenizer.from_pretrained(
        exp["model"], revision=provenance["model_revision"], trust_remote_code=False
    )
    prompts = [
        render_prompt(row, tokenizer, exp["prompt_template"], exp["max_model_len"] - exp["max_output_tokens"] - 1)[0]
        for row in sample
    ]
    llm = LLM(
        model=exp["model"],
        revision=provenance["model_revision"],
        kv_cache_dtype=args.kv_dtype,
        kv_cache_dtype_skip_layers=skip_layers,
        max_logprobs=args.top_k,
        max_model_len=exp["max_model_len"],
        gpu_memory_utilization=exp["gpu_memory_utilization"],
        enforce_eager=True,
        trust_remote_code=False,
    )
    resolved_skip_layers = llm.llm_engine.vllm_config.cache_config.kv_cache_dtype_skip_layers
    started = time.perf_counter()
    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0, max_tokens=exp["max_output_tokens"], seed=0, logprobs=args.top_k),
    )
    records = []
    for row, output in zip(sample, outputs):
        positions = []
        for step_logprobs in output.outputs[0].logprobs or []:
            positions.append({str(tid): lp.logprob for tid, lp in step_logprobs.items()})
        records.append({"id": row["_id"], "positions": positions})
    payload = {
        "schema_version": 1,
        "config_hash": canonical_hash(config),
        "data_sha256": sha256_file(args.data),
        "script_sha256": sha256_file(Path(__file__)),
        "kv_dtype": args.kv_dtype,
        "kv_skip_layers_requested": skip_layers,
        "kv_skip_layers_resolved": resolved_skip_layers,
        "top_k": args.top_k,
        "elapsed_seconds": time.perf_counter() - started,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "records"}))


if __name__ == "__main__":
    main()
