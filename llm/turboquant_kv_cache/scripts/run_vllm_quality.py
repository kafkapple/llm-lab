"""Run one KV dtype from a locked LongBench confirmatory configuration."""

import argparse
import json
import random
import time
from pathlib import Path

from qai_poc.eval_utils import canonical_hash, load_toml, sha256_file, token_f1
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def render_prompt(row: dict, tokenizer, template: str, budget: int) -> tuple[str, dict]:
    raw = template.format(context=row["context"], input=row["input"])
    raw_ids = tokenizer.encode(raw, add_special_tokens=False)
    chat_overhead = len(tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}],
        tokenize=True,
        add_generation_prompt=True,
    ))
    raw_budget = budget - chat_overhead
    truncated = len(raw_ids) > raw_budget
    if truncated:
        half = raw_budget // 2
        raw_ids = raw_ids[:half] + raw_ids[-(raw_budget - half):]
        raw = tokenizer.decode(raw_ids, skip_special_tokens=True)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": raw}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt, {
        "original_tokens": len(tokenizer.encode(
            template.format(context=row["context"], input=row["input"]),
            add_special_tokens=False,
        )),
        "truncated": truncated,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--kv-dtype", required=True)
    parser.add_argument("--kv-skip-layers", default="", help="Comma-separated layer indices or attention type names (e.g. 'sliding_window' or '0,1,2')")
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
    rendered = [
        render_prompt(
            row, tokenizer, exp["prompt_template"],
            exp["max_model_len"] - exp["max_output_tokens"] - 1,
        )
        for row in sample
    ]
    prompts = [item[0] for item in rendered]
    llm = LLM(
        model=exp["model"],
        revision=provenance["model_revision"],
        kv_cache_dtype=args.kv_dtype,
        kv_cache_dtype_skip_layers=skip_layers,
        max_model_len=exp["max_model_len"],
        gpu_memory_utilization=exp["gpu_memory_utilization"],
        enforce_eager=True,
        trust_remote_code=False,
    )
    resolved_skip_layers = llm.llm_engine.vllm_config.cache_config.kv_cache_dtype_skip_layers
    started = time.perf_counter()
    outputs = llm.generate(
        prompts,
        SamplingParams(
            temperature=0, max_tokens=exp["max_output_tokens"], seed=0
        ),
    )
    records = []
    for row, metadata, output in zip(sample, rendered, outputs):
        prediction = output.outputs[0].text.strip()
        answers = row["answers"]
        records.append({
            "id": row["_id"],
            **metadata[1],
            "prompt_tokens": len(output.prompt_token_ids),
            "output_tokens": len(output.outputs[0].token_ids),
            "finish_reason": output.outputs[0].finish_reason,
            "prediction": prediction,
            "answers": answers,
            "f1": max(token_f1(prediction, answer) for answer in answers),
        })
    payload = {
        "schema_version": 2,
        "config_hash": canonical_hash(config),
        "config": config,
        "data_sha256": sha256_file(args.data),
        "script_sha256": sha256_file(Path(__file__)),
        "kv_dtype": args.kv_dtype,
        "kv_skip_layers_requested": skip_layers,
        "kv_skip_layers_resolved": resolved_skip_layers,
        "elapsed_seconds": time.perf_counter() - started,
        "mean_f1": sum(row["f1"] for row in records) / len(records),
        "truncated_count": sum(row["truncated"] for row in records),
        "length_finished_count": sum(
            row["finish_reason"] == "length" for row in records
        ),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "records"}))


if __name__ == "__main__":
    main()
