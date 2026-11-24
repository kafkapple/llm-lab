"""One-process vLLM KV-cache smoke benchmark; run once per dtype."""

import argparse
import json
import time
from pathlib import Path

from vllm import LLM, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kv-dtype", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--haystack-repeat", type=int, default=300)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.40)
    args = parser.parse_args()

    secret = "ORANGE-7319"
    haystack = (
        "Long-context systems must count actual cache bytes. "
        * args.haystack_repeat
    )
    prompts = [
        "Reply with only the number: 17 + 25 =",
        "Reply with only the capital: The capital of France is",
        f"Remember this code: {secret}. {haystack} Reply with only the code:",
    ]
    start_load = time.perf_counter()
    llm = LLM(
        model=args.model,
        revision=args.revision,
        kv_cache_dtype=args.kv_dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        trust_remote_code=False,
    )
    load_seconds = time.perf_counter() - start_load
    tokenizer = llm.get_tokenizer()
    formatted_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    start_generate = time.perf_counter()
    outputs = llm.generate(
        formatted_prompts, SamplingParams(temperature=0, max_tokens=32, seed=0)
    )
    generate_seconds = time.perf_counter() - start_generate
    records = [
        {
            "prompt_tokens": len(output.prompt_token_ids),
            "output_tokens": len(output.outputs[0].token_ids),
            "text": output.outputs[0].text.strip(),
        }
        for output in outputs
    ]
    payload = {
        "model": args.model,
        "revision": args.revision,
        "kv_dtype": args.kv_dtype,
        "chat_template": True,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "load_seconds": load_seconds,
        "generate_seconds": generate_seconds,
        "total_output_tokens": sum(row["output_tokens"] for row in records),
        "needle_expected": secret,
        "needle_pass": secret in records[-1]["text"],
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
