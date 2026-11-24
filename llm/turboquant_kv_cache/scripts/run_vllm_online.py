"""Run randomized, warmed, repeated vLLM serving benchmarks."""

import argparse
import json
import os
import random
import subprocess
import time
import urllib.request
from pathlib import Path

from qai_poc.eval_utils import canonical_hash, load_toml, sha256_file


METHODS = ("auto", "fp8_e4m3", "turboquant_k8v4", "turboquant_4bit_nc")


def wait_ready(url: str, process: subprocess.Popen, timeout: int = 240) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with code {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2):
                return
        except Exception:
            time.sleep(2)
    raise TimeoutError("vLLM server did not become ready")


def bench_command(config: dict, workload: dict, port: int) -> list[str]:
    exp = config["experiment"]
    return [
        "vllm", "bench", "serve",
        "--backend", "vllm",
        "--base-url", f"http://127.0.0.1:{port}",
        "--model", exp["model"],
        "--dataset-name", "random",
        "--random-input-len", str(workload["input_tokens"]),
        "--random-output-len", str(workload["output_tokens"]),
        "--request-rate", "inf",
        "--max-concurrency", str(workload["max_concurrency"]),
        "--temperature", "0",
        "--seed", str(exp["seed"]),
        "--metric-percentiles", "50,95,99",
        "--percentile-metrics", "ttft,tpot,e2el",
        "--disable-tqdm",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--gpu", default="3")
    parser.add_argument("--port", type=int, default=8012)
    args = parser.parse_args()
    config = load_toml(args.config)
    exp = config["experiment"]
    config_hash = canonical_hash(config)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": args.gpu,
        "TOKENIZERS_PARALLELISM": "false",
    }
    rng = random.Random(exp["seed"])
    schedule = []
    for repeat in range(exp["restarts"]):
        methods = list(METHODS)
        rng.shuffle(methods)
        schedule.extend((repeat, method) for method in methods)
    (args.result_dir / "schedule.json").write_text(json.dumps(schedule))

    for repeat, dtype in schedule:
        stem = f"r{repeat}_{dtype}"
        log_path = args.result_dir / f"{stem}.server.log"
        with log_path.open("w+") as log:
            server = subprocess.Popen(
                [
                    "vllm", "serve", exp["model"],
                    "--revision", config["provenance"]["model_revision"],
                    "--kv-cache-dtype", dtype,
                    "--max-model-len", str(exp["max_model_len"]),
                    "--gpu-memory-utilization", str(exp["gpu_memory_utilization"]),
                    "--enforce-eager",
                    "--port", str(args.port),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
            )
            try:
                wait_ready(f"http://127.0.0.1:{args.port}/health", server)
                for workload in config["workload"]:
                    command = bench_command(config, workload, args.port)
                    warmup_prompts = (
                        exp["turboquant_short_warmup_prompts"]
                        if dtype.startswith("turboquant")
                        and workload["name"] == "short"
                        else exp["warmup_prompts"]
                    )
                    subprocess.run(
                        command + [
                            "--num-prompts", str(warmup_prompts),
                        ],
                        check=True,
                        env=env,
                        stdout=subprocess.DEVNULL,
                    )
                    log.flush()
                    result_name = f"{stem}_{workload['name']}.json"
                    jit_retries = 0
                    while True:
                        log.seek(0, os.SEEK_END)
                        measured_from = log.tell()
                        subprocess.run(
                            command + [
                                "--num-prompts", str(workload["num_prompts"]),
                                "--save-result",
                                "--save-detailed",
                                "--result-dir", str(args.result_dir),
                                "--result-filename", result_name,
                            ],
                            check=True,
                            env=env,
                        )
                        log.flush()
                        log.seek(0, os.SEEK_END)
                        measured_to = log.tell()
                        log.seek(measured_from)
                        measured_log = log.read(measured_to - measured_from)
                        jit_detected = (
                            "JIT compilation during inference" in measured_log
                        )
                        if (
                            not jit_detected
                            or jit_retries >= exp["jit_clean_retries"]
                        ):
                            break
                        jit_retries += 1
                    result_path = args.result_dir / result_name
                    payload = json.loads(result_path.read_text())
                    payload["confirmatory"] = {
                        "schema_version": 2,
                        "config_hash": config_hash,
                        "config": config,
                        "repeat": repeat,
                        "kv_dtype": dtype,
                        "workload": workload["name"],
                        "jit_during_measurement": jit_detected,
                        "jit_clean_retries": jit_retries,
                        "runner_sha256": sha256_file(Path(__file__)),
                    }
                    result_path.write_text(json.dumps(payload, indent=2) + "\n")
            finally:
                server.terminate()
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()


if __name__ == "__main__":
    main()
