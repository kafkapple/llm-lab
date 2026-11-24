# TurboQuant KV cache PoC

vLLM 0.26.0 TurboQuant KV cache 양자화를 Qwen2.5-7B 에서 평가한 PoC. 학습용 단일 노트북 중심으로 통합했다.

## 결과 한 줄

K8V4 만 HotpotQA 200문항 F1 비열등성(−0.02)을 통과했고, 처리량은 BF16 보다 낮았다.

## 진입점

- 학습용 노트북: `notebooks/05_turboquant_7b_report.ipynb` (실행 출력 포함, GPU 없이 `results/*.csv` 요약으로 재현)
- 공유용 보고서: `reports/turboquant_7b_confirmatory_report.html`
- 실행 계약: `benchmarks.toml`
- 실행 코드: `src/qai_poc`, `scripts/`, `tests/`

## 실행

```bash
conda env create -f environment.yml
conda activate quantumai-poc
jupyter lab
```

- GPU 단계는 `RUN_GPU=1` 일 때만 실행하며 vLLM 환경(`environment-vllm.yml`)이 필요하다.
- 원시 GPU 로그와 전체 results 는 이 repo 에 없다. private 원본 `quantumai-tech-poc` 에 있다.
- 데이터(LongBench HotpotQA)는 repo 에 두지 않는다. 공식 LongBench 저장소에서 받아 `data/longbench/` 에 둔다.
