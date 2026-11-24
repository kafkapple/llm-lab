# TurboQuant KV cache (학습용 단일 노트북)

vLLM 0.26.0 TurboQuant KV cache 양자화를 Qwen2.5-7B 에서 평가한 PoC 의 학습용 사본.

- `05_turboquant_7b_report.ipynb`: 실행 출력 포함. 저장된 요약 CSV 로 표·그림을 GPU 없이 재현한다. `RUN_GPU=1` 은 원본 repo 의 스크립트가 필요하다.
- `turboquant_7b_confirmatory_report.html`: 개념부터 결과·한계까지 한 문서.
- 결과 한 줄: K8V4 만 HotpotQA 200문항 F1 비열등성(−0.02)을 통과했고, 처리량은 BF16 보다 낮았다.
- 원본(실행 스크립트·전체 results): private repo `quantumai-tech-poc`.
