"""Build the beginner-facing, result-only TurboQuant final notebook."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/12_turboquant_final_report.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md("""
# TurboQuant 공식 구현 PoC: 최종 읽기·실행 보고서

이 노트북은 **결론 → 개념 → 실험 조건 → 결과 → 의사결정** 순서의 단일 진입점이다.
GPU 추론은 `gpu03`에서 이미 완료했으며, 여기서는 저장된 원시 결과를 다시 읽어
표와 그림을 재현한다. 따라서 초심자도 GPU 없이 전체 판단 흐름을 실행할 수 있다.

## 한 줄 결론

TurboQuant는 같은 36.28 GiB KV 예산에서 capacity를 크게 늘렸지만, 이번 짧은 요청
온라인 부하에서는 BF16/FP8보다 느렸다. 실제 HotpotQA 품질 비교는 0.6B BF16
baseline 자체가 너무 낮아 비열등성 결론을 낼 수 없다.
"""),
    md("""
## 1. 무엇을 양자화했나?

Transformer는 새 token을 만들 때 과거 token의 Key/Value를 KV cache에 저장한다.
KV bit-width를 낮추면 같은 GPU 메모리에 더 많은 token 또는 동시 요청을 담을 수 있다.

- `auto`: BF16 KV, 정확도 기준선
- `fp8_e4m3`: vLLM 일반 FP8 KV 기준선
- `turboquant_k8v4`: Key 8-bit + Value 4-bit
- `turboquant_4bit_nc`: Key/Value 4-bit, no correction

TurboQuant의 핵심은 Key에 Hadamard 회전과 Lloyd–Max 양자화를 적용해 outlier 영향을
완화하는 것이다. Value에는 uniform quantization을 사용한다. vLLM 0.26.0 공식
kernel을 사용했으며, 교육용 NumPy 모사는 최종 성능 증거로 사용하지 않았다.
"""),
    code("""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd()
if not (ROOT / "results").exists():
    ROOT = ROOT.parent

capacity = pd.read_csv(ROOT / "results/vllm_turboquant_smoke_summary.csv")
quality = pd.read_csv(ROOT / "results/vllm_turboquant_quality_summary.csv")
online = pd.read_csv(ROOT / "results/vllm_turboquant_online_summary.csv")
labels = {
    "auto": "BF16",
    "fp8_e4m3": "FP8",
    "turboquant_k8v4": "TQ K8V4",
    "turboquant_4bit_nc": "TQ 4bit",
}
for frame in (capacity, quality, online):
    frame["method"] = frame["kv_dtype"].map(labels)
"""),
    md("""
## 2. 실험 조건과 판정 원칙

### Capacity / smoke

- 모델: `Qwen/Qwen3-0.6B`
- GPU: RTX PRO 6000 Blackwell 1장
- vLLM: 0.26.0
- KV 예산: 모든 방법 36.28 GiB
- 최대 길이: 8,192 token

### 실제 품질

- 데이터: LongBench가 가공한 실제 Wikipedia 기반 HotpotQA
- 고정 표본: seed 20260728, 50개
- 지표: normalized token-F1
- 비교: 동일 문항의 paired bootstrap 10,000회
- 비열등성 한계: BF16 대비 −0.02
- 필수 floor: BF16 평균 F1 ≥ 0.10

마지막 floor는 매우 중요하다. 못 푸는 모델끼리 비슷하면 “품질 유지”가 아니기 때문이다.

### Online serving

- 공식 `vllm bench serve`
- 각 방법 1,000 요청, 실패 허용 0
- 입력 256 / 출력 64 token
- 동시성 제한 64, 전 요청 즉시 투입
- 지표: req/s, TTFT, TPOT, end-to-end P99

이는 포화 microbenchmark다. 실제 서비스 도착률 분포나 긴 prompt workload를
대표하지 않으므로, 제품 결론 전에는 trace replay가 추가로 필요하다.
"""),
    code("""
capacity[[
    "method", "kv_tokens", "max_length_concurrency",
    "latency_mean_s", "latency_p99_s", "needle_pass"
]].sort_values("kv_tokens")
"""),
    md("""
## 3. Capacity 결과 읽기

`kv_tokens`는 동일한 메모리에 실제로 할당된 KV token 수다. BF16 대비 FP8 약 2배,
K8V4 약 2.12배, 4bit 약 2.72배다. 즉 **메모리 capacity 이득은 공식 kernel에서
재현**됐다. 다만 단일 4K prompt offline latency는 K8V4와 4bit가 더 느렸다.
"""),
    code("""
order = ["BF16", "FP8", "TQ K8V4", "TQ 4bit"]
cap = capacity.set_index("method").loc[order]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
cap["kv_tokens"].div(1e3).plot.bar(ax=axes[0], color="#4C78A8")
axes[0].set(title="동일 36.28 GiB에서 KV capacity", ylabel="천 token", xlabel="")
cap["latency_p99_s"].plot.bar(ax=axes[1], color="#F58518")
axes[1].set(title="Offline batch=1 P99", ylabel="초", xlabel="")
plt.tight_layout()
plt.savefig(ROOT / "figures/vllm_turboquant_capacity_latency_final.png", dpi=160)
plt.show()
"""),
    code("""
quality[[
    "method", "n", "mean_f1", "delta_vs_bf16", "ci95_low", "ci95_high",
    "baseline_floor_pass", "non_inferior"
]].sort_values("mean_f1", ascending=False)
"""),
    md("""
## 4. 실제 품질 결과 읽기

K8V4의 평균 delta는 양수이고 4bit의 평균 delta는 −0.0086이지만, 표본 50개의
신뢰구간은 넓다. 더 근본적으로 BF16 평균 F1이 0.061로 사전등록 floor 0.10을
실패했다. 그러므로 `non_inferior=False`가 올바른 판정이다.

이 결과는 “TurboQuant 품질이 나쁘다”도 “유지된다”도 증명하지 않는다.
다음 품질 실험은 HotpotQA를 유의미하게 푸는 더 큰 모델에서 반복해야 한다.
"""),
    code("""
q = quality.set_index("method").loc[order]
y = q["delta_vs_bf16"]
lower = y - q["ci95_low"]
upper = q["ci95_high"] - y
plt.figure(figsize=(8, 4))
plt.errorbar(order, y, yerr=[lower, upper], fmt="o", capsize=5)
plt.axhline(-0.02, color="red", linestyle="--", label="NI margin −0.02")
plt.axhline(0, color="black", linewidth=0.8)
plt.ylabel("BF16 대비 paired F1 차이")
plt.title("LongBench HotpotQA: 평균 차이와 bootstrap 95% CI")
plt.legend()
plt.tight_layout()
plt.savefig(ROOT / "figures/vllm_turboquant_quality_ci.png", dpi=160)
plt.show()
"""),
    code("""
online[[
    "method", "completed", "failed", "request_throughput",
    "p99_ttft_ms", "p99_tpot_ms", "p99_e2el_ms"
]].sort_values("request_throughput", ascending=False)
"""),
    md("""
## 5. Online serving 결과 읽기

모든 방법이 1,000/1,000 요청을 성공했다. 이 짧은 workload에서는 FP8이 처리량과
P99 E2E 모두 가장 좋았다. K8V4는 BF16보다 처리량이 약 18% 낮고, 4bit는 약 33%
낮았다. 압축·복원 kernel 비용이 capacity 이득보다 크게 나타난 조건이다.

반대로 context가 길거나 KV capacity 때문에 BF16이 동시 요청을 수용하지 못하는
구간에서는 순위가 바뀔 수 있다. 따라서 이번 결과는 “TurboQuant가 항상 느리다”가
아니라 **현재 workload에서 crossover가 아직 발생하지 않았다**는 뜻이다.
"""),
    code("""
srv = online.set_index("method").loc[order]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
srv["request_throughput"].plot.bar(ax=axes[0], color="#54A24B")
axes[0].set(title="1,000-request 처리량", ylabel="req/s", xlabel="")
srv["p99_e2el_ms"].plot.bar(ax=axes[1], color="#E45756")
axes[1].set(title="End-to-end P99", ylabel="ms", xlabel="")
plt.tight_layout()
plt.savefig(ROOT / "figures/vllm_turboquant_online_final.png", dpi=160)
plt.show()
"""),
    md("""
## 6. 최종 의사결정

| 질문 | 현재 답 | 근거 |
|---|---|---|
| 공식 구현이 동작하는가? | 예 | vLLM 0.26.0 네 preset 실행 성공 |
| KV capacity가 늘어나는가? | 예 | K8V4 2.12×, 4bit 2.72× |
| 짧은 요청 serving이 빨라지는가? | 아니오 | FP8 74.1 > BF16 71.1 > K8V4 58.5 > 4bit 47.6 req/s |
| 실제 QA 품질이 유지되는가? | 아직 판단 불가 | BF16 baseline floor 실패 |
| 지금 제품 default로 채택할가? | 아니오 | workload crossover와 큰 모델 품질 gate 미충족 |

### 다음 우선순위

1. 7B 이상 모델로 LongBench ≥200문항 재실행
2. 실제 서비스 trace의 길이·동시성 분포로 rate sweep
3. FP8이 OOM/queueing에 들어가는 지점에서 TurboQuant crossover 측정
4. 그 뒤에만 품질–capacity–P99 Pareto frontier로 preset 결정

재현 명령과 상세 이론은 `README.md` 및
`QuantumAI/_Agent/260728_TurboQuant_이론과_PoC_가이드.md`가 정본이다.
"""),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        }
    },
)
OUTPUT.write_text(nbf.writes(notebook))
