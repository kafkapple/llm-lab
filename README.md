# LLM Lab

LLM/ML 프로젝트 통합 모노레포. 추후 `llm-lab` (LLM) + `ml-lab` (ML) 분리 예정.

## 구조

```
llm-lab/
├── shared/                         # LLM 공통 유틸 (Factory, Config, Model)
├── llm/                            # LLM 프로젝트 ← 미래 llm-lab
│   ├── llm_tasks/                  # Hydra LLM pipeline (감정분석, 요약, 챗봇)
│   ├── rag/                        # RAG + Streamlit UI
│   ├── llava_emotion/              # LLaVA 이미지/비디오 감정 분석
│   ├── multi_agent/                # 다중 AI 모델 협업
│   ├── prompt_eng/                 # 프롬프트 엔지니어링
│   └── _archive/rag_study/         # (아카이브) 초기 RAG 연구
└── ml/                             # ML 프로젝트 ← 미래 ml-lab
    ├── pytorch_integrated/         # PyTorch 통합 프레임워크
    ├── stt/                        # STT/TTS 실험
    ├── sam/                        # SAM2 + ByteTrack
    └── behavior_ssm/              # 동물 행동 SSM
```

## LLM 프로젝트

| 프로젝트 | 설명 | 실행 |
|----------|------|------|
| `llm/llm_tasks/` | 감정분석, 요약, 챗봇 | `python main.py` |
| `llm/rag/` | RAG 질의응답 데모 | `streamlit run app.py` |
| `llm/llava_emotion/` | 멀티모달 감정 분석 | `python main.py` |
| `llm/multi_agent/` | 다중 AI 모델 협업 | `python main.py` |
| `llm/prompt_eng/` | 프롬프트 엔지니어링 | - |

## ML 프로젝트

| 프로젝트 | 설명 |
|----------|------|
| `ml/pytorch_integrated/` | VAE, VQ-VAE, CLIP 모델 실험 |
| `ml/stt/` | 음성-텍스트 변환 |
| `ml/sam/` | SAM2 세그멘테이션 + 추적 |
| `ml/behavior_ssm/` | 동물 행동 State Space Model |

## 환경 설정

```bash
# 프로젝트별 독립 conda 환경
cd llm/llm_tasks && conda env create -f environment.yml   # llm_lab_tasks
cd llm/rag && conda env create -f environment.yml          # llm_lab_rag
cd llm/llava_emotion && conda env create -f environment.yml # llm_lab_llava

# API keys
cp .env.example .env  # 키 입력
```
