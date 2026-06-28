# 🧪 LLM & ML Lab (통합 모노레포)

본 저장소는 대규모 언어 모델(LLM) 및 머신러닝(ML) 관련 독립 프로젝트들을 효율적으로 개발하고 공유 코드를 일관성 있게 관리하기 위해 구성된 **통합 모노레포(Monorepo)**입니다.

> [!NOTE]  
> 향후 프로젝트 성숙도에 따라 LLM 전용 저장소(`llm-lab`)와 ML 전용 저장소(`ml-lab`)로 분리될 예정입니다.

---

## 📁 1. 전체 디렉토리 구조 및 역할

본 모노레포는 공통 유틸리티(`shared/`), 언어 모델 프로젝트(`llm/`), 그리고 머신러닝 실험 공간(`ml/`)의 3가지 핵심 영역으로 분리되어 있습니다.

```text
llm-lab/
├── shared/                         # 📦 LLM/ML 프로젝트 공통 유틸리티
│   ├── base_factory.py             # - Registry 기반 Factory 추상 클래스
│   ├── config_utils.py             # - Hydra 및 OmegaConf 설정 헬퍼
│   └── model_utils.py              # - 4-bit/8-bit 양자화 모델 로더
├── llm/                            # 💬 LLM 프로젝트 영역 (향후 llm-lab 분리 예정)
│   ├── llm_tasks/                  # - 감정 분석, 요약, 챗봇 파이프라인 (Hydra 기반)
│   ├── rag/                        # - RAG 기반 질의응답 (Streamlit UI 데모 포함)
│   ├── isp_rag_cluster/            # - RAG 기반 감정 분류/클러스터링 실험 (Hydra) [추가]
│   ├── nlp_task_sum/               # - 한국어 대화 요약 및 감정 분류 (BART/BERT/LLaMA) [추가]
│   ├── llava_emotion/              # - LLaVA-NeXT-Video 활용 멀티모달 감정 분석
│   ├── multi_agent/                # - 다중 모델(OpenAI/Gemini 등) 협업 에이전트
│   ├── prompt_eng/                 # - 프롬프트 엔지니어링 실험실
│   └── _archive/                   # - (아카이브) 초기 RAG 연구 및 기타 프로토타입
└── ml/                             # 🤖 ML 프로젝트 영역 (향후 ml-lab 분리 예정)
    ├── pytorch_integrated/         # - PyTorch 통합 실험 프레임워크 (VAE, VQ-VAE, CLIP 등)
    ├── stt/                        # - STT(Whisper) 및 TTS 실험 공간
    ├── sam/                        # - SAM2 + ByteTrack 객체 탐지 및 추적 UI
    └── behavior_ssm/               # - 동물 행동 분석 State Space Model 실험
```

---

## 🚀 2. 프로젝트 일람

### 💬 LLM 프로젝트 목록

| 하위 디렉토리 | 프로젝트 목적 및 핵심 기술 | 실행 진입점 (Entry Point) |
| :--- | :--- | :--- |
| **`llm/llm_tasks`** | 감정 분석, 요약, 챗봇 등 다양한 태스크를 Factory 패턴과 8-bit 양자화 모델(Bllossom/LLaMA)로 테스트 | `python main.py task_type=emotion` |
| **`llm/rag`** | PDF/웹 문서를 FAISS/BM25 앙상블 검색기를 통해 검색 후 답변을 제공하는 Streamlit 데모 | `streamlit run app.py` |
| **`llm/isp_rag_cluster`** | RAG 기술과 임베딩 클러스터링을 결합하여 정교한 감정 라벨 분류를 진행하는 실험 파이프라인 | `python main_llm.py` |
| **`llm/nlp_task_sum`** | 한국어 대화 데이터 요약 및 성능 지표 측정을 위해 KoBART, Bllossom, Solar 등을 훈련/평가 | `python main.py` |
| **`llm/llava_emotion`** | 멀티모달 LLM(LLaVA)을 활용하여 이미지/비디오 내 상황별 인물 감정 분석 | `python main.py` |
| **`llm/multi_agent`** | ReACT 프레임워크와 API 도구들을 연동해 상호작용하는 다중 에이전트 시스템 구현 | `python main.py` |

<br>

### 🤖 ML 프로젝트 목록

| 하위 디렉토리 | 프로젝트 목적 및 핵심 기술 |
| :--- | :--- |
| **`ml/pytorch_integrated`** | MNIST VAE, CIFAR-10 VQ-VAE, Flickr CLIP 등 PyTorch 기반 학습 템플릿 실증 |
| **`ml/sam`** | Segment Anything 2(SAM2) 모델과 ByteTrack을 연동하여 비디오 내 객체 추적을 지원하는 웹 UI 대시보드 |
| **`ml/stt`** | Whisper 모델 기반 다국어 STT 추출 및 TTS 오디오 파일 음색 합성 실험 |
| **`ml/behavior_ssm`** | 동물의 상태 전이 패턴을 파악하기 위한 State Space Model(SSM) 적용 연구 |

---

## ⚙️ 3. 환경 설정 및 설치 가이드

각 프로젝트는 서로 다른 패키지 버전(예: PyTorch, Transformers 등)을 요구하므로, **독립된 Conda 또는 Virtualenv 환경** 사용을 강력히 권장합니다.

### ① 아나콘다(Conda) 가상환경 생성

원하는 하위 프로젝트로 이동하여 `environment.yml`을 로드하여 가상환경을 생성합니다.

```bash
# 1. LLM 기본 태스크 테스트 환경
cd llm/llm_tasks
conda env create -f environment.yml
conda activate llm_lab_tasks

# 2. RAG 데모 실행 환경
cd ../rag
conda env create -f environment.yml
conda activate llm_lab_rag

# 3. 한국어 대화 요약/감정 분류 환경
cd ../nlp_task_sum
conda env create -f environment.yml
conda activate dialogue_summary_env
```

### ② API 키 및 환경 변수 설정

다양한 LLM API 및 로깅 툴(WandB)을 사용하기 위해 루트 디렉토리에 `.env` 파일을 구성합니다.

```bash
# .env.example 복사
cp .env.example .env
```

복사한 `.env` 파일에 각 플랫폼의 API 토큰을 채워 넣습니다:
```env
# API Keys for LLM APIs
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=xkeys-...
UPSTAGE_API_KEY=upstage-...
GEMINI_API_KEY=AIzaSy...

# Optional Hugging Face & Logger settings
HUGGINGFACE_TOKEN=hf_...
WANDB_API_KEY=your_wandb_api_key
```

---

## 🛠️ 4. 공통 유틸리티 활용 방법 (`shared/`)

하위 프로젝트 내 코드 중복을 최소화하기 위해 공통 모듈인 `shared/`를 적극적으로 임포트하여 사용합니다.

* **Registry 기반 구조 구현 (`base_factory.py`)**:
  신규 모델이나 모듈 정의 시 `BaseFactory`를 확장하여 손쉽게 동적 등록 프로세스를 적용할 수 있습니다.
* **Hydra 설정 다형성 활용 (`config_utils.py`)**:
  복잡하게 얽힌 다중 프로젝트 구조 내에서 Hydra 및 OmegaConf 파싱을 간편하게 제어합니다.
* **양자화 모델 일관 로드 (`model_utils.py`)**:
  로컬 환경의 VRAM 제약을 해결하기 위한 4-bit NF4/8-bit 양자화 모델 로딩 로직이 통일되어 있어 메모리를 절약할 수 있습니다.

---

> [!TIP]  
> 신규 프로젝트를 모노레포에 추가할 경우, 소스 내에 `.git` 디렉토리가 생성되거나 무거운 데이터셋 파일이 직접 추적되지 않도록 항상 `.gitignore` 정책을 준수해야 합니다.
