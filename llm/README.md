# 💬 LLM Projects Workspace

이 디렉토리는 대규모 언어 모델(LLM)을 활용한 다양한 자연어 처리(NLP), 프롬프트 엔지니어링, 검색 증강 생성(RAG), 그리고 멀티 에이전트 협업 시스템 실험 환경을 포함합니다.

---

## 📁 1. 하위 프로젝트 상세 소개 및 실행 방법

### ① **`llm_tasks/`** (Hydra 기반 표준 LLM 파이프라인)
* **설명**: Hydra 설정을 활용하여 감정 분석, 대화 요약, 자유 챗봇 등의 다양한 자연어 처리 태스크를 통일된 구조로 테스트하는 파이프라인입니다.
* **핵심 기능**: 
  - Factory 패턴을 활용한 태스크 분기 관리
  - Hugging Face의 Bllossom, LLaMA 계열 모델을 8-bit 양자화 기법으로 서빙
* **실행 예시**:
  ```bash
  cd llm/llm_tasks
  python main.py task_type=emotion  # 감정 분석 수행
  python main.py task_type=summary  # 대화 요약 수행
  ```

### ② **`rag/`** (실시간 RAG 데모 및 웹 인터페이스)
* **설명**: PDF 문서나 웹 링크의 텍스트를 파싱하여 임베딩한 후, 로컬 또는 클라우드 LLM을 연동하여 지식 기반 질의응답을 수행하는 Streamlit 웹 애플리케이션입니다.
* **핵심 기능**:
  - BM25와 FAISS 기반의 하이브리드 검색 앙상블(Ensemble Retriever)
  - Streamlit을 이용한 실시간 채팅 대화 UI
* **실행 예시**:
  ```bash
  cd llm/rag
  streamlit run app.py
  ```

### ③ **`isp_rag_cluster/`** (RAG & 임베딩 클러스터링 감정 분석) 🌟
* **설명**: 단순한 프롬프팅을 넘어, RAG 지식베이스 검색 결과와 다차원 임베딩 클러스터링을 융합하여 입력 데이터의 정교한 감정 라벨을 예측하는 연구형 파이프라인입니다.
* **핵심 기능**:
  - Ollama(로컬), OpenAI, Upstage, Anthropic 등 다중 백엔드 지원
  - 유사 예제를 실시간으로 벡터 DB에서 검색하여 Few-shot 프롬프트 구성
  - Hydra 설정을 통한 다양한 임계치(Threshold) 및 템플릿 제어
* **실행 예시**:
  ```bash
  cd llm/isp_rag_cluster
  python main_llm.py model.type=ollama model.name=llama3  # 로컬 모델 실행
  python main_llm.py model.type=openai model.name=gpt-4    # OpenAI 모델 실행
  ```

### ④ **`nlp_task_sum/`** (한국어 대화 요약 및 복합 감정 분류) 🌟
* **설명**: 한국어로 이뤄진 대화 데이터를 정제하고 요약문 및 감정을 추출하는 통합 지도학습/추론 프로젝트입니다.
* **핵심 기능**:
  - `gogamza/kobart-summarization` (Seq2Seq 요약) 및 `klue/roberta-small` (인코더 기반 분류)
  - `llama-3-Korean-Bllossom-8B` 지도학습(SFT) 및 Hugging Face Trainer 연동
  - Upstage Solar API 활용 요약 비교 분석
* **실행 예시**:
  ```bash
  cd llm/nlp_task_sum
  # 기본 추론 및 훈련 실행
  python main.py
  # 특정 모델 및 훈련 속성 오버라이드
  python main.py model=kobart train=finetune
  ```

### ⑤ **`llava_emotion/`** (멀티모달 이미지/비디오 감정 분석)
* **설명**: LLaVA-NeXT-Video 모델을 활용해 시각 데이터를 바탕으로 감정 상태와 행동 양식을 서술하고 분류하는 멀티모달 실험 공간입니다.
* **핵심 기능**:
  - 4-bit NF4 양자화 적용으로 VRAM 효율화
  - 이미지/비디오를 인식하여 상황 묘사와 감정 라벨 동시 추출
* **실행 예시**:
  ```bash
  cd llm/llava_emotion
  python main.py
  ```

### ⑥ **`multi_agent/`** (다중 모델 협업 에이전트)
* **설명**: OpenAI, Gemini, Perplexity, Ollama 등의 서로 다른 이종 모델들을 에이전트로 정의하고, ReACT 프레임워크를 기반으로 상호 질문하며 복잡한 문제를 해결하는 분산 에이전트 시스템입니다.
* **실행 예시**:
  ```bash
  cd llm/multi_agent
  python main.py
  ```

---

## 🛠️ 2. 공유 아키텍처 및 공통 모듈 (`shared/`)

`llm/` 내부의 모든 프로젝트들은 루트 레벨의 `shared/` 유틸리티를 참조하도록 유도하여 일관된 코드 스타일과 유지보수성을 극대화합니다.

1. **`base_factory.py` (동적 팩토리)**: Registry 디자인 패턴을 활용하여 개별 프로젝트가 수동 `if-else` 분기 처리 없이 설정 파일 변경만으로 모델 및 태스크 객체를 동적으로 인스턴스화할 수 있도록 지원합니다.
2. **`config_utils.py` (설정 정규화)**: Hydra 설정 로드 시 발생할 수 있는 누락값 검증 및 구조 분해 기능을 제공합니다.
3. **`model_utils.py` (메모리 최적화 로더)**: 4-bit/8-bit 양자화 옵션(`bitsandbytes`)을 자동으로 활성화하여 로컬 GPU 환경에서도 거대 모델이 VRAM OOM 없이 구동될 수 있도록 돕습니다.
