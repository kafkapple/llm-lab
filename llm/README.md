# LLM Projects

LLM(Large Language Model) 관련 프로젝트 모음.

## 프로젝트

### llm_tasks/
Hydra 기반 LLM 파이프라인. 감정분석, 요약, 챗봇 태스크 지원.
- Factory Pattern으로 태스크 관리
- Bllossom/LLaMA 모델 + 8-bit 양자화
- `python main.py task_type=emotion`

### rag/
RAG(Retrieval-Augmented Generation) 파이프라인 + Streamlit UI.
- PDF/웹 문서 로드 → 분할 → 임베딩 → FAISS 벡터 저장
- BM25/FAISS/Ensemble Retriever 지원
- `streamlit run app.py`

### llava_emotion/
LLaVA-NeXT-Video 모델을 활용한 이미지/비디오 감정 분석.
- 자유형 감정 설명 + 라벨 분류 2가지 모드
- 4-bit NF4 양자화로 메모리 최적화
- `python main.py`

### multi_agent/
다중 AI 모델(OpenAI, Perplexity, Gemini, Ollama) 협업 시스템.
- ReACT / Tool-calling 에이전트
- `python main.py`

### prompt_eng/
프롬프트 엔지니어링 실험 (prompt_eng_1, prompt_eng_2).

### _archive/rag_study/
초기 RAG 연구 코드 (아카이브).

## 공통 유틸리티

`shared/` 디렉토리에 프로젝트 간 공통 패턴이 추출되어 있습니다:
- `base_factory.py` - Registry 기반 Factory ABC
- `config_utils.py` - Hydra/OmegaConf 헬퍼
- `model_utils.py` - 양자화 모델 로딩 (4-bit/8-bit)
