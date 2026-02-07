# LLM Tasks

Hydra 기반 LLM 텍스트 분석 파이프라인.

## Tasks

| Task | 설명 | Config Key |
|------|------|------------|
| emotion | 7가지 감정 분류 (JSON 출력) | `task_type: emotion` |
| summary | 마크다운 형식 텍스트 요약 | `task_type: summary` |
| chat | 대화형 챗봇 | `task_type: chat` |

## Usage

```bash
conda env create -f environment.yml
conda activate llm_lab_tasks
python main.py                          # default: emotion
python main.py task_type=summary        # summary mode
python main.py task_type=chat           # chat mode
python main.py model=llama              # switch model
```

## Structure

```
llm_tasks/
├── main.py                 # Entry point (Hydra)
├── configs/
│   ├── config.yaml         # Main config
│   └── model/*.yaml        # Model configs (bllossom, llama)
├── src/
│   ├── pipeline.py         # LLMPipeline (task + model orchestration)
│   ├── prep.py             # Text preprocessing (Korean/English)
│   └── tasks/
│       ├── factory.py      # TaskFactory (registry pattern)
│       ├── base_task.py    # BaseTask ABC
│       ├── emotion_task.py
│       ├── summary_task.py
│       └── chat_task.py
└── environment.yml
```

## Models

- **Bllossom** (default): `MLP-KTLim/llama-3-Korean-Bllossom-8B` (8-bit quantization)
- **LLaMA**: Configurable via `configs/model/llama.yaml`
