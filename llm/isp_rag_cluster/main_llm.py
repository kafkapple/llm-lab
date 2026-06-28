from dotenv import load_dotenv, find_dotenv
import os
import pandas as pd
import json
from ollama import chat, ChatResponse
from anthropic import Anthropic
load_dotenv()
from pathlib import Path
import hydra
from tqdm import tqdm
from openai import OpenAI
from langchain_ollama import OllamaLLM
from typing import Dict, Any, Optional, List, Union
import time
from src.logger import setup_logging
from src.rag import EmotionRAG
from src.eval import save_metrics, save_prediction_stats
from src.data_prep import load_data, clean_json_response
from src.util import sanitize_model_name, get_output_dir_name
from src.cot import EmotionCoT
from src.util import send_telegram_message
import asyncio
import logging
import traceback
import numpy as np
from src.eval import calculate_efficiency_metrics, count_tokens

# Add global flag for prompt logging
_PROMPT_LOGGED = False
_INPUT_LOGGED = False
log_separator = "="*80  # 전역 변수로 정의

# 전역 logger 설정
logger = logging.getLogger(__name__)

def initialize_models(cfg):
    """Initialize LLM and RAG model"""
    client = None
    
    if cfg.model.type == "ollama":
        client = OllamaLLM(
            model=cfg.model.name,
            temperature=cfg.model.temperature
        )
    elif cfg.model.type == "openai":
        base_url = "https://api.openai.com/v1"
        api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
    elif cfg.model.type == "upstage":
        base_url = "https://api.upstage.ai/v1/solar"
        api_key = os.getenv("UPSTAGE_API_KEY")
        client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
    elif cfg.model.type == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        client = Anthropic(api_key=api_key)
    
    return client


def process_response(response, cfg, retry_count=0):
    """응답 처리"""
    try:
        # 특수 문자 처리
        def clean_text(text):
            if not isinstance(text, str):
                return str(text)
            # ASCII 범위를 벗어나는 문자를 제거하고 공백으로 대체
            return ''.join(char if ord(char) < 128 else ' ' for char in text)
        
        if cfg.model.function_calling and cfg.model.name in cfg.model.function_call_models:
            # Ollama 모델의 경우
            if cfg.model.name == "gemma3":
                try:
                    # clean_json_response 모듈 사용
                    result = clean_json_response(response, logger, cfg)
                    if result and all(key in result for key in ['emotion', 'confidence_score', 'explanation']):
                        if result['confidence_score'] is None:
                            result['confidence_score'] = 0.5  # 기본값 설정
                        return result
                    else:
                        logger.warning(f"필수 필드 누락 또는 잘못된 값")
                        return {
                            "emotion": cfg.data.default_emotion,
                            "confidence_score": 0.0,
                            "explanation": "Missing required fields"
                        }
                except Exception as e:
                    logger.warning(f"JSON 파싱 오류: {str(e)}\n응답: {clean_text(response)}")
                    return {
                        "emotion": cfg.data.default_emotion,
                        "confidence_score": 0.0,
                        "explanation": "JSON parsing error"
                    }
            else:
                # 다른 모델들의 경우
                try:
                    if isinstance(response, str):
                        result = clean_json_response(response, logger, cfg)
                    else:
                        result = response
                    if result and 'confidence_score' in result and result['confidence_score'] is None:
                        result['confidence_score'] = 0.5  # 기본값 설정
                    return result
                except Exception as e:
                    logger.warning(f"JSON 파싱 오류: {clean_text(response)}")
                    return {
                        "emotion": cfg.data.default_emotion,
                        "confidence_score": 0.0,
                        "explanation": "JSON parsing error"
                    }
        else:
            # 일반 응답 처리
            try:
                if isinstance(response, str):
                    result = clean_json_response(response, logger, cfg)
                else:
                    result = response
                if result and 'confidence_score' in result and result['confidence_score'] is None:
                    result['confidence_score'] = 0.5  # 기본값 설정
                return result
            except Exception as e:
                logger.warning(f"JSON 파싱 오류: {clean_text(response)}")
                return {
                    "emotion": cfg.data.default_emotion,
                    "confidence_score": 0.0,
                    "explanation": "JSON parsing error"
                }
    except Exception as e:
        logger.warning(f"응답 처리 오류: {str(e)}")
        return {
            "emotion": cfg.data.default_emotion,
            "confidence_score": 0.0,
            "explanation": "Response processing error"
        }

def get_function_call(cfg, model_name, client, text, logger):
    labels = cfg.data.datasets[cfg.data.name].labels
    # ListConfig를 일반 Python 리스트로 변환
    labels = list(labels)
    
    logger.info(f">>>>>> Using function calling for {model_name}")
    system_prompt = f"You MUST classify the emotion as one of these ONLY: {', '.join(labels)}. Do not use any other emotion words."
    function_description = f"Strictly classify the emotion as one of these ONLY: {', '.join(labels)}"
    emotion_description = f"Must be one of: {', '.join(labels)}"
    tools = [{
        "type": "function",
        "function": {
            "name": "search",
            "description": function_description,
            "parameters": {
                "type": "object",
                "properties": {
                    "emotion": {
                        "type": "string",
                        "enum": labels,
                        "description": emotion_description
                    },
                    "confidence_score": {
                        "type": "number",
                        "description": "confidence score of the emotion: 0.0-1.0"
                    },
                    "explanation": {
                        "type": "string",
                        "description": "explain why the emotion is decided"
                    },
                },
                "required": ["emotion", "confidence_score", "explanation"]
            }
        }
    }]
    
    try:
        if "gpt" in model_name.lower():
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "search"}},
                temperature=cfg.model.temperature,
                timeout=cfg.model.timeout
            )
            
            if response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                return process_response(json.loads(tool_call.function.arguments), cfg)
        
        elif "claude" in model_name.lower():
            response = client.messages.create(
                model=model_name,
                max_tokens=cfg.model.max_tokens,
                temperature=cfg.modeltemperature,
                system=system_prompt,
                messages=[{"role": "user", "content": text}],
                tools=tools
            )
            
            if response.content and hasattr(response.content[0], 'tool_calls'):
                tool_call = response.content[0].tool_calls[0]
                return process_response(json.loads(tool_call.function.arguments), cfg)
        
        else:  # ollama 모델
            prompt = f"""System: {system_prompt}

User: {text}

Please classify the emotion using the following function:
{json.dumps(tools[0], indent=2)}

Respond with a JSON object containing the function call arguments."""

            response = client.invoke(prompt)
            
            try:
                # 응답에서 JSON 부분 추출
                response_text = response.content if hasattr(response, 'content') else str(response)
                
                # ```json 마커가 있는 경우 제거
                if '```json' in response_text:
                    json_str = response_text.split('```json')[1].split('```')[0].strip()
                else:
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = response_text[json_start:json_end]
                    else:
                        raise ValueError("No valid JSON found in response")
                
                # JSON 문자열에서 불필요한 문자 제거
                json_str = json_str.replace('\n', '').replace('\r', '')
                
                # JSON 파싱
                function_args = json.loads(json_str)
                return process_response(function_args, cfg)
                
            except Exception as e:
                logger.error(f"Error parsing Ollama response: {str(e)}")
                logger.error(f"Raw response: {response_text}")
                raise
            
    except Exception as e:
        logger.error(f"Error in function calling for {model_name}: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise

def get_model_response(
    client: Any,
    text: str,
    model_name: str,
    prompt_name: str,
    fallback_prompts: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None,
    cfg: Optional[Dict[str, Any]] = None,
    rag: Optional[Any] = None,
    model: Optional[str] = None,
    final_prompt: Optional[str] = None,
    ensemble_data: Optional[Dict[str, pd.DataFrame]] = None
) -> Dict[str, Any]:
    """모델 응답을 가져오는 함수"""
    max_retries = cfg.general.retry.max_retries
    retry_delay = cfg.general.retry.retry_delay
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # 재시도 횟수 초기화
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Ensemble이 활성화된 경우
            # if cfg.general.ensemble.enabled and ensemble_data and prompt_name == "ensemble_prompt":
            #     final_prompt = get_ensemble_prompt(text, ensemble_data, cfg)
            
            # CoT가 필요한 경우
            # if "react" in prompt_name.lower() or "cot" in prompt_name.lower():
            #     cot = EmotionCoT(cfg, client=client, logger=logger, rag=rag)
            #     result = cot.analyze(text)
            #     if result and result.get("status") == "success":
            #         result = process_response(result["result"], cfg)
            #         if result:
            #             result["retry_count"] = retry_count
            #         return result
            
            # RAG가 필요한 경우
            if "rag" in prompt_name.lower() and rag:
                # OpenAI와 Anthropic 모델의 경우 다른 API 호출 방식 사용
                if isinstance(client, (OpenAI, Anthropic)):
                    if isinstance(client, OpenAI):
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=[{"role": "user", "content": final_prompt}],
                            temperature=cfg.model.temperature,
                            max_tokens=cfg.model.max_tokens
                        )
                        response_text = response.choices[0].message.content
                    else:  # Anthropic
                        response = client.messages.create(
                            model=model_name,
                            max_tokens=cfg.model.max_tokens,
                            temperature=cfg.model.temperature,
                            messages=[{"role": "user", "content": final_prompt}]
                        )
                        response_text = response.content[0].text
                else:
                    response = client.invoke(final_prompt)
                    response_text = response.content if hasattr(response, 'content') else str(response)
                
                # Raw response 로깅 추가
                logger.info(f"""
================================================================================
[Raw Response - Attempt {retry_count + 1}]
================================================================================
{response_text}
================================================================================
""")
                
                result = process_response(response_text, cfg)
                if result:
                    result["retry_count"] = retry_count
                return result
            
            # Function calling이 지원되는 경우
            elif cfg.model.function_calling and model and any(model_name.lower() in model.lower() for model_name in cfg.model.function_call_models):
                result = get_function_call(cfg, model_name, client, text, logger)
                if result:
                    result["retry_count"] = retry_count
                return result
            
            # 일반 프롬프트 처리
            else:
                # OpenAI와 Anthropic 모델의 경우 다른 API 호출 방식 사용
                if isinstance(client, (OpenAI, Anthropic)):
                    if isinstance(client, OpenAI):
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=[{"role": "user", "content": final_prompt}],
                            temperature=cfg.model.temperature,
                            max_tokens=cfg.model.max_tokens
                        )
                        response_text = response.choices[0].message.content
                    else:  # Anthropic
                        response = client.messages.create(
                            model=model_name,
                            max_tokens=cfg.model.max_tokens,
                            temperature=cfg.model.temperature,
                            messages=[{"role": "user", "content": final_prompt}]
                        )
                        response_text = response.content[0].text
                else:
                    response = client.invoke(final_prompt)
                    response_text = response.content if hasattr(response, 'content') else str(response)
                
                # Raw response 로깅 추가
                logger.info(f"""
================================================================================
[Raw Response - Attempt {retry_count + 1}]
================================================================================
{response_text}
================================================================================
""")
                
                result = process_response(response_text, cfg)
                if result:
                    result["retry_count"] = retry_count
                return result
            
        except Exception as e:
            logger.warning(f"Attempt {retry_count + 1} failed: {str(e)}")
            retry_count += 1
            if retry_count >= max_retries:
                break
            time.sleep(retry_delay)
    
    # 모든 시도 실패 시 기본 응답 반환
    return {
        "emotion": cfg.data.default_emotion,
        "confidence_score": 0.0,
        "explanation": f"Failed after {max_retries} retries",
        "response_time": 0,
        "retry_count": retry_count
    }


def call_prompt(cfg, row, index, rag, ensemble_data):
    if rag:
        similar_examples = []  # None 대신 빈 리스트로 초기화
        if "rag" in cfg.model.template.lower():
            rag.exclude_index(index)
            retrieved_examples = rag.get_similar_examples(
                str(row.text),
                query_emotion=str(row.emotion),
                k=cfg.rag.k_examples,
                threshold=cfg.rag.threshold
            )
            
            # None이 아닌 경우에만 할당, 아니면 빈 리스트 유지
            if retrieved_examples is not None:
                similar_examples = retrieved_examples
                logger.debug(f"Using RAG prompt with {len(similar_examples)} examples (threshold: {cfg.rag.threshold})")
            else:
                logger.warning("No similar examples found, using empty list")
            
            # 빈 리스트여도 get_rag_prompt 호출 가능
            final_prompt = rag.get_rag_prompt(str(row.text), similar_examples)
        
        # Document 객체를 딕셔너리로 변환 (로깅용)
        serializable_examples = [
            {
                "text": example.page_content if hasattr(example, 'page_content') else str(example),
                "metadata": example.metadata if hasattr(example, 'metadata') else {},
                "similarity": example.metadata.get('similarity', 0) if hasattr(example, 'metadata') else 0
            }
            for example in similar_examples
        ]
    elif "ensemble" in cfg.model.template.lower():
        serializable_examples = []
        final_prompt = f"""
        {get_ensemble_prompt(row.text, ensemble_data, cfg)}
        Text: {row.text}"""
    else:
        serializable_examples = []
        final_prompt = f"""
        {cfg.prompt.get(cfg.model.template)}
        Text: {row.text}
        """
    return final_prompt, serializable_examples

def call_setup_rag(cfg, df_isear):
    # Initialize RAG
    rag = None
    try:
        # RAG/ReAct 초기화가 필요한 경우 체크
        needs_rag = "rag" in cfg.model.template.lower()
        needs_react = "react" in cfg.model.template.lower() or "cot_react" in cfg.model.template.lower()
        
        if needs_rag or needs_react:
            logger.info(f"Initializing EmotionRAG for {'RAG' if needs_rag else 'ReAct'} templates...")
            try:
                # 임베딩 모델 초기화 상태 로깅 추가
                logger.info(f"Initializing RAG with embedding model: {cfg.rag.embedding_model}")
                rag = EmotionRAG(cfg, logger=logger)
                
                # RAG 초기화 후 임베딩 모델 상태 확인
                if not rag.embedding_model:
                    raise ValueError("Embedding model not initialized properly")
                    
                if needs_rag:  # RAG 기능이 필요한 경우만 인덱스 생성
                    logger.info("Creating RAG index...")
                    rag.create_index(df_isear)
                    logger.info("RAG index creation completed")
                logger.info(f"EmotionRAG initialization completed for {'RAG' if needs_rag else 'ReAct'}")
            except Exception as e:
                logger.error(f"Error initializing EmotionRAG: {str(e)}")
                logger.error(f"Stack trace: {traceback.format_exc()}")
                logger.warning(f"Continuing without {'RAG' if needs_rag else 'ReAct'} features...")
                rag = None  # RAG 초기화 실패시 None으로 설정
                return rag
    except Exception as e:
        logger.error(f"Error in initialization: {str(e)}")
        logger.warning("Continuing without RAG/ReAct features...")
        rag = None
        return rag
    return rag

def log_prompt(cfg, examples, rag, final_prompt, prompt_file_path):
    """프롬프트 로깅"""
    global _PROMPT_LOGGED
    
    if _PROMPT_LOGGED:
        return
    
    try:
        # 특수 문자 처리
        def clean_text(text):
            if not isinstance(text, str):
                return str(text)
            # ASCII 범위를 벗어나는 문자를 제거하고 공백으로 대체
            return ''.join(char if ord(char) < 128 else ' ' for char in text)
        
        # 예제 데이터 정제
        if examples:
            cleaned_examples = []
            for example in examples:
                cleaned_example = {
                    'text': clean_text(example.get('text', '')),
                    'metadata': example.get('metadata', {}),
                    'similarity': example.get('similarity', 0)
                }
                cleaned_examples.append(cleaned_example)
            examples_str = json.dumps(cleaned_examples, ensure_ascii=False, indent=2)
        else:
            examples_str = "Not using RAG"
        
        # 프롬프트 로그 생성
        prompt_log = f"""
================================================================================
1. Configuration:
- Template Type: {cfg.model.template}
================================================================================
3. Retrieved Examples (if RAG): 
{examples_str}

4. Final Prompt (Model Input):
================================================================================
{clean_text(final_prompt)}
================================================================================
"""
        
        # 프롬프트 로그 저장
        logger.info(prompt_log)
        
        # 프롬프트 파일 저장
        with open(prompt_file_path, 'w', encoding='utf-8') as f:
            f.write(prompt_log)
        
        _PROMPT_LOGGED = True
        
    except Exception as e:
        logger.error(f"프롬프트 로깅 중 오류 발생: {str(e)}")
        logger.error(f"스택 트레이스: {traceback.format_exc()}")

def load_ensemble_data(cfg, logger):
    """Load prediction data from multiple model folders for ensemble"""
    if not 'ensemble' in cfg.model.template or not cfg.model.ensemble.model_folders:
        return None
    
    ensemble_data = {}
    for folder in cfg.model.ensemble.model_folders:
        try:
            folder_path = Path(cfg.general.result_path) / Path(folder)
            if not folder_path.exists():
                logger.warning(f"Ensemble folder not found: {folder}")
                continue
                
            # Find the dataset CSV file
            csv_files = list(folder_path.glob(f"dataset-{cfg.data.name}_model-*.csv"))
            if not csv_files:
                logger.warning(f"No dataset CSV found in {folder}")
                continue
                
            # Load the CSV file
            df = pd.read_csv(csv_files[0])
            
            # Extract model name from filename
            model_name = csv_files[0].stem.split('_model-')[-1]
            
            # Get required columns
            required_cols = ['text', 'emotion',f'predicted_emotion_{model_name}', 
                           f'confidence_score_{model_name}', f'explanation_{model_name}']
            
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"Missing required columns in {csv_files[0]}")
                continue
                
            # Store the dataframe with model name
            ensemble_data[model_name] = df[required_cols]
            logger.info(f"Loaded ensemble data from {folder} for model {model_name}")
            
        except Exception as e:
            logger.error(f"Error loading ensemble data from {folder}: {str(e)}")
            continue
    
    return ensemble_data if ensemble_data else None

def get_ensemble_prompt(text: str, ensemble_data: Dict[str, pd.DataFrame], cfg) -> str:
    """Generate ensemble prompt using predictions from multiple models"""
    model_predictions = []
    
    # Get predictions from each model
    for model_name, df in ensemble_data.items():
        # Find matching text row
        matching_row = df[df['text'] == text]
        if not matching_row.empty:
            # Get ground truth and prediction
            ground_truth = matching_row['emotion'].iloc[0]
            predicted_emotion = matching_row[f'predicted_emotion_{model_name}'].iloc[0]
            confidence_score = matching_row[f'confidence_score_{model_name}'].iloc[0]
            explanation = matching_row[f'explanation_{model_name}'].iloc[0]
            
            prediction = {
                'model': model_name,
                'predicted_emotion': predicted_emotion,
                'confidence_score': confidence_score,
                'explanation': explanation,
                
            }
            if cfg.model.ensemble.is_correct:
                 # Check if prediction matches ground truth
                is_correct = ground_truth == predicted_emotion
                prediction['is_correct'] = is_correct
            model_predictions.append(prediction)
    
    # Calculate overall accuracy of previous predictions
    # total_predictions = len(model_predictions)
    # correct_predictions = sum(1 for pred in model_predictions if pred['is_correct'])
    # overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    if cfg.model.ensemble.is_correct:
        snippet_is_correct = 'Yes' if cfg.model.ensemble.is_correct else 'No'
        snippet_is_correct = f'Is Correct: {snippet_is_correct}, '
    else:
        snippet_is_correct = ''
    # Generate the prompt
    prompt = f"""## Instructions
You are an expert in emotion analysis.
Classify the primary emotion in the given text strictly as one of: {', '.join(cfg.data.datasets[cfg.data.name].labels)}.

## Role
You are an expert synthesizing insights from multiple expert models. Evaluate the following model predictions carefully and integrate them, giving higher weight to predictions with higher confidence scores. Use the detailed guidelines provided below to resolve any ambiguities, especially for the emotions: anger, disgust, shame, and guilt.

<Previous Model Predictions>
{chr(10).join(f"{i+1}. Model: {pred['model']}, Predicted Emotion: {pred['predicted_emotion']}, {snippet_is_correct}Confidence Score: {pred['confidence_score']:.2f}, Explanation: {pred['explanation']}" for i, pred in enumerate(model_predictions))}
</Previous Model Predictions>

## Detailed Emotion Definitions and Analysis Guidelines:
- **Joy:** happiness, pleasure, delight.
- **Fear:** anxiety, dread regarding danger or uncertainty.
- **Anger:** An emotion characterized by feelings of frustration or rage, often arising from a perceived injustice or violation of one’s rights. Anger involves a drive to address the perceived wrong and is typically focused on external factors rather than personal shortcomings.
- **Sadness:** sorrow, grief, disappointment.
- **Disgust:** A strong, aversive emotion characterized by revulsion or repulsion toward something offensive, unclean, or morally objectionable. It involves an immediate aversive reaction and often includes physical expressions such as wrinkling the nose or turning away.
- **Shame:** A self-conscious emotion arising from a perception of personal inadequacy or failure, often accompanied by a negative evaluation of the self. It is linked to feelings of exposure to social judgment or humiliation.
- **Guilt:** An emotion experienced when recognizing that one’s actions have caused harm or violated moral standards, associated with remorse and a desire to make amends.

### Pairwise Emotional Comparisons:
1. **Anger vs. Disgust:**  
   - *Anger* focuses on perceived injustice and involves a drive for corrective action, whereas *Disgust* is about repulsion and an immediate aversive reaction to something offensive or unclean.
2. **Shame vs. Guilt:**  
   - *Shame* emphasizes a negative evaluation of the self, often feeling inherently flawed, while *Guilt* focuses on remorse over specific actions and a desire to correct or apologize for the wrongdoing.

## Additional Synthesis Guidelines:
1. Weight predictions based on confidence scores.
2. Prioritize textual evidence and context from both the input text and previous model explanations.
3. In ambiguous cases, use the pairwise comparisons and detailed definitions above to resolve conflicts.
4. An external binary correctness indicator (Is Correct) may be provided as part of the context. This indicator is critical to your decision-making:
   - If the indicator is "No", it suggests that the previous model predictions might be unreliable and should be either discounted or re-evaluated more critically.
   - If the indicator is "Yes", then the previous predictions are likely reliable and should be given higher weight.
5. Integrate this external correctness indicator into your final synthesis process, adjusting the influence of the previous predictions accordingly.

""" + cfg.prompt.modules.output_format

    return prompt



@hydra.main(version_base="1.2", config_path='config', config_name='llm')
def main(cfg):
    # Initialize models
    client = initialize_models(cfg)
    

    # Convert config values to Python basic types first
    model = str(cfg.model.name)
    
    # Set up output directory first
    model_name = sanitize_model_name(model)
    output_dir_name = get_output_dir_name(model_name, cfg)
    output_dir = Path('outputs') / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logs directory
    logs_dir = output_dir / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Update cfg output_path
    cfg.general.output_path = str(output_dir)
    
    # Set output file path (directory already created)
    output_path = Path(cfg.general.output_path) / f'dataset-{cfg.data.name}_model-{model_name}.csv'
    
    # 첫 프롬프트 저장을 위한 파일 경로 설정
    prompt_file_path = output_dir / f'initial_prompts_{model_name}_{cfg.model.template}.txt'
    
    # Now setup logging with the created output directory
    logger = setup_logging(cfg, output_dir)

    project_name = "LLM based emotion classification"
    if cfg.data.n_samples == -1:
        message = f"\n=================\n{project_name}\n=================\nRun name: {cfg.general.run_name}\nData: {cfg.data.name}\nModel: {cfg.model.name}\nPrompt Template: {cfg.model.template}\nEmbedding Model: {cfg.rag.embedding_model}\ncwd: {os.getcwd()}\n\nTraining Starts"
        asyncio.run(send_telegram_message(message, 
                                        os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")))    
        
        # Load ensemble data if enabled
    ensemble_data = None
    if 'ensemble' in cfg.model.template:
        print("####################################################################")
        logger.info("Loading ensemble data...")
        ensemble_data = load_ensemble_data(cfg, logger)
        if ensemble_data is None:
            logger.warning("Failed to load ensemble data. Proceeding without ensemble.")
    
    ########################################################################
    # 데이터 로딩 및 전처리
    df_isear, labels = load_data(cfg, logger)
    # Set n_samples
    n_samples = len(df_isear) if cfg.data.n_samples == -1 else min(cfg.data.n_samples, len(df_isear))
    logger.info(f"Using sanitized model name: {model_name}")
    logger.info(f"Processing {n_samples} samples out of {len(df_isear)} total samples")
    # Select first n_samples
    df_isear = df_isear.head(n_samples)

    labels = list(map(str, labels))  # labels를 문자열로 변환
    # emotion 컬럼이 숫자인 경우 문자열로 변환
    if df_isear['emotion'].dtype in ['int64', 'float64']:
        df_isear['emotion'] = df_isear['emotion'].map(lambda x: labels[int(x)-1] if pd.notnull(x) else 'undefined')
    elif df_isear['emotion'].dtype == 'object':  # 문자열이지만 숫자로 저장된 경우
        df_isear['emotion'] = df_isear['emotion'].map(lambda x: labels[int(x)-1] if str(x).isdigit() else x)
    
    # Initialize invalid prediction statistics
    invalid_predictions = {
        "total": 0,
        "invalid_samples": [],
        "by_true_label": {str(label): 0 for label in labels}
    }
 # 토큰 카운트 통계를 위한 변수
    total_tokens = 0
    ###################################
    # 데이터 정제 후에 tqdm을 사용하도록 수정
    total_samples = len(df_isear)  # 실제 처리할 샘플 수
    tic = time.time()

    # RAG 모델 초기화 (한 번만 수행)
    rag = call_setup_rag(cfg, df_isear)
    
    for index, row in tqdm(df_isear.iterrows(), total=total_samples):
        try:
            final_prompt, serializable_examples = call_prompt(cfg, row, index, rag, ensemble_data)
            ################## Prompt prep
            if not _PROMPT_LOGGED:
                log_prompt(cfg, serializable_examples, rag, final_prompt, prompt_file_path)
            
            result = get_model_response(
                client=client,
                text=str(row.text),
                model_name=model,
                prompt_name=cfg.model.template,
                fallback_prompts=None,
                logger=logger,
                cfg=cfg,
                rag=rag,
                model=model,
                final_prompt=final_prompt,
                ensemble_data=ensemble_data
            )
            
            if result:
                df_isear.at[index, f'predicted_emotion_{model}'] = result["emotion"]
                df_isear.at[index, f'confidence_score_{model}'] = result["confidence_score"]
                df_isear.at[index, f'explanation_{model}'] = result["explanation"]
                if cfg.general.retry.save_retry_count:
                    df_isear.at[index, f'retry_count_{model}'] = result["retry_count"]
            else:
                df_isear.at[index, f'predicted_emotion_{model}'] = "unknown"
                df_isear.at[index, f'confidence_score_{model}'] = 0.0
                df_isear.at[index, f'explanation_{model}'] = "Failed to process response"
            
            # 예측 결과 로깅 (한 번만 수행)
            if logger and cfg.general.logging.save_responses:
                result_log = f"""
{log_separator}
[Sample Index: {index}] Prediction Result
{log_separator}
Input Text: {str(row.text).encode('ascii', errors='ignore').decode('ascii')}
Ground Truth: {str(row.emotion)}
Predicted Emotion: {str(result['emotion']).encode('ascii', errors='ignore').decode('ascii')}
Confidence Score: {float(result['confidence_score']):.2f}
Explanation: {str(result['explanation']).encode('ascii', errors='ignore').decode('ascii')}
{log_separator}
"""
                logger.info(result_log)
            
            if index % cfg.general.logging.log_interval == 0:
                df_isear.to_csv(output_path, index=False)

        except Exception as e:
            print(f"Error processing row {index}: {e}")
           
            df_isear.at[index, f'predicted_emotion_{model}'] = "unknown"
            df_isear.at[index, f'confidence_score_{model}'] = 0.0
            df_isear.at[index, f'explanation_{model}'] = f"Error: {str(e)}"
            
            error_log = f"""
{log_separator}
[Error at Sample Index: {index}]
{log_separator}
Input Text: {str(row.text).encode('ascii', errors='ignore').decode('ascii')}
Ground Truth: {str(row.emotion)}
Error: {str(e)}
{log_separator}
"""
            logger.error(error_log)
            continue
    ###############################################################################
    # Final result save
    df_isear.to_csv(output_path, index=False)
    
    # Calculate and save evaluation metrics
    report, cm = save_metrics(output_path, cfg)
    
    # 효율성 지수 계산 및 저장
    # efficiency_df, f1_per_time = calculate_efficiency_metrics(df_isear, model, model_name, output_dir)
    
    # # 효율성 지수 저장
    # efficiency_path = output_dir / f'efficiency_metrics_{model_name}.csv'
    # efficiency_df.to_csv(efficiency_path, index=False)
    # logger.info(f"Saved efficiency metrics to {efficiency_path}")
    
    # 토큰 통계 로깅
    token_stats = f"""
{log_separator}
[Token Usage Statistics]
{log_separator}
Total Samples: {len(df_isear)}
{log_separator}
"""
    logger.info(token_stats)
    
    # 잘못 분류된 샘플 찾기
    mask = df_isear['emotion'] != df_isear[f'predicted_emotion_{model}']
    df_misclassified = df_isear[mask].copy()
    df_misclassified.sort_values(by='emotion', ascending=False, inplace=True)
    misclassified_counts = df_misclassified['emotion'].value_counts()
    logger.info(misclassified_counts)

    # Save misclassified samples
    df_misclassified.to_csv(output_dir / f'misclassified_samples_{model_name}.csv', index=False)

    # Save invalid prediction statistics
    invalid_stats = {
        "total_samples": total_samples,
        "invalid_count": invalid_predictions["total"],
        "invalid_percentage": (invalid_predictions["total"] / total_samples) * 100,
        "by_true_label": invalid_predictions["by_true_label"],
        "invalid_samples": invalid_predictions["invalid_samples"]
    }
    
    print(f"\nInvalid prediction statistics:")
    print(f"Total invalid: {invalid_stats['invalid_count']}")
    print(f"Invalid percentage: {invalid_stats['invalid_percentage']:.2f}%")
    print(f"Number of invalid samples collected: {len(invalid_stats['invalid_samples'])}")
    
    # Save statistics
    save_prediction_stats(invalid_stats, output_dir, model_name, labels)

    # Save RAG search results and analyze performance if available
    if rag and hasattr(rag, 'search_results'):
        # Save search results
        rag_results_path = output_dir / f'rag_search_results_{model_name}.csv'
        rag.save_search_results(str(rag_results_path))
        logger.info(f"Saved RAG search results to {rag_results_path}")
        
        # Analyze retrieval performance
        rag_stats = rag.analyze_retrieval_performance(output_dir)
        
        # Add RAG performance to final log if stats are available
        if rag_stats:
            rag_log = "\n4. RAG Retrieval Performance:"
            rag_log += f"\n- Overall accuracy: {rag_stats['overall']['accuracy']:.2%}"
            rag_log += f"\n- Total queries: {rag_stats['overall']['total_queries']}"
            rag_log += f"\n- Total retrievals: {rag_stats['overall']['total_retrievals']}"
            rag_log += "\n\nClass-wise RAG Performance:"
            for emotion, stats in rag_stats['by_class'].items():
                rag_log += f"\n{emotion}:"
                rag_log += f"\n  - Accuracy: {stats['accuracy']:.2%}"
                rag_log += f"\n  - Queries: {stats['total_queries']}"
                rag_log += f"\n  - Correct matches: {stats['correct_matches']}"
        else:
            logger.warning("No RAG statistics available for analysis")
            rag_log = "\n4. RAG Retrieval Performance: No data available"
    else:
        rag_log = ""
    marco_avg = ', '.join(f"{key}: {value:.4f}" for key, value in report['overall']['macro_avg'].items())
    # Final results log creation
    final_log = f"""
{'='*80}
[Final Results Summary]
{'='*80}

1. Wrong Prediction Statistics:
- Total samples: {invalid_stats['total_samples']}
- Wrong predictions: {invalid_stats['invalid_count']}
- Wrong prediction percentage: {invalid_stats['invalid_percentage']:.2f}%

2. Classification Performance:
- Overall accuracy: {report['overall']['accuracy']:.4f}
- Macro Average : {marco_avg}


3. Class-wise Performance:"""

    # Add per-class performance metrics
    for label, metrics in report['per_class'].items():
        final_log += f"""
{label}:
  - Precision: {metrics['precision']:.4f}
  - Recall: {metrics['recall']:.4f}
  - F1 Score: {metrics['f1-score']:.4f}"""

    final_log += rag_log
    final_log += f"\n{'='*80}\n"
    
    # Log to file
    logger.info(final_log)
    
    toc = time.time()
    time_taken = (toc - tic)/60  # minutes
    
    # 시간 메트릭을 report에 추가
    report['efficiency'] = {
        'total_time_minutes': time_taken,
        'time_per_sample_seconds': (time_taken * 60) / total_samples,
        'samples_per_minute': total_samples / time_taken
    }
    
    # 시간 메트릭을 별도 파일로 저장
    time_metrics = {
        'model_name': model_name,
        'template': cfg.model.template,
        'total_samples': total_samples,
        'total_time_minutes': time_taken,
        'time_per_sample_seconds': (time_taken * 60) / total_samples,
        'samples_per_minute': total_samples / time_taken,
        'accuracy': report['overall']['accuracy'],
        'macro_avg_f1': report['overall']['macro_avg']['f1-score'],
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 시간 메트릭 저장
    time_metrics_path = output_dir / f'time_metrics_{model_name}.json'
    with open(time_metrics_path, 'w', encoding='utf-8') as f:
        json.dump(time_metrics, f, indent=2)
    logger.info(f"Saved time metrics to {time_metrics_path}")
    
    message = f"\n===============\n{project_name}\n=============\nRun name: {cfg.general.run_name}\nData: {cfg.data.name}\nModel: {cfg.model.name}\nPrompt Template: {cfg.model.template}"
    if "rag" in cfg.model.template:
        message += f"\nEmbedding Model: {cfg.rag.embedding_model}\n"
    
    message += f"\n -> Training is completed!\n\n- Accuracy: {report['overall']['accuracy']:.4f}\n- Macro Average : {marco_avg}\n(Total Samples: {total_samples})\n- Wrong prediction count: {invalid_predictions['total']}\n- Wrong prediction percentage: {invalid_predictions['total']/total_samples:.2f}%\nTime taken: {time_taken:.2f} minutes\n\nCwd: {os.getcwd()}"
    
    if cfg.data.n_samples == -1:
        asyncio.run(send_telegram_message(message, 
                                        os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")))
    logger.info(message)

if __name__ == "__main__":
    main()
