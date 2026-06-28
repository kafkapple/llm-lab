import json
import pandas as pd
from pathlib import Path
import re
from src.logger import safe_log
from typing import Dict
HIERARCHY = {
    "admiration": "Positive",
    "amusement": "Positive",
    "approval": "Positive",
    "caring": "Positive",
    "desire": "Positive",
    "excitement": "Positive",
    "gratitude": "Positive",
    "joy": "Positive",
    "love": "Positive",
    "optimism": "Positive",
    "pride": "Positive",
    "relief": "Positive",


    "anger": "Negative",
    "annoyance": "Negative",
    "disappointment": "Negative",
    "disapproval": "Negative",
    "disgust": "Negative",
    "embarrassment": "Negative",
    "fear": "Negative",
    "grief": "Negative",
    "nervousness": "Negative",
    "remorse": "Negative",
    "sadness": "Negative",

    "confusion": "Ambiguous",
    "curiosity": "Ambiguous",
    "realization": "Ambiguous",
    "surprise": "Ambiguous",

    "neutral": "neutral"
}
EKMAN_MAPPING = {
    "anger": "anger",
    "annoyance": "anger",
    "disapproval": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "nervousness": "fear",
    "joy": "joy",
    "amusement": "joy",
    "excitement": "joy",
    "gratitude": "joy",
    "love": "joy",
    "optimism": "joy",
    "pride": "joy",
    "relief": "joy",
    "realization": "joy",
    "admiration": "joy",
    "approval": "joy",

    "sadness": "sadness",
    "disappointment": "sadness",
    "grief": "sadness",
    "remorse": "sadness",
    "embarrassment": "sadness",
    "surprise": "surprise",
    "neutral" :"neutral"
}
ISEAR_MAPPING = {
    "anger": "anger",
    "annoyance": "anger",
    "disapproval": "anger",

    "disgust": "disgust",

    "fear": "fear",
    "nervousness": "fear",

    "joy": "joy",
    "amusement": "joy",
    "excitement": "joy",
    "gratitude": "joy",
    "love": "joy",
    "optimism": "joy",
    "pride": "joy",
    "relief": "joy",
    "realization": "joy",  # Keeping realization in joy
    "admiration": "joy",   # Keeping admiration in joy
    "approval": "joy",     # Keeping approval in joy

    "sadness": "sadness",
    "disappointment": "sadness",
    "grief": "sadness",

    "remorse": "guilt",

    "embarrassment": "shame",

    "surprise": "other",  # No direct match
    "caring": "other",  # No direct match
    "confusion": "other",  # No direct match
    "curiosity": "other",  # No direct match
    "desire": "other",  # No direct match
    "neutral": "neutral"  # Keeping neutral
}

def detect_encoding(file_path):
    """파일의 인코딩을 자동으로 감지"""
    import chardet
    
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        return result['encoding']

def load_data(cfg, logger=None):
    dataset_cfg = cfg.data.datasets[cfg.data.name]
    labels = dataset_cfg.labels
    
    # 인코딩 감지 또는 기본값 사용
    file_path = dataset_cfg.path
    if cfg.data.auto_detect_encoding:
        detected_encoding = detect_encoding(file_path)
        if logger:
            logger.info(f"File encoding detection:")
            logger.info(f"- File path: {file_path}")
            logger.info(f"- Detected encoding: {detected_encoding}")
            logger.info(f"- Auto detection enabled: {cfg.data.auto_detect_encoding}")
        encoding = detected_encoding
    else:
        encoding = cfg.data.encoding if hasattr(cfg.data, 'encoding') else 'utf-8'
        if logger:
            logger.info(f"File encoding settings:")
            logger.info(f"- File path: {file_path}")
            logger.info(f"- Using specified encoding: {encoding}")
            logger.info(f"- Auto detection disabled: {cfg.data.auto_detect_encoding}")
    
    try:
        df = pd.read_csv(
            file_path,
            sep=dataset_cfg.separator,
            on_bad_lines='skip',
            encoding=encoding,
            engine='python',
            quoting=3,
            dtype=str
        )
        if logger:
            logger.info(f"Successfully read CSV file with encoding: {encoding}")
    except UnicodeDecodeError:
        if logger:
            logger.warning(f"Failed to read with {encoding} encoding, trying with 'utf-8'")
            logger.info(f"Fallback encoding attempt:")
            logger.info(f"- Original encoding: {encoding}")
            logger.info(f"- Fallback encoding: utf-8")
        df = pd.read_csv(
            file_path,
            sep=dataset_cfg.separator,
            on_bad_lines='skip',
            encoding='utf-8',
            engine='python',
            quoting=3,
            dtype=str
        )
        if logger:
            logger.info(f"Successfully read CSV file with fallback encoding: utf-8")
    
    required_columns = dataset_cfg.required_columns
    
    if not all(col in df.columns for col in required_columns):
        df.columns = [col.split(dataset_cfg.separator)[0] for col in df.columns]
    
    print(f"Labels: {labels}\nRequired Columns: {required_columns}\nColumn Mapping: {dataset_cfg.column_mapping}")
    df = df[required_columns]
    df.rename(columns=dataset_cfg.column_mapping, inplace=True)
    #df.rename(columns={'SIT': 'text', 'EMOT': 'emotion'}, inplace=True)

    # 레이블 변환 전 로깅
    if logger:
        logger.info(f"Original emotion value counts:\n{df['emotion'].value_counts()}")
        logger.info(f"Emotion column dtype: {df['emotion'].dtype}")
    
    # 레이블 변환 로직 개선
    def convert_label(x, labels):
        try:
            if pd.isna(x):
                return 'undefined'
            if isinstance(x, (int, float)):
                return labels[int(x)-1]
            if str(x).isdigit():
                return labels[int(x)-1]
            return str(x)
        except (IndexError, ValueError) as e:
            if logger:
                logger.warning(f"Error converting label {x}: {str(e)}")
            return 'undefined'
    
    df['emotion'] = df['emotion'].apply(lambda x: convert_label(x, labels))
    
    # 변환 후 로깅
    if logger:
        logger.info(f"Converted emotion value counts:\n{df['emotion'].value_counts()}")
        logger.info(f"Final emotion column dtype: {df['emotion'].dtype}")
    
    print(f"Emotion value counts after conversion:\n{df['emotion'].value_counts()}")
    
    def preprocess_text(text):
        if not isinstance(text, str):
            return text
        text = ' '.join(text.split())
        text = text.lower()
        return text
    
    # Log message construction
    log_messages = []
    log_messages.append("\nData preprocessing stats before:")
    log_messages.append(f"Total samples: {len(df)}")
    log_messages.append(f"Missing values:\n{df.isna().sum()}")
    log_messages.append(f"Duplicates: {df.duplicated().sum()}")
    
    # Create subfolder for saving data-related files
    data_dir = Path(cfg.general.output_path) / "data_stats"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Save missing data
    missing_data = df[df.isna().any(axis=1)].copy()
    if not missing_data.empty:
        missing_data['removal_reason'] = 'missing_value'
        missing_data['status'] = 'removed'
        missing_data.to_csv(data_dir / 'missing_data.csv', index=True)
        log_messages.append(f"\nSaved {len(missing_data)} rows with missing values")
    
    # Remove missing values
    df_no_missing = df.dropna()
    
    # Apply text preprocessing
    df_no_missing['text'] = df_no_missing['text'].apply(preprocess_text)
    
    # exclude_phrases processing
    excluded_rows = pd.DataFrame()
    if cfg.data.exclude_phrases:
        exclude_phrases = list(cfg.data.exclude_phrases)
        original_len = len(df_no_missing)
        
        # Save rows to exclude in advance
        for phrase in exclude_phrases:
            matched_rows = df_no_missing[
                (df_no_missing['text'].str.contains(phrase, case=False, na=False)) & 
                (df_no_missing['text'].str.len() < cfg.data.exclude_length)
            ].copy()
            
            if not matched_rows.empty:
                matched_rows['removal_reason'] = f'contains_phrase: {phrase}'
                matched_rows['status'] = 'removed'
                excluded_rows = pd.concat([excluded_rows, matched_rows])
        
        # Remove rows containing excluded phrases (including length condition)
        for phrase in exclude_phrases:
            df_no_missing = df_no_missing[
                ~((df_no_missing['text'].str.contains(phrase, case=False, na=False)) & 
                  (df_no_missing['text'].str.len() < cfg.data.exclude_length))
            ]
        
        filtered_len = len(df_no_missing)
        log_messages.append(f"\nRemoved {original_len - filtered_len} rows containing excluded phrases (length < {cfg.data.exclude_length})")
        
        # Save excluded rows
        if not excluded_rows.empty:
            excluded_rows.to_csv(data_dir / 'excluded_phrases_data.csv', index=True)
            log_messages.append(f"Saved {len(excluded_rows)} rows containing excluded phrases")
    
    # Duplicate data processing
    duplicates_all = df_no_missing[df_no_missing.duplicated(subset=['text'], keep=False)].copy()
    if not duplicates_all.empty:
        # Add group ID to duplicate data
        duplicates_all['duplicate_group'] = duplicates_all.groupby('text').ngroup()
        
        # Only first item in each group is kept, others are marked for removal
        duplicates_all['status'] = 'removed'
        duplicates_all.loc[~duplicates_all.duplicated(subset=['text'], keep='first'), 'status'] = 'kept'
        
        # Sort by duplicate_group and then by status (kept comes first)
        duplicates_all = duplicates_all.sort_values(['duplicate_group', 'status'], 
                                                  ascending=[True, True])
        
        # Save duplicate data (both kept and removed)
        duplicates_all.to_csv(data_dir / 'duplicate_data.csv', index=True)
        log_messages.append(f"\nSaved {len(duplicates_all)} rows of duplicate data (including kept and removed)")
        log_messages.append(f"- Kept: {len(duplicates_all[duplicates_all['status'] == 'kept'])} rows")
        log_messages.append(f"- Removed: {len(duplicates_all[duplicates_all['status'] == 'removed'])} rows")
    
    # Remove duplicates (keep first item)
    df_final = df_no_missing.drop_duplicates(subset=['text'], keep='first')
    
    # Calculate emotion class statistics
    emotion_stats = {}
    for emotion in labels:
        initial_count = len(df[df['emotion'] == emotion])
        excluded_count = len(excluded_rows[excluded_rows['emotion'] == emotion])
        duplicates_removed = len(duplicates_all[(duplicates_all['emotion'] == emotion) & 
                                              (duplicates_all['status'] == 'removed')]) if not duplicates_all.empty else 0
        final_count = len(df_final[df_final['emotion'] == emotion])
        
        emotion_stats[emotion] = {
            'initial_count': initial_count,
            'excluded_count': excluded_count,
            'duplicates_removed': duplicates_removed,
            'final_count': final_count,
            'total_removed': excluded_count + duplicates_removed,
            'removal_percentage': ((initial_count - final_count) / initial_count * 100) if initial_count > 0 else 0
        }
    
    # Save removed data statistics
    removal_stats = {
        'total_initial_samples': len(df),
        'total_final_samples': len(df_final),
        'removed_missing_values': len(missing_data),
        'removed_excluded_phrases': len(excluded_rows),
        'duplicate_stats': {
            'total_duplicates': len(duplicates_all) if not duplicates_all.empty else 0,
            'kept_duplicates': len(duplicates_all[duplicates_all['status'] == 'kept']) if not duplicates_all.empty else 0,
            'removed_duplicates': len(duplicates_all[duplicates_all['status'] == 'removed']) if not duplicates_all.empty else 0
        },
        'excluded_phrases_list': list(cfg.data.exclude_phrases) if cfg.data.exclude_phrases else [],
        'emotion_class_stats': emotion_stats
    }
    
    with open(data_dir / 'removal_stats.json', 'w', encoding='utf-8') as f:
        json.dump(removal_stats, f, indent=2, ensure_ascii=False)
    
    log_messages.append("\nData preprocessing stats after:")
    log_messages.append(f"Total samples: {len(df_final)}")
    log_messages.append(f"Total removed samples:")
    log_messages.append(f"- Missing values: {len(missing_data)}")
    log_messages.append(f"- Excluded phrases: {len(excluded_rows)}")
    log_messages.append(f"- Duplicates: {len(duplicates_all[duplicates_all['status'] == 'removed']) if not duplicates_all.empty else 0}")
    log_messages.append("\nEmotion class statistics:")
    for emotion, stats in emotion_stats.items():
        log_messages.append(f"\n{emotion}:")
        log_messages.append(f"- Initial count: {stats['initial_count']}")
        log_messages.append(f"- Excluded: {stats['excluded_count']}")
        log_messages.append(f"- Duplicates removed: {stats['duplicates_removed']}")
        log_messages.append(f"- Final count: {stats['final_count']}")
        log_messages.append(f"- Removal percentage: {stats['removal_percentage']:.2f}%")
    
    # Combine log messages into a single string
    log_message = "\n".join(log_messages)
    
    # Log to file
    if logger:
        safe_log(logger, 'info', log_message)
    else:
        print(log_message)
    if cfg.data.save_preprocessed_data:
        print(data_dir / 'preprocessed_data.csv')
        df_final.to_csv(data_dir / 'preprocessed_data.csv', index=False)
    
    return df_final, labels

def extract_json_from_text(text):
    """텍스트에서 JSON 형식의 문자열을 추출합니다.
    
    Args:
        text (str): JSON을 포함할 수 있는 텍스트
        
    Returns:
        str: 추출된 JSON 문자열
        
    Raises:
        ValueError: JSON을 찾을 수 없는 경우
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
        
    # Find JSON-like structure
    json_pattern = r'\{[^{}]*\}'
    matches = re.finditer(json_pattern, text)
    
    # Get the last (usually most complete) JSON match
    json_str = None
    for match in matches:
        json_str = match.group()
    
    if not json_str:
        raise ValueError("No JSON structure found in text")
        
    return json_str

def clean_json_response(response, logger=None, cfg=None):
    """LLM 응답에서 JSON을 추출하고 정리합니다.
    
    Args:
        response (str): LLM의 원본 응답
        logger: 로깅을 위한 logger 객체 (선택사항)
        cfg: 설정 객체 (선택사항)
    
    Returns:
        dict: 정리된 JSON 응답
    
    Raises:
        json.JSONDecodeError: JSON 파싱 실패 시 (retry_on_json_error가 True인 경우)
        ValueError: 필수 필드 누락 시 (retry_on_missing_fields가 True인 경우)
    """
    if logger:
        logger.debug(f"Raw LLM response:\n{response}")
    
    # 설정이 없는 경우 기본값 설정
    retry_on_json_error = True
    retry_on_missing_fields = True
    if cfg and hasattr(cfg.general, 'retry'):
        retry_on_json_error = getattr(cfg.general.retry, 'retry_on_json_error', True)
        retry_on_missing_fields = getattr(cfg.general.retry, 'retry_on_missing_fields', True)
    
    try:
        # JSON 부분 추출
        json_str = extract_json_from_text(response)
        if logger:
            logger.debug(f"Extracted JSON:\n{json_str}")
        
        # JSON 파싱
        result = json.loads(json_str)
        
        # 필수 필드 확인
        required_fields = ["emotion", "confidence_score", "explanation"]
        for field in required_fields:
            if field not in result:
                if logger:
                    logger.warning(f"Missing required field in parsed result: {field}")
                if retry_on_missing_fields:
                    raise ValueError(f"Missing required field: {field}")
                else:
                    result[field] = None if field != "confidence_score" else 0.5
        
        if logger:
            logger.debug(f"Final parsed result:\n{json.dumps(result, indent=2)}")
        
        return result
        
    except json.JSONDecodeError as e:
        if logger:
            logger.error(f"JSON parsing error: {str(e)}\nResponse: {response}")
        if retry_on_json_error:
            raise  # JSON 파싱 에러를 상위로 전파
        else:
            return {
                "emotion": None,
                "confidence_score": 0.5,
                "explanation": f"JSON parsing error: {str(e)}"
            }
    except Exception as e:
        if logger:
            logger.error(f"Unexpected error while cleaning response: {str(e)}\nResponse: {response}")
        if retry_on_json_error:
            raise  # 기타 에러를 상위로 전파
        else:
            return {
                "emotion": None,
                "confidence_score": 0.5,
                "explanation": f"Unexpected error: {str(e)}"
            }

