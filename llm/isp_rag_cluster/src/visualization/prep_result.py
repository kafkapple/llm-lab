import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import dataframe_image as dfi
from tqdm import tqdm
from omegaconf import DictConfig
from .constants import (
    DEFAULT_DPI, FILE_PATTERNS, METRIC_COLUMNS
)
from .plot import plot_unknown_analysis, plot_rag_search_analysis
import json
import traceback
# 로깅 설정
def setup_logging():
    """로깅 설정"""
    # 로그 디렉토리 생성
    log_dir = Path('./logs')
    log_dir.mkdir(exist_ok=True)
    
    # 파일 핸들러 설정
    file_handler = logging.FileHandler(log_dir / 'prep_result.log', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 콘솔 핸들러 설정 (인코딩 문제 해결)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 포맷터 설정
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 로거 설정
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 새 핸들러 추가
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 로거 초기화
logger = setup_logging()

# ---------------------------
# 2. 유틸리티 함수
# ---------------------------
col_name_model = 'model'
col_name_prompt = 'prompt'
def highlight_max(s, style='text-color'):
    """
    Series에서 최대값과 최소값에 대해 스타일 지정
    - 최대값: 초록색 텍스트 (#2ecc71)
    - 최소값: 빨간색 텍스트 (#e74c3c)
    """
    is_max = s == s.max()
    is_min = s == s.min()
    if style == 'text-color':
        return [f'color: #2ecc71; font-weight: bold' if v else f'color: #e74c3c; font-weight: bold' if m else '' 
                for v, m in zip(is_max, is_min)]
    else:  # background color
        return [f'background-color: #2ecc71' if v else f'background-color: #e74c3c' if m else '' 
                for v, m in zip(is_max, is_min)]

def save_table_as_image(df: pd.DataFrame, output_path: Path, logger: Optional[logging.Logger] = None, precision: int = 1, metric: str = 'all') -> bool:
    """DataFrame을 이미지로 저장"""
    try:
        if logger:
            logger.info("=== DataFrame Info ===")
            logger.info(f"Columns: {df.columns.tolist()}")
            logger.info(f"Column types: {df.dtypes}")
            logger.info(f"Metric filter: {metric}")
        
        # Model과 Prompt 컬럼 제외하고 숫자형 컬럼만 선택
        numeric_cols = []
        for col in df.columns:
            if isinstance(col, tuple):
                # MultiIndex의 경우 첫 번째 레벨이 'Model'이나 'Prompt'가 아니고, 두 번째 레벨이 숫자형인 경우만 선택
                if col[0] not in ['Model', 'Prompt']:
                    # 메트릭 필터링 적용
                    if metric == 'all' or col[1].upper() == metric.upper():
                        # 실제 데이터가 숫자형인지 확인
                        if pd.api.types.is_numeric_dtype(df[col]):
                            numeric_cols.append(col)
                            if logger:
                                logger.info(f"Selected numeric column: {col}, dtype: {df[col].dtype}")
                                logger.info(f"Sample values: {df[col].head()}")
            else:
                # 단일 인덱스의 경우 숫자형 컬럼만 선택
                if pd.api.types.is_numeric_dtype(df[col]):
                    numeric_cols.append(col)
                    if logger:
                        logger.info(f"Selected numeric column: {col}, dtype: {df[col].dtype}")
                        logger.info(f"Sample values: {df[col].head()}")
        
        if logger:
            logger.info(f"Total numeric columns: {len(numeric_cols)}")
            logger.info(f"Selected numeric columns: {numeric_cols}")
        
        # 스타일링 적용 (소수점 1자리로 반올림)
        styled_df = df.style.format(precision=precision)
        
        # 각 컬럼별로 최대값과 최소값 하이라이트 적용
        def highlight_max_min(data, color_max='#2ecc71', color_min='#e74c3c'):
            """각 컬럼별로 최대값과 최소값에 스타일 적용"""
            if pd.api.types.is_numeric_dtype(data):
                max_val = data.max()
                min_val = data.min()
                if logger:
                    logger.info(f"Column {data.name}: max={max_val}, min={min_val}")
                return ['color: ' + color_max + '; font-weight: bold' if v == max_val 
                       else 'color: ' + color_min + '; font-weight: bold' if v == min_val 
                       else '' for v in data]
            return [''] * len(data)
        
        # 각 숫자형 컬럼에 대해 스타일 적용
        for col in numeric_cols:
            if logger:
                logger.info(f"Applying style to column: {col}")
            styled_df = styled_df.apply(highlight_max_min, subset=[col], axis=0)
        
        # 이미지 파일 경로 생성 (CSV 파일 경로에서 확장자만 변경)
        img_path = output_path.with_name(output_path.stem + '.png')
        
        # DataFrame을 이미지로 저장
        dfi.export(styled_df, str(img_path), dpi=300)
        
        if logger:
            logger.info(f"Successfully saved table image to {img_path}")
        return True
        
    except Exception as e:
        if logger:
            logger.warning(f"Failed to save table image: {str(e)}")
            logger.warning(f"Error details: {traceback.format_exc()}")
        return False

# ---------------------------
# 3. 테이블 생성 함수 (gen_table)
# ---------------------------
# @safe_process_dataframe
def gen_table(
    df_info: pd.DataFrame,
    output_path: Union[str, Path],
    round_digits: int = 1,
    model_mapping: Optional[Dict] = None,
    prompt_mapping: Optional[Dict] = None,
    model_order: Optional[List[str]] = None,
    prompt_order: Optional[List[str]] = None,
    cfg: Optional[DictConfig] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """테이블 생성 및 저장"""
    try:
        # mapping이 None인 경우 빈 딕셔너리로 초기화
        model_mapping = model_mapping or {}
        prompt_mapping = prompt_mapping or {}
        
        # 설정에서 테이블 옵션 가져오기
        table_config = getattr(cfg, 'plotting', {}).get('table', {})
        metric_filter = table_config.get('metric', 'all')
        model_filter = table_config.get('models', 'all')
        prompt_filter = table_config.get('prompts', 'all')
        
        logger.info(f"Table configuration: metric={metric_filter}, models={model_filter}, prompts={prompt_filter}")
        
        # 모델과 프롬프트 순서가 지정되지 않은 경우 모든 고유값 사용
        if model_order is None:
            model_order = sorted(df_info['model'].unique())
        if prompt_order is None:
            prompt_order = sorted(df_info['prompt'].unique())
            
        # 필터링 적용
        if model_filter != 'all' and model_filter:
            model_order = [m for m in model_order if m in model_filter]
        if prompt_filter != 'all' and prompt_filter:
            prompt_order = [p for p in prompt_order if p in prompt_filter]
            
        # 매핑된 모델과 프롬프트 이름 생성
        display_models = [model_mapping.get(model, model) for model in model_order]
        display_prompts = [prompt_mapping.get(prompt, prompt) for prompt in prompt_order]
        
        logger.info(f"\n============\nGenerate Classification Table\n============\n")
        logger.info(f"Model order: {model_order}")
        logger.info(f"Display models: {display_models}")
        logger.info(f"Prompt order: {prompt_order}")
        logger.info(f"Display prompts: {display_prompts}")

        # 2. 데이터 리스트 초기화
        all_data = []
        plot_data = []
     
        #     # 3.2. CSV 파일 읽기
        for index, row in tqdm(df_info.iterrows()):
            file_classification_path = row['classification_report']
            
            model_name = row[col_name_model]
            prompt_name = row[col_name_prompt]
         
            df = pd.read_csv(file_classification_path)
            # 3.3.1. class 컬럼을 emotion으로 변경
            if 'class' in df.columns:
                df = df.rename(columns={'class': 'emotion'})
            
            if 'emotion' not in df.columns:
                logger.warning(f"No 'emotion' column found in {file_classification_path}")
                continue
            
            # 3.3.2. 매핑 적용
            display_model = model_mapping.get(model_name, model_name)
            display_prompt = prompt_mapping.get(prompt_name, prompt_name)
            
            # 3.3.3. 숫자형 컬럼 반올림
            #numeric_cols = df.select_dtypes(include=[np.number]).columns
  
            # 3.3.4. 'accuracy', 'macro avg', 'weighted avg' 행 제외
            class_rows = df[~df['emotion'].isin(['accuracy', 'weighted avg'])].copy()
            
            # 3.3.5. 데이터 구성
            row_data = {('Model', ''): display_model, ('Prompt', ''): display_prompt, ('original_model', ''): model_name}
            
            for _, row in class_rows.iterrows():
                emotion = row['emotion']
                try:
                    # 원본 값을 백분율로 변환 (반올림 없이)
                    p = float(row['precision']) * 100
                    r = float(row['recall']) * 100
                    f = float(row['f1-score']) * 100
                    #logger.info(f"Processing emotion {emotion}: P={p}, R={r}, F1={f}")
                except ValueError as e:
                    logger.error(f"Error converting values for emotion {emotion}: {str(e)}")
                    p = r = f = np.nan
                
                row_data[(emotion, 'P')] = p
                row_data[(emotion, 'R')] = r
                row_data[(emotion, 'F1')] = f
                
                # Plot 데이터 준비
                plot_data.append({
                    'Model': display_model,
                    'Prompt': display_prompt,
                    'Emotion': emotion,
                    'Metric': 'precision',
                    'Value': p
                })
                plot_data.append({
                    'Model': display_model,
                    'Prompt': display_prompt,
                    'Emotion': emotion,
                    'Metric': 'recall',
                    'Value': r
                })
                plot_data.append({
                    'Model': display_model,
                    'Prompt': display_prompt,
                    'Emotion': emotion,
                    'Metric': 'f1-score',
                    'Value': f
                })
            
            all_data.append(row_data)
            logger.info(f"Added data for {display_model} - {display_prompt}")
        print("================================================================")
        # 4. DataFrame 구성 및 정렬
        if all_data:
            final_df = pd.DataFrame(all_data)
            final_df.columns = pd.MultiIndex.from_tuples(final_df.columns)
            
            # 4.1. 정렬 적용
            if model_order and prompt_order:
                # 모델 순서 매핑
                model_map = {model: idx for idx, model in enumerate(model_order)}
                final_df['model_sort'] = final_df[('original_model', '')].map(model_map)
                
                # 프롬프트 순서 매핑
                prompt_map = {prompt_mapping.get(p, p) if prompt_mapping else p: idx 
                            for idx, p in enumerate(prompt_order)}
                final_df['prompt_sort'] = final_df[('Prompt', '')].map(prompt_map)
                
                # 정렬 적용 (level 파라미터 추가)
                final_df = final_df.sort_values(['model_sort', 'prompt_sort'])
                final_df = final_df.drop(['model_sort', 'prompt_sort'], axis=1, level=0)
                
            # 4.2. 원본 모델명 컬럼 제거
            final_df = final_df.drop(('original_model', ''), axis=1)
            final_df = final_df.reset_index(drop=True)

            model_order = [model_mapping.get(model, model) for model in model_order]
            prompt_order = [prompt_mapping.get(prompt, prompt) for prompt in prompt_order]

            
            mask = final_df[('Model', '')].isin(model_order)
            final_df = final_df[mask]
            
            mask = final_df[('Prompt', '')].isin(prompt_order)
            final_df = final_df[mask]
            # 4.3. 결과 저장
            output_name = output_path.name
            base_path = output_path.parent
            save_path = base_path / f"df_{output_name}.csv"
            final_df.to_csv(save_path)
            logger.info(f"Saved classification table to {save_path}")
            
            # 4.4. Plot 데이터프레임 생성
            plot_df = pd.DataFrame(plot_data)
            
            # 4.5. 이미지로 저장
            try:
                # 최종 테이블을 이미지 파일로 저장
                # Model과 Prompt 컬럼은 유지하고, 나머지 컬럼은 숫자형인 경우만 선택
                numeric_cols = []
                for col in final_df.columns:
                    if isinstance(col, tuple):
                        if col[0] not in ['Model', 'Prompt']:
                            # 메트릭 필터링 적용
                            if metric_filter == 'all' or col[1].upper() == metric_filter.upper():
                                if pd.api.types.is_numeric_dtype(final_df[col]):
                                    numeric_cols.append(col)
                    else:
                        if pd.api.types.is_numeric_dtype(final_df[col]):
                            numeric_cols.append(col)
                
                # Model, Prompt 컬럼과 숫자형 컬럼만 선택
                keep_cols = [col for col in final_df.columns if col[0] in ['Model', 'Prompt']] + numeric_cols
                final_df_filtered = final_df[keep_cols]
                
                if logger:
                    logger.info(f"Columns to be included in the image: {keep_cols}")
                    logger.info(f"Numeric columns: {numeric_cols}")
                    logger.info(f"Metric filter: {metric_filter}")
                
                # 스타일 설정 추가
                style_config = {
                    'font_size': '8pt',
                    'cell_padding': '2px',
                    'max_rows_per_page': 50,
                    'col_widths': {
                        'default': '150px',
                        'numeric': '80px',
                        'text': '200px'
                    }
                }
                
                save_table_as_image(final_df_filtered, base_path / f"fig_{output_name}.png", logger, round_digits, metric_filter)
            except Exception as e:
                logger.warning(f"Failed to save table image: {str(e)}")
                logger.warning(f"Error details: {str(e.__class__.__name__)}")
                logger.warning(f"Traceback:\n{traceback.format_exc()}")
            
            return final_df, plot_df
        else:
            logger.warning("No data to process")
            return pd.DataFrame(), pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error in gen_table: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return pd.DataFrame(), pd.DataFrame()

# ---------------------------
# 5. 분석 관련 함수들
# ---------------------------
# @safe_process_dataframe
def analyze_unknown_predictions(
    file_path: Union[str, Path],
    model_name: str,
    prompt_name: str,
    timestamp: str
) -> Optional[Dict]:
    """폴더 내 dataset CSV 파일에서 unknown 예측을 분석"""
    try:

        df = pd.read_csv(file_path)
        pred_cols = [col for col in df.columns if 'predicted_emotion' in col]
        if not pred_cols:
            logger.warning(f"No prediction column found in {file_path}")
     
        pred_col = pred_cols[0]
        total_samples = len(df)
        total_unknown = (df[pred_col] == 'unknown').sum()
        total_unknown_ratio = total_unknown / total_samples * 100
        
        class_unknown_ratios = {}
        for emotion in df['emotion'].unique():
            class_samples = df[df['emotion'] == emotion]
            class_unknown = (class_samples[pred_col] == 'unknown').sum()
            class_unknown_ratios[emotion] = class_unknown / len(class_samples) * 100
        
        unknown_stats = {
            'timestamp': timestamp,
            col_name_model: model_name,
            col_name_prompt: prompt_name,
            'total_samples': total_samples,
            'total_unknown': total_unknown,
            'total_unknown_ratio': total_unknown_ratio,
        }
        # 각 클래스별 unknown ratio 추가
        for emotion, ratio in class_unknown_ratios.items():
            unknown_stats[f'{emotion}_unknown_ratio'] = ratio
        return unknown_stats
    except Exception as e:
        logger.error(f"Error analyzing unknown predictions in {file_path}: {str(e)}")
    return None

# @safe_process_dataframe
def analyze_rag_search_results(
    file_path: Union[str, Path],
    model_name: str,
    prompt_name: str,
    timestamp: str
) -> tuple[Optional[Dict], List[Dict]]:
    """RAG 검색 결과 CSV 파일을 분석"""
    mismatched_samples = []
    if True:
        if True:
            try:
                df = pd.read_csv(file_path)
                if df is None:
                    logger.warning(f"No data found in {file_path}")
                    return None, []
                    
                total_samples = len(df)
                matches = (df['query_emotion'] == df['retrieved_emotion'])
                total_matches = matches.sum()
                match_ratio = (total_matches / total_samples) * 100
                mismatch_ratio = ((total_samples - total_matches) / total_samples) * 100
                
                class_stats = {}
                for emotion in df['query_emotion'].unique():
                    class_df = df[df['query_emotion'] == emotion]
                    class_matches = (class_df['query_emotion'] == class_df['retrieved_emotion']).sum()
                    class_match_ratio = (class_matches / len(class_df)) * 100
                    class_mismatch_ratio = ((len(class_df) - class_matches) / len(class_df)) * 100
                    class_stats[emotion] = {
                        'total': len(class_df),
                        'matches': class_matches,
                        'mismatches': len(class_df) - class_matches,
                        'match_ratio': class_match_ratio,
                        'mismatch_ratio': class_mismatch_ratio
                    }
                
                mismatched = df[~matches].copy()
                mismatched[col_name_model] = model_name
                mismatched[col_name_prompt] = prompt_name
                mismatched_samples.extend(mismatched.to_dict('records'))
                
                rag_stats = {
                    col_name_model: model_name,
                    col_name_prompt: prompt_name,
                    'total_samples': total_samples,
                    'total_matches': total_matches,
                    'total_mismatches': total_samples - total_matches,
                    'total_match_ratio': match_ratio,
                    'total_mismatch_ratio': mismatch_ratio,
                }
                for emotion, stats in class_stats.items():
                    rag_stats[f'{emotion}_match_ratio'] = stats['match_ratio']
                    rag_stats[f'{emotion}_mismatch_ratio'] = stats['mismatch_ratio']
                return rag_stats, mismatched_samples
            except Exception as e:
                logger.error(f"Error analyzing RAG search results in {file_path}: {str(e)}")
    return None, []

def find_matching_key(folder_name: str, mapping: Optional[Dict[str, str]] = None) -> Optional[Dict[str, str]]:
    """
    폴더 이름에서 매핑 딕셔너리에 있는 키와 매핑되는 값 찾기
    """

    matched_keys = [k for k in mapping if k in folder_name]

    if matched_keys:
        longest_keys = [k for k in matched_keys if len(k) == max(map(len, matched_keys))]
        if len(longest_keys) > 1:
            print(f"⚠️ 여러 개 발견: {longest_keys}")
        assigned_key, assigned_value = longest_keys[0], mapping[longest_keys[0]]
        print(f"Matched: {assigned_key} → {assigned_value}")
        return assigned_key, assigned_value
    else:
        print("No matching folder found.")
        return None, None

def parse_folder_name(folder_name: str, model_mapping: Optional[Dict[str, str]] = None, 
                     prompt_mapping: Optional[Dict[str, str]] = None) -> Optional[Dict[str, str]]:
    """
    폴더 이름에서 timestamp, model, prompt, embedding 정보 파싱
    
    Args:
        folder_name: 파싱할 폴더 이름 (예: "250302_140203_llama3.2_rag_prompt_ada002")
        model_mapping: 모델 이름 매핑 딕셔너리
        prompt_mapping: 프롬프트 타입 매핑 딕셔너리
    
    Returns:
        Dict[str, str]: 파싱된 정보를 담은 딕셔너리
    """
    try:
        # timestamp 파싱 (앞부분의 날짜_시간 형식)
        parts = folder_name.split('_')
        if len(parts) < 3:
            logger.warning(f"Invalid folder name format: {folder_name}")
            return None
            
        timestamp = f"{parts[0]}_{parts[1]}"
        rest = '_'.join(parts[2:])
        
        model_key, model_name = find_matching_key(rest, model_mapping)
        prompt_key, prompt_type = find_matching_key(rest, prompt_mapping)
        result = {
            'timestamp': timestamp,
            col_name_model: model_key,
            col_name_prompt: prompt_key,
            'original_name': folder_name
        }
        return result
        
    except Exception as e:
        logger.error(f"Error parsing folder name {folder_name}: {str(e)}")
        return None 
    

def find_csv_files_keys(directory, string_keys=['classification_report', 'dataset', 'rag_search_results']):
    """
    주어진 디렉토리에서 이름에 'B' 문자열이 포함된 CSV 파일을 찾습니다.
    """
    csv_files = {
        key: [] for key in string_keys
    }
    try:
        dir_path = Path(directory)
        # CSV 파일 중 이름에 b_string이 포함된 파일 찾기
        for key in string_keys:
            csv_files[key] = []
            for file_path in tqdm(dir_path.glob('*.csv'), desc="CSV 파일 검색 중"):
                if key in file_path.name and not '._' in file_path.name:
                    csv_files[key].append(str(file_path))
       
            if len(csv_files[key]) > 1:
                """주어진 디렉토리에서 특정 키워드가 포함된 CSV 파일을 찾습니다."""
                csv_files = {key: None for key in string_keys}
                
                try:
                    dir_path = Path(directory)
                    for key in string_keys:
                        matching_files = [
                            str(file_path) 
                            for file_path in dir_path.glob('*.csv') 
                            if key in file_path.name and not '._' in file_path.name
                        ]
                        
                        if matching_files:
                            if len(matching_files) > 1:
                                print(f"=====⚠️ 여러 개 발견: {matching_files}")
                            csv_files[key] = matching_files[0]
                        else:
                            print(f"Warning: No CSV file found for key '{key}' in {directory}")
                            
                    return csv_files
            
                except Exception as e:
                    print(f"CSV 파일 검색 중 오류 발생: {directory}, 오류: {e}")
                    return csv_files
           
            csv_files[key] = csv_files[key][0]
        return csv_files

    except Exception as e:
        print(f"CSV 파일이 없습니다.: {directory}, 오류: {e}")
    
    return csv_files


def process_folder(folder_path, MODEL_MAPPING, PROMPT_MAPPING, string_keys=['classification_report', 'dataset', 'rag_search_results']):
    """
    하위 폴더를 처리하고 필요한 정보를 추출합니다.
    """
    csv_files = find_csv_files_keys(folder_path, string_keys)
    folder_name = os.path.basename(folder_path)
    parsed_info = parse_folder_name(folder_name, MODEL_MAPPING, PROMPT_MAPPING)
    results ={'timestamp': None,
            col_name_model: None,
            col_name_prompt: None}
    for key in string_keys:
        if csv_files[key]:
            results[key] = csv_files[key]
  
    if parsed_info is None:
        print(f"⚠️ 폴더명 파싱 실패: {folder_name}")
        return results
    else:
        results['timestamp'] = parsed_info['timestamp']
        results[col_name_model] = parsed_info[col_name_model]
        results[col_name_prompt] = parsed_info[col_name_prompt]
    return results
def call_prep_results(base_dir, path_output, MODEL_MAPPING, PROMPT_MAPPING, string_keys=['classification_report', 'dataset', 'rag_search_results']):
    
    folder_info = [process_folder(Path(base_dir) / folder, MODEL_MAPPING, PROMPT_MAPPING, string_keys=string_keys) for folder in tqdm(os.listdir(base_dir), desc="폴더 처리 중")]
    df_folder = pd.DataFrame(folder_info)
    df_folder.sort_values(by=[col_name_model, col_name_prompt, 'timestamp'], inplace=True)
    df_folder.to_csv(path_output, index=False)
    
    filtered_prompt = sorted([x for x in list(df_folder[col_name_prompt].unique()) if x is not None])
    filtered_model = sorted([x for x in list(df_folder[col_name_model].unique()) if x is not None])
    json_exp = {'stats': 
                {'num_models':len(filtered_model), 
                 'num_prompts':len(filtered_prompt)}
                , 'models':filtered_model, 'prompts':filtered_prompt}

    with open(path_output.parent / "experiment_info.json", "w") as f:
        json.dump(json_exp, f, indent=4)

    logger.info(f"Folder info saved to {path_output}")
    return df_folder

def filter_df(df, model_order, prompt_order):
    """DataFrame을 model_order와 prompt_order에 따라 필터링"""
    # model_order가 문자열인 경우 리스트로 변환
    if isinstance(model_order, str):
        model_order = [model_order]
    # prompt_order가 문자열인 경우 리스트로 변환
    if isinstance(prompt_order, str):
        prompt_order = [prompt_order]
        
    if model_order:
        df = df[df[col_name_model].isin(model_order)]
    if prompt_order:
        df = df[df[col_name_prompt].isin(prompt_order)]
    return df

# 6. 결과 준비 및 폴더 순회 (prep_results)
# ---------------------------
def prep_results(
    df_info: pd.DataFrame,
    path_output: Union[str, Path],
    prep_file_output: str,
    model_order: Optional[List[str]] = None,
    prompt_order: Optional[List[str]] = None,
    MODEL_MAPPING: Optional[Dict[str, str]] = None,
    PROMPT_MAPPING: Optional[Dict[str, str]] = None,
    cfg: Optional[Dict[str, str]] = None,
) -> None:
    """지정된 base_dir 내의 폴더들을 순회하며 분석 수행"""
    results = []
    unknown_results = []
    rag_results = []
    all_mismatches = []
   
    for index, row in tqdm(df_info.iterrows(), total=len(df_info), desc="Processing rows"):
        file_path = row['dataset']
        file_rag_path = row['rag_search_results']
        file_classification_path = row['classification_report']
        
        model_name = row[col_name_model]
        prompt_name = row[col_name_prompt]
        # embedding_name = row['embedding']
        timestamp = row['timestamp']

        # Unknown 분석
        unknown_stats = analyze_unknown_predictions(file_path=file_path, model_name=model_name, prompt_name=prompt_name, timestamp=timestamp)#, embedding_name=embedding_name)
        if unknown_stats:
            unknown_results.append(unknown_stats)
        
        # RAG 분석 (rag가 포함된 프롬프트만)
        if prompt_name and 'rag' in prompt_name.lower():
            rag_stats, mismatches = analyze_rag_search_results(file_path=file_rag_path, model_name=model_name, prompt_name=prompt_name, timestamp=timestamp)
            if rag_stats:
                rag_results.append(rag_stats)
                all_mismatches.extend(mismatches)
        try:
            df = pd.read_csv(file_classification_path)
            
            if 'class' in df.columns:
                macro_row = df[df['class'] == 'macro avg']
            else:
                df.set_index(df.columns[0], inplace=True)
                macro_row = df.loc['macro avg'] if 'macro avg' in df.index else None
            
            if macro_row is not None:
                if isinstance(macro_row, pd.Series):
                    precision = macro_row.get('precision', None)
                    recall = macro_row.get('recall', None)
                    f1_score = macro_row.get('f1-score', None)
                else:
                    precision = macro_row['precision'].values[0]
                    recall = macro_row['recall'].values[0]
                    f1_score = macro_row['f1-score'].values[0]
                
                result = {
                    col_name_model: model_name,
                    col_name_prompt: prompt_name,
                    # 'embedding name': embedding_name,
                    'precision': precision,
                    'recall': recall,
                    'f1-score': f1_score
                }
                logger.info(f"Added result: {result}")
                results.append(result)
            else:
                logger.warning(f"Warning: 'macro avg' row not found in {file_classification_path}")
        except Exception as e:
            logger.error(f"Error processing file ({file_classification_path}): {str(e)}")



    # 결과 저장 및 시각화
    if results:
        df_bar = filter_df(pd.DataFrame(results), model_order, prompt_order)
        output_path = Path(path_output) / prep_file_output
        df_bar.to_csv(output_path, index=False)
        logger.info(f"Saved results to {output_path}")
    

    if unknown_results:
        df_unknown = filter_df(pd.DataFrame(unknown_results), model_order, prompt_order)
        if len(df_unknown) > 0:
            sort_columns = []
            if model_order:
                df_unknown['model_order'] = df_unknown[col_name_model].map({m: i for i, m in enumerate(model_order)})
                sort_columns.append('model_order')
            if prompt_order:
                df_unknown['prompt_order'] = df_unknown[col_name_prompt].map({p: i for i, p in enumerate(prompt_order)})
                sort_columns.append('prompt_order')
            sort_columns.append('timestamp')
            
            df_unknown = df_unknown.sort_values(sort_columns)
            if 'model_order' in df_unknown.columns:
                df_unknown = df_unknown.drop('model_order', axis=1)
            if 'prompt_order' in df_unknown.columns:
                df_unknown = df_unknown.drop('prompt_order', axis=1)
            
            unknown_csv_path = Path(path_output) / "df_unknown.csv"
            df_unknown.to_csv(unknown_csv_path, index=False)
            logger.info(f"Unknown analysis saved to {unknown_csv_path}")
            
            if len(df_unknown) > 0:
                plot_unknown_analysis(df_unknown, Path(path_output) / "unknown_analysis",
                                  model_mapping=MODEL_MAPPING, prompt_mapping=PROMPT_MAPPING,
                                  model_order=model_order, prompt_order=prompt_order, cfg=cfg)
    
    if rag_results:
        df_rag = filter_df(pd.DataFrame(rag_results), model_order, prompt_order)
        if len(df_rag) > 0:
            sort_columns = []
            if model_order:
                df_rag['model_order'] = df_rag[col_name_model].map({m: i for i, m in enumerate(model_order)})
                sort_columns.append('model_order')
            if prompt_order:
                df_rag['prompt_order'] = df_rag[col_name_prompt].map({p: i for i, p in enumerate(prompt_order)})
                sort_columns.append('prompt_order')
            df_rag = df_rag.sort_values(sort_columns)
            if 'model_order' in df_rag.columns:
                df_rag = df_rag.drop('model_order', axis=1)
            if 'prompt_order' in df_rag.columns:
                df_rag = df_rag.drop('prompt_order', axis=1)
            
            rag_csv_path = Path(path_output) / "df_rag_search_analysis.csv"
            df_rag.to_csv(rag_csv_path, index=False)
            logger.info(f"RAG search analysis saved to {rag_csv_path}")
            
            if all_mismatches:
                df_mismatches = pd.DataFrame(all_mismatches)
                mismatches_path = Path(path_output) / "df_rag_mismatched_samples.csv"
                df_mismatches.to_csv(mismatches_path, index=False)
                logger.info(f"RAG mismatched samples saved to {mismatches_path}")
            
            if len(df_rag) > 0:
                plot_rag_search_analysis(df_rag, Path(path_output) / "rag_search_analysis",
                                     model_mapping=MODEL_MAPPING, prompt_mapping=PROMPT_MAPPING,
                                     model_order=model_order, prompt_order=prompt_order)
