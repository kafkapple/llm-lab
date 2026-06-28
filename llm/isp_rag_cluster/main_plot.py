#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from pathlib import Path
import logging
import hydra
import pandas as pd
from src.visualization.plot import (
    plot_grouped_bar_chart, 
    plot_class_metrics_comparison,
)
from src.visualization.prep_result import prep_results, gen_table, call_prep_results

# 로깅 설정
def setup_logging():
    """로깅 설정"""
    # 로그 디렉토리 생성
    log_dir = Path('./logs')
    log_dir.mkdir(exist_ok=True)
    
    # 로거 설정
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # 파일 핸들러 설정
    file_handler = logging.FileHandler(log_dir / 'plot.log', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 콘솔 핸들러 설정
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 포맷터 설정
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 핸들러 추가
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 로거 초기화
logger = setup_logging()

@hydra.main(config_path="config", config_name="llm", version_base="1.2")
def main(cfg):
    """메인 실행 함수"""
    # mapping이 None인 경우 빈 딕셔너리로 초기화
    if not hasattr(cfg, 'mapping'):
        logger.warning("cfg.mapping is not found, initializing with empty dictionary")
        cfg.mapping = {}
    
    MODEL_MAPPING = getattr(cfg.mapping, 'model_mapping', {})
    PROMPT_MAPPING = getattr(cfg.mapping, 'prompt_mapping', {})
    
    # 매핑 상태 로깅
    logger.info("=== Mapping Status ===")
    logger.info(f"Model Mapping: {MODEL_MAPPING}")
    logger.info(f"Prompt Mapping: {PROMPT_MAPPING}")
    
    target= 'isear' #'goemotions' #'goemotions_sampling' #
    try:
        # 1. 기본 설정
        logger.info("Starting visualization process...")
        # 2. 경로 설정
        PATH_OUTPUT = Path(f'./plots/{target}')
        PATH_TARGET = Path(f"./results/{target}")
        os.makedirs(PATH_OUTPUT, exist_ok=True)
        
        # 3. 출력 파일명 설정
        prep_file_output = "df_table_metric_average.csv"
        table_output = "table_metric_class-wise"
        bar_output = "fig_bar_metric_avg"
        confusion_output = "fig_bar_metric_class-wise"
        
        PATH_INFO = PATH_OUTPUT / "metadata.csv"
        load_data = False
        if os.path.exists(PATH_INFO) and load_data:
            logger.info(f"Folder info already exists at {PATH_INFO}")
            df_info = pd.read_csv(PATH_INFO)
        else:
            logger.info("Generating new metadata...")
            df_info = call_prep_results(PATH_TARGET, PATH_INFO, MODEL_MAPPING, PROMPT_MAPPING,
                                    string_keys=['classification_report', 'dataset', 'rag_search_results'])
            logger.info(f"Metadata shape: {df_info.shape}")
            logger.info(f"Available models: {df_info['model'].unique()}")
            logger.info(f"Available prompts: {df_info['prompt'].unique()}")
        
        #########################################################
        # 4. 실험 설정
        prompt_main = ['baseline_prompt', 'zero_shot_prompt', 'few_shot_prompt']
        prompt_enhanced = ['custom_prompt','custom_enhanced_prompt','few_shot_enhanced_prompt'] #baseline_cot_prompt, custom_enhanced_cot_prompt
        prompt_persona = ['negative_persona_prompt', 'no_persona_prompt']
        prompt_persona_enhanced = ['persona_a_prompt', 'persona_b_prompt']
        prompt_embedding = ['rag_prompt_j-hartmann_emotion-english-distilroberta-base',
                            'rag_prompt_baai_bge-m3',
                          'rag_prompt_bhadresh-savani_distilbert-base-uncased-emotion', 
                          'rag_prompt_sentence-transformers_all-minilm-l6-v2'
                          ]
        prompt_cheat = ['rag_prompt_baai_bge-m3_cheat']

        prompt_cot = ['baseline_cot_prompt', 'custom_enhanced_cot_prompt']
        
        # 실험 설정 로깅
        logger.info("=== Experiment Settings ===")
        logger.info(f"Main prompts: {prompt_main}")
        logger.info(f"Enhanced prompts: {prompt_enhanced}")
        logger.info(f"Persona prompts: {prompt_persona}")
        logger.info(f"Embedding prompts: {prompt_embedding}")
        
        # 모델 설정
        model_commercial = ['gpt-4o', 'claude-3-5-sonnet-20241022']
        model_qwen = ['qwen2.5_3b','qwen2.5', 'qwen2.5_14b']
        model_gemma = ['gemma3_1b', 'gemma3', 'gemma3_12b']
        model_llama = ['llama3.2', 'llama3.1', 'llama2_13b']
        model_7b = ['llama3.1', 'qwen2.5']
        model_other_open_source = ['mistral', 'hf.co_tensorblock_emotion-llama-gguf']
        
        # 모델 설정 로깅
        logger.info("=== Model Settings ===")
        logger.info(f"Commercial models: {model_commercial}")
        logger.info(f"Qwen models: {model_qwen}")
        logger.info(f"Gemma models: {model_gemma}")
        logger.info(f"Llama models: {model_llama}")
        logger.info(f"7B models: {model_7b}")
        logger.info(f"Other open source models: {model_other_open_source}")
        
        model_open_source = model_qwen + model_gemma + ['gemma']+ model_llama + model_other_open_source
        if target == "isear":
            dict_exp = {
                # '1_prompt_matters': {
                #     'model_order': model_7b + [model_other_open_source[0]],
                #     'prompt_order': [prompt_persona[1]] + prompt_main[1:]
                # },
                '1-2_prompt_matters_detailed': {
                    'model_order': model_7b + [model_other_open_source[0]],
                    'prompt_order': prompt_persona + prompt_main + [prompt_embedding[1]]#, prompt_embedding[-1]]
                },
                # '2_size_matters': {
                #     'model_order': model_gemma + model_qwen,
                #     'prompt_order': prompt_persona + prompt_main + [prompt_embedding[1]] #,prompt_enhanced[0]
                # },
                # '3_commercial_models': {
                #     'model_order':  model_commercial, #['qwen2.5_14b', 'gemma3_12b'] +
                #     'prompt_order': prompt_main + [prompt_embedding[1]]#,prompt_enhanced[0]]
                # },
                # '3_2_all_models': {
                #     'model_order':  model_gemma + model_qwen +model_commercial, #['qwen2.5_14b', 'gemma3_12b'] +
                #     'prompt_order': prompt_persona + prompt_main + [prompt_embedding[1]]#,prompt_enhanced[0]]
                # },
                # 'test': {
                #     'model_order':    model_commercial, #['qwen2.5_14b', 'gemma3_12b'] +['gemma3_1b','qwen2.5_3b', 'gemma3', 'qwen2.5','gemma3_12b', 'qwen2.5_14b' ]+
                #     'prompt_order': prompt_persona +  [prompt_main[1],prompt_embedding[1]]#,prompt_enhanced[0]]
                # },
                # '4_sota': {
                #     'model_order':  model_commercial, #['qwen2.5_14b', 'gemma3_12b'] +
                #     'prompt_order': prompt_persona +  [prompt_main[1],prompt_embedding[1]]#,prompt_enhanced[0]]
                # },
                # '4_2_worst': {
                #     'model_order':  ['gemma3:1b', 'qwen2.5:3b'], 
                #     'prompt_order': [prompt_persona[0]] +  prompt_main[0:2]
                # },
                # '4_emo-llama': {
                #     'model_order':  model_7b + model_other_open_source,
                #     'prompt_order': [prompt_persona[-1]] +prompt_cheat
                # },
                # 'supple_1_special': {
                #     'model_order':  [model_gemma[1], model_qwen[1], model_other_open_source[-1]],
                #     'prompt_order':  [prompt_persona[1]] + [prompt_embedding[1]] + prompt_cheat
                # },
                # 'supple_2_size_matters_detailed': {
                #     'model_order': model_gemma + model_qwen,
                #     'prompt_order': prompt_persona + prompt_main + prompt_embedding[0:2] + prompt_enhanced + prompt_cot #[prompt_enhanced[0]]
                # },
                # # 'supple_3_detailed': {
                # #     'model_order': [model_gemma[0], model_gemma[-1], model_qwen[-1]], 
                # #     'prompt_order': prompt_persona + prompt_persona_enhanced + prompt_embedding[0:2] + prompt_enhanced
                # # },
                # 'supple_3-1_all_persona': {
                #     'model_order':  model_open_source,
                #     'prompt_order': prompt_persona + prompt_persona_enhanced
                # },
    
                # 'supple_3-2_all_enhanced_embedding': {
                #     'model_order':  model_open_source + model_commercial,
                #     'prompt_order': prompt_enhanced + prompt_embedding[0:2] 
                # }
            }
        elif target == "goemotions":
            dict_exp = {
                '1_prompt_matters': {
                    'model_order': [model_qwen[0]],
                    'prompt_order': prompt_persona + prompt_main + [prompt_embedding[0]]  #, prompt_embedding[-1]]
                }
            }
        elif target == "goemotions_sampling":
            dict_exp = {
                '1_prompt_matters': {
                    'model_order': ['gemma'],
                    'prompt_order': [ prompt_main[-1], prompt_embedding[0]] #, prompt_embedding[-1]]
                }
            }
        
        # 실험 설정 로깅
        logger.info("=== Experiment Configurations ===")
        for exp_name, exp_config in dict_exp.items():
            if exp_name == '1_prompt_matters':
                cfg.plotting.font.value_size = 7
            logger.info(f"\nExperiment: {exp_name}")
            logger.info(f"Model order: {exp_config['model_order']}")
            logger.info(f"Prompt order: {exp_config['prompt_order']}")
            
            # 매핑된 이름 확인
            mapped_models = [MODEL_MAPPING.get(m, m) for m in exp_config['model_order']]
            mapped_prompts = [PROMPT_MAPPING.get(p, p) for p in exp_config['prompt_order']]
            logger.info(f"Mapped models: {mapped_models}")
            logger.info(f"Mapped prompts: {mapped_prompts}")
            
            path_output = PATH_OUTPUT/exp_name
            path_table = path_output / table_output
            os.makedirs(path_output, exist_ok=True)
            
            # 5.1. 데이터 준비 및 테이블 생성
            logger.info(f"\nProcessing {exp_name}...")
            prep_results(
                df_info=df_info,
                path_output=path_output,
                prep_file_output=prep_file_output,
                model_order=exp_config['model_order'],
                prompt_order=exp_config['prompt_order'],
                MODEL_MAPPING=MODEL_MAPPING,
                PROMPT_MAPPING=PROMPT_MAPPING,
                cfg=cfg
            )
            
            # 5.2. 테이블 생성 및 저장
            try:
                logger.info("Generating table...")
                final_df, plot_df = gen_table(
                    df_info=df_info,
                    output_path=path_table,
                    round_digits=1,
                    model_mapping=MODEL_MAPPING,
                    prompt_mapping=PROMPT_MAPPING,
                    model_order=exp_config['model_order'],
                    prompt_order=exp_config['prompt_order'],
                    cfg=cfg
                )
                
                # Multi-index 컬럼 처리
                if isinstance(final_df.columns, pd.MultiIndex):
                    logger.info("Converting MultiIndex columns to single level...")
                    final_df.columns = final_df.columns.get_level_values(-1)
                
                # 결과 저장
                logger.info(f"Saving table to {path_table}.csv")
                final_df.to_csv(f"{path_table}.csv", index=False)
                
            except Exception as e:
                logger.error(f"Error in table generation: {str(e)}")
                logger.warning("Skipping table generation and continuing with visualization...")
                final_df, plot_df = None, None
            
            # 5.3. 그룹 바 차트 생성
            logger.info("Generating grouped bar chart...")
            plot_grouped_bar_chart(
                df_info=df_info,
                csv_path=Path(path_output) / prep_file_output,
                save_path=path_output / bar_output,
                model_mapping=MODEL_MAPPING,
                prompt_mapping=PROMPT_MAPPING,
                model_order=exp_config['model_order'],
                prompt_order=exp_config['prompt_order'],
                cfg=cfg 
            )
            
            # 5.4. 클래스 메트릭 비교 그래프
            if plot_df is not None:
                logger.info("Generating class metrics comparison plot...")
                # 매핑된 모델과 프롬프트 순서 생성
                mapped_model_order = [MODEL_MAPPING.get(m, m) for m in exp_config['model_order']]
                mapped_prompt_order = [PROMPT_MAPPING.get(p, p) for p in exp_config['prompt_order']]

                plot_class_metrics_comparison(
                    df=plot_df,
                    save_path=f"{path_output}/{confusion_output}",
                    model_mapping=MODEL_MAPPING,
                    prompt_mapping=PROMPT_MAPPING,
                    model_order=mapped_model_order,
                    prompt_order=mapped_prompt_order,
                    cfg=cfg
                )
            
            logger.info(f"{exp_name} - All processes completed.")
        
        logger.info("All visualization processes completed successfully.")
   
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main()
