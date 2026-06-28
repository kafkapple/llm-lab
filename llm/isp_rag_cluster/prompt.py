from src.util import enable_logging
from dotenv import load_dotenv
import os
import hydra
load_dotenv()
import pandas as pd
import re

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

########################################################
rag_guideline = """
    ## Retrieved Context:
        Consider the following similar texts and their emotion classifications:
        
"""
persona_enhanced = """
    ## Persona Module
    Assume the role of a seasoned psychologist and emotion analysis expert. You have extensive experience distinguishing subtle emotional nuances, particularly among negative emotions such as guilt, shame, and disgust. Approach every analysis with a critical, inquisitive, and detail-oriented mindset.
"""
######################################################## 

def get_prompt(cfg, prompt_name, text, examples):
    instruction = cfg.prompt.modules.instruction
    instruction_only = cfg.prompt.modules.instruction_only
    instruction_negative = cfg.prompt.modules.instruction_negative

    definition = cfg.prompt.modules.definition
    definition_enhanced = cfg.prompt.modules.definition_enhanced
    definition_specific = cfg.prompt.modules.definition_specific

    output_format = cfg.prompt.modules.output_format
    few_shot_examples = cfg.prompt.modules.few_shot_examples
    few_shot_enhanced = cfg.prompt.modules.few_shot_enhanced
#########################
    baseline_prompt = instruction + output_format
    no_persona_prompt = instruction_only + output_format
    negative_persona_prompt = instruction_negative + output_format

    zero_shot_prompt = instruction + definition + output_format
    few_shot_prompt = instruction + definition + few_shot_examples + output_format
    few_shot_enhanced_prompt = instruction + definition + few_shot_enhanced + output_format

    custom_prompt = instruction +  definition + definition_specific + output_format
    custom_enhanced_prompt = instruction +  definition_enhanced + definition_specific + output_format

    rag_prompt = instruction + definition + rag_guideline + f"{examples}" + output_format

    # persona_a = instruction_a + role_a + persona_a + definition_enhanced + output_format
    # persona_b = instruction_b + role_b + persona_b + definition_enhanced + output_format
    baseline_cot_prompt = baseline_prompt + "Let's think step by step."
    custom_enhanced_cot_prompt = custom_enhanced_prompt + "Let's think step by step."


    prompts = {'baseline': baseline_prompt,
                'no_persona': no_persona_prompt, 'negative_persona': negative_persona_prompt, 
                # 'persona_a': persona_a, "persona_b": persona_b, 
                'zero_shot': zero_shot_prompt, 'few_shot': few_shot_prompt, 
               'custom': custom_prompt, 'custom_enhanced': custom_enhanced_prompt,
               'rag': rag_prompt, 'rag_prompt_baai_bge-m3': rag_prompt,
               'few_shot_enhanced': few_shot_enhanced_prompt,
               'baseline_cot': baseline_cot_prompt, 'custom_enhanced_cot': custom_enhanced_cot_prompt,
              }
    

    prompt = prompts[prompt_name] + f"\n\nText: {text}" 
    prompt_name = cfg.mapping.prompt_mapping.get(prompt_name+'_prompt')

    return prompt, prompt_name

def invoke(cfg, prompt_name, text="", examples=""):
    prompt, prompt_name = get_prompt(cfg, prompt_name, text, examples)
    return prompt, prompt_name


def gen_prompt_log(cfg):
    print(os.getcwd())
    outputs_path = "outputs/logs"
    if not os.path.exists(outputs_path):
        os.makedirs(outputs_path)
    enable_logging(f"{outputs_path}/print_log.txt")
    # from src.util import count_tokens

    # Example usage for a Llama-based model:
    text = "Hi there! This is input text query :)" #"Hello, this module calculates token counts for different models."
    examples = """
            Example 1:
            Text: when i fell in love.
            Emotion: joy
            Confidence: 0.75

            Example 2:
            Text: being in love for the first time.
            Emotion: joy
            Confidence: 0.74
            """
    examples = ""
    prompts = ['baseline', 'no_persona', 'negative_persona', 
               'zero_shot', 'few_shot', 
               'custom', 'rag', 'rag_prompt_baai_bge-m3',
               'custom_enhanced', 
               'few_shot_enhanced',
               'baseline_cot', 'custom_enhanced_cot'
              ]
  
    models = ["google/gemma-3-4b-it"] #["Qwen/Qwen2.5-7B"]#, "google/gemma-3-4b-it"]#, "meta-llama/Llama-3.2-3B-Instruct"]# "meta-llama/Llama-3.2-3B-Instruct",
    # 모델의 토크나이저 로드 (예: google/gemma-3-4b-it)
    from transformers import AutoTokenizer
    import pandas as pd
    
    # 예시: 프롬프트 토큰 수 계산
   
    results = []
    print("\n<Prompts for LLM based emotion recognition>")
    for model in models:
        tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=HUGGINGFACE_TOKEN, use_fast=False)
        print(f"\n-Model: <{model}>")
        for idx, prompt in enumerate(prompts):
            prompt_text, prompt_name = invoke(cfg, prompt, text, examples)
            tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
            token_count = len(tokens)
            
            print(f"\n{idx+1}. Prompt name: <{prompt_name}>")
            print("="*100)
            print(f"{prompt_text}:")
            print("="*10)
            print(f"Token count: {token_count}")
            print("="*100)
            if prompt == 'rag_prompt_baai_bge-m3':
                prompt = 'rag_prompt_baai_bge-m3'
            else:
                prompt = prompt+'_prompt'
            results.append({"model": model, "prompt": prompt, "token_count": token_count})

    pd.DataFrame(results).to_csv(f"{outputs_path}/token_counts.csv", index=False)



def extract_model_size(model_name: str) -> int:
    """
    주어진 model name에서 공백으로 split 후 마지막 토큰에서 숫자만 추출.
    예) "Model A 123" -> 123
    """
    try:
        # 공백 기준으로 분리 후 마지막 토큰 선택
        last_token = model_name.split()[-1]
        # 정규 표현식을 사용하여 숫자 추출
        match = re.search(r'\d+', last_token)
        if match:
            return int(match.group())
        else:
            return None
    except Exception as e:
        print(f"Error extracting model size from '{model_name}': {e}")
        return None
import math
def compute_efficiency(macro_f1, token_count, c=1):
    """
    효율성 지표 계산 함수.
    Efficiency = macro F1 score / (log(token_count + 1) + c)
    
    Parameters:
        macro_f1 (float): Emotion recognition의 macro F1 score
        token_count (int): 사용된 token 수
        c (float): 보정 상수 (default: 1)
        
    Returns:
        float: 계산된 효율성 지표
    """
    return macro_f1 / (math.log(token_count + 1) + c)

import numpy as np
def calc_metric_effeciency(cfg, df_path, out_path):
    # 지정된 폴더 경로 (실제 경로로 수정)
    # 1. df_table_metric_average.csv 불러오기
    # 보정 상수 설정 (예: c=1)
    c = 10
    model_mapping = cfg.mapping.model_mapping
    prompt_mapping = cfg.mapping.prompt_mapping
    csv_file = os.path.join(df_path, "df_table_metric_average.csv")
    df = pd.read_csv(csv_file)
    
    # 2. model 및 "prompt 2" 컬럼의 값을 매핑 딕셔너리(model_mapping, prompt_mapping)를 사용해 치환
    df['model'] = df['model'].map(model_mapping).fillna(df['model'])
    df['prompt'] = df['prompt'].map(prompt_mapping).fillna(df['prompt'])
    
    # 3. 치환된 model name에서 모델 사이즈 추출 후 "model size" 컬럼에 저장
    df['model size'] = df['model'].apply(lambda x: extract_model_size(x))
    
    # 4. 현재 모델, 프롬프트 및 평가 지표(precision, recall, f1-score) 출력
    print("현재 정보:")
    print(df[['model', 'prompt', 'precision', 'recall', 'f1-score']])
    
    # 5. token_counts.csv 불러오기
    token_csv = os.path.join(out_path, "token_counts.csv")
    df_token = pd.read_csv(token_csv)
    
    # 6. token_counts의 prompt 컬럼도 prompt_mapping을 적용해 치환
    df_token['prompt'] = df_token['prompt'].map(prompt_mapping).fillna(df_token['prompt'])
    
    # 7. df와 df_token 병합 (df의 "prompt 2"와 df_token의 "prompt" 기준)
    #    df_token에 token_count 컬럼이 존재한다고 가정합니다.
    df = df.merge(df_token[['prompt', 'token_count']], left_on='prompt', right_on='prompt', how='left')
    # 불필요한 병합 키 제거
    #df.drop(columns=['prompt'], inplace=True)
    # 병합된 token_count 컬럼명을 "token count"로 변경
    w_model = df['model size'].max()- df['model size'].min()
    w_token = df['token_count'].max() - df['token_count'].min()
    print(f"w_model: %s" % w_model)
    print(f"w_token: %s" % w_token)

    # # 8. 모델 사이즈와 토큰 카운트를 0-1 사이 값으로 정규화
   
    df['model_size_norm']  = (np.log(df['model size'] + 1) + c)
    df['token_size_norm'] = (np.log(df['token_count'] + 1) + c)

    # 8. 평가 지표(precision, recall, f1-score)를 모델 사이즈 및 토큰 카운트로 나눈 값 계산
    metrics = ['precision', 'recall', 'f1-score']
    for metric in metrics:
        df[f"{metric} / model size"] = df[metric] / df['model_size_norm']
        df[f"{metric} / token count"] = df[metric] / df['token_size_norm']

    # 9. 최종 데이터프레임을 CSV 파일로 저장
    df.sort_values(by='f1-score / token count', inplace=True)
    output_file = os.path.join(out_path, "df_processed.csv")
    df.to_csv(output_file, index=False)
    print(f"처리된 데이터프레임이 {output_file}에 저장되었습니다.")

@hydra.main(version_base="1.2", config_path='config', config_name='llm')
def main(cfg):
    gen_prompt_log(cfg)
    
    df_path = 'plots/isear/2_size_matters'
    out_path = 'outputs/logs/'
    calc_metric_effeciency(cfg, df_path, out_path)


if __name__ == "__main__":
    main()