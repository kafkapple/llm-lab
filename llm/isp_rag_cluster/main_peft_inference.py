import hydra
from pathlib import Path
import pandas as pd
import torch
from src.inference import load_emotion_classifier
from src.eval import save_metrics
from src.data_prep import load_data
from src.util import sanitize_model_name, get_output_dir_name
from tqdm import tqdm
from sklearn.model_selection import train_test_split

@hydra.main(version_base="1.2", config_path='config', config_name='llm')
def main(cfg):
    # 모델 설정
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    model_name = model_id.split("/")[-1]
    base_model_path = model_id

    # LoRA 모델 경로
    lora_path = "outputs/lora/20250317_215011_qwen2.5-7b-instruct_none/final_model"

    # 출력 디렉토리 설정
    output_dir_name = get_output_dir_name(sanitize_model_name(model_name), cfg)
    output_dir = Path('outputs') / 'lora_inference' / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 데이터 로드
    df, labels = load_data(cfg)
    
    # emotion 컬럼을 문자열로 변환
    df['emotion'] = df['emotion'].astype(str)
    
    # 데이터 분할 (6:2:2 비율로 수정)
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.4,
        random_state=cfg.general.seed, 
        stratify=df["emotion"]
    )
    
    # train과 val 분리 (사용하지는 않지만, 학습 시와 동일한 분할 유지)
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5, 
        random_state=cfg.general.seed, 
        stratify=temp_df["emotion"]
    )
    
    print(f"Total samples: {len(df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Test set class distribution:")
    print(test_df['emotion'].value_counts())
    
    # 모델 로드
    classifier = load_emotion_classifier(
        base_model_path=base_model_path,
        lora_path=lora_path
    )

    # 결과 저장을 위한 데이터프레임 준비
    results = []

    # 테스트 세트에 대해서만 배치 처리
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating test set"):
        text = row['text']
        true_emotion = row['emotion']
        
        # 예측 수행
        prediction = classifier.predict(text)
        
        # 결과 저장
        result = {
            'text': text,
            'emotion': true_emotion,
            f'predicted_emotion_{model_name}': prediction['predicted_class'],
            f'confidence_score_{model_name}': max(prediction['probabilities'].values()),
        }
        
        # 각 클래스별 확률 추가
        for emotion, prob in prediction['probabilities'].items():
            result[f'prob_{emotion}'] = prob
            
        results.append(result)

    # 결과를 데이터프레임으로 변환
    results_df = pd.DataFrame(results)
    
    print("\nBefore mapping:")
    print(results_df.info())
    print("\nEmotion unique values:", results_df['emotion'].unique())
    
    # cfg에서 레이블 리스트 가져오기
    label_names = list(cfg.data.datasets[cfg.data.name].labels)
    
    # 숫자 인덱스를 실제 레이블로 매핑
    index_to_label = {i: label for i, label in enumerate(label_names, 1)}  # 1부터 시작하는 인덱스
    
    # emotion과 predicted_emotion 컬럼을 레이블로 매핑
    results_df['emotion'] = results_df['emotion'].astype(int).map(index_to_label)
    results_df[f'predicted_emotion_{model_name}'] = results_df[f'predicted_emotion_{model_name}'].astype(int).map(index_to_label)
    
    print("\nAfter mapping:")
    print(results_df.info())
    print("\nEmotion unique values:", results_df['emotion'].unique())
    print(f"Predicted emotion unique values:", results_df[f'predicted_emotion_{model_name}'].unique())

    # prob_ 컬럼 이름도 숫자 대신 감정 레이블 사용
    for i, label in enumerate(label_names, 1):
        if f'prob_{i}' in results_df.columns:
            results_df[f'prob_{label}'] = results_df[f'prob_{i}']
            results_df.drop(columns=[f'prob_{i}'], inplace=True)

    # 결과 파일 저장
    output_file = output_dir / f'{cfg.data.name}_model-{model_name}_test.csv'
    results_df.to_csv(output_file, index=False)
    
    print("\nSaved file preview:")
    print(results_df.head())
    print("\nSaved columns:", results_df.columns.tolist())

    # eval.py의 save_metrics 함수를 사용하여 평가 지표 계산 및 저장
    try:
        metrics_result, cm = save_metrics(output_file, cfg)
    except Exception as e:
        print("\nError in save_metrics:")
        print(f"Error type: {type(e)}")
        print(f"Error message: {str(e)}")
        raise

    return metrics_result

if __name__ == "__main__":
    main()