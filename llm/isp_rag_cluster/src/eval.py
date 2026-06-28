from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import json
import pandas as pd
import os
import logging
from transformers import AutoTokenizer
import tiktoken
import traceback
import numpy as np

logger = logging.getLogger(__name__)

def sanity_check(file_path):
    output_dir = None
    model_name = None
    df = None
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    output_dir = file_path.parent
    model_name = file_path.stem.split("_model-")[1]
    print(f"Looking for file at: {file_path}/n Model name: {model_name}")  # For debugging
    df = pd.read_csv(file_path)
    # Convert emotion labels to lowercase
    df['emotion'] = df['emotion'].str.lower()
    
    try:
        
        df[f'predicted_emotion_{model_name}'] = df[f'predicted_emotion_{model_name}'].str.lower()
    except:
        print('====No predicted_emotion column found')
        matching_columns = [col for col in df.columns if 'predicted_emotion' in col]
        if matching_columns:
            print(f"Matching columns: {matching_columns}")
            print(f"Using column: {matching_columns[0]} for predicted_emotion_{model_name}")
            df.rename(columns={matching_columns[0]: f'predicted_emotion_{model_name}'}, inplace=True)
            df[f'predicted_emotion_{model_name}'] = df[f'predicted_emotion_{model_name}'].str.lower()
            df.to_csv(file_path, index=False)
            print(f"Renamed column {matching_columns[0]} to {model_name}, saved to {file_path}")
        else:
            raise ValueError(f"No matching predicted_emotion column found for model: {model_name}")
    return output_dir, model_name, df

def analyze_unknown_predictions(file, model_name):
    """
    폴더 내 dataset CSV 파일에서 unknown 예측을 분석.
    각 파일에서 예측 컬럼과 unknown 비율, 클래스별 unknown 비율을 계산.
    """
    # for file in os.listdir(folder_path):
    file_name = os.path.basename(file.name)
    print(f"\n==file_name: {file_name}")
    if file_name.startswith("dataset-") and file_name.endswith(".csv"):
        try:
            df = pd.read_csv(file)
            pred_cols = [col for col in df.columns if 'predicted_emotion' in col]
            if not pred_cols:
                logger.warning(f"No prediction column found in {file}")
                
            pred_col = pred_cols[0]
            total_samples = len(df)
            total_unknown = (df[pred_col] == 'unknown').sum()
            total_unknown_ratio = total_unknown / total_samples * 100
            class_unknown_ratios = {}
            class_unknown_samples = {}
            print(f"Total samples: {total_samples}")
            print(f"Total unknown: {total_unknown}")
            print(f"Total unknown ratio: {total_unknown_ratio}")
            for emotion in df['emotion'].unique():
                class_samples = df[df['emotion'] == emotion]
                class_unknown = (class_samples[pred_col] == 'unknown').sum()
                class_unknown_samples[emotion] = class_unknown
                class_unknown_ratios[emotion] = class_unknown / len(class_samples) * 100
            
            unknown_stats = {
                # 'timestamp': timestamp,
                'model': model_name,
                # 'prompt': prompt_type,
                'total_samples': total_samples,
                'total_unknown': total_unknown,
                'total_unknown_ratio': total_unknown_ratio,
            }
            # 각 클래스별 unknown ratio 추가
            for emotion, ratio in class_unknown_ratios.items():
                unknown_stats[f'{emotion}_unknown_ratio'] = ratio
            for emotion, sample in class_unknown_samples.items():
                unknown_stats[f'{emotion}_unknown_samples'] = sample
            unknown_stats 
            print(f"unknown_stats: {unknown_stats}")
            output_dir = file.parent
    
            with open(output_dir / f'unknown_ratios_{model_name}.txt', "w", encoding="utf-8") as f:
                for key, value in unknown_stats.items():
                    f.write(f"{key}: {value}\n")

            print("TXT 저장 완료!")

            return unknown_stats
        except Exception as e:
            logger.error(f"Error analyzing unknown predictions in {file}: {str(e)}")
            try:
                logger.error(df.info())
            except Exception:
                pass
    return None
def save_prediction_stats(invalid_stats, output_dir, model_name, labels):
    """Save prediction statistics"""
    try:
        # Save statistics file
        stats_file = output_dir / f'prediction_stats_{model_name}.json'
        with open(stats_file, 'w', encoding='ascii') as f:
            json.dump(invalid_stats, f, indent=2, ensure_ascii=True)
    except Exception as e:
        print(f"Error saving prediction stats: {e}")

def save_metrics(df_path, cfg):

    output_dir, model_name, df = sanity_check(df_path)

    analyze_unknown_predictions(df_path, model_name)

    """Save prediction statistics and metrics"""
    print("\nData preprocessing statistics (before):")
    print(f"Total samples: {len(df)}")
    print(f"Missing values:\n{df.isna().sum()}")
    print(f"Duplicates: {df.duplicated().sum()}")
    
    # Remove missing values and duplicates
    df = df.dropna(subset=['emotion', f'predicted_emotion_{model_name}'])
    df = df.drop_duplicates()
    
    # Use label list defined in config (convert ListConfig to regular list)
    labels = list(cfg.data.datasets[cfg.data.name].labels)
    default_emotion = str(cfg.data.default_emotion)
    
    # Get unique predicted labels
    predicted_labels = df[f'predicted_emotion_{model_name}'].unique()
    
    # Find invalid predictions (not in defined labels)
    invalid_predictions = df[~df[f'predicted_emotion_{model_name}'].isin(labels)]
    
    # Create invalid predictions statistics
    invalid_stats = {
        "total_samples": len(df),
        "invalid_count": len(invalid_predictions),
        "invalid_percentage": (len(invalid_predictions) / len(df)) * 100 if len(df) > 0 else 0,
        "invalid_labels": [label for label in predicted_labels if label not in labels],
        "samples": []
    }
    
    # Add detailed information for each invalid prediction
    for _, row in invalid_predictions.iterrows():
        invalid_stats["samples"].append({
            "text": row['text'],
            "true_emotion": row['emotion'],
            "predicted_emotion": row[f'predicted_emotion_{model_name}'],
            "confidence_score": row.get(f'confidence_score_{model_name}', None),
            "explanation": row.get(f'explanation_{model_name}', None)
        })
    
    # Save invalid predictions statistics
    invalid_file = output_dir / f'invalid_predictions_{model_name}.json'
    with open(invalid_file, 'w', encoding='utf-8') as f:
        json.dump(invalid_stats, f, indent=2, ensure_ascii=False)
    
    # Select only valid samples (predictions in defined labels)
    valid_samples = df[df[f'predicted_emotion_{model_name}'].isin(labels)]
    
    print("\nPrediction validation statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Valid predictions: {len(valid_samples)}")
    print(f"Invalid predictions: {len(invalid_predictions)}")
    print(f"Invalid labels found: {invalid_stats['invalid_labels']}")
    
    # Extract actual and predicted labels for valid samples only
    y_true = valid_samples['emotion']
    y_pred = valid_samples[f'predicted_emotion_{model_name}']
    
    print(f"\nLabel information:")
    print(f"Defined labels: {labels}")
    print(f"Actual label unique values: {y_true.unique()}")
    print(f"Predicted label unique values (valid only): {y_pred.unique()}")
    
    # Generate classification report for valid samples
    report_dict = classification_report(y_true, y_pred, 
                                      labels=labels,
                                      output_dict=True, 
                                      zero_division=0)
    report_str = classification_report(y_true, y_pred, 
                                     labels=labels,
                                     zero_division=0)
    
    # Save classification report as text file
    with open(output_dir / f'classification_report_{model_name}.txt', "w", encoding='utf-8') as file:
        file.write(f"Total samples: {len(df)}\n")
        file.write(f"Valid samples: {len(valid_samples)}\n")
        file.write(f"Invalid predictions: {len(invalid_predictions)}\n")
        file.write(f"Invalid labels found: {', '.join(invalid_stats['invalid_labels'])}\n\n")
        file.write("=== Classification Report (Valid Samples Only) ===\n")
        file.write(report_str)
    
    # Save classification report as CSV with transpose
    report_df = pd.DataFrame(report_dict).round(4)
    report_df = report_df.drop(['accuracy'], errors='ignore')  # accuracy 행이 있다면 제거
    report_df = report_df.transpose()  # 행과 열을 전치
    report_df.index.name = 'class'  # 인덱스 이름을 'metric'으로 변경
    report_df.to_csv(output_dir / f'classification_report_{model_name}.csv')
    
    # Save metrics result
    metrics_result = {
        "overall": {
            "accuracy": float(report_dict['accuracy']),
            "macro_avg": report_dict.get('macro avg', {}),
            "weighted_avg": report_dict.get('weighted avg', {})
        },
        "per_class": {
            label: report_dict[label] for label in report_dict.keys()
            if label not in ['accuracy', 'macro avg', 'weighted avg']
        },
        "invalid_predictions_info": {
            "total": len(invalid_predictions),
            "percentage": (len(invalid_predictions) / len(df)) * 100 if len(df) > 0 else 0,
            "invalid_labels": invalid_stats['invalid_labels']
        }
    }
    
    
    # Calculate confusion matrix for valid samples only
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_normalized = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    
    # Plot confusion matrix in two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    
    # Original confusion matrix
    sns.heatmap(cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=labels,
                yticklabels=labels,
                ax=ax1)
    ax1.set_title(f'Confusion Matrix - {model_name}\n(Valid samples only: {len(valid_samples)})')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # Normalized confusion matrix
    sns.heatmap(cm_normalized, 
                annot=True, 
                fmt='.2f', 
                cmap='Blues',
                xticklabels=labels,
                yticklabels=labels,
                ax=ax2)
    ax2.set_title(f'Normalized Confusion Matrix - {model_name}\n(Valid samples only: {len(valid_samples)})')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_dir / f'confusion_matrix_{model_name}.png', 
                bbox_inches='tight',
                dpi=300)
    plt.close()
    
    return metrics_result, cm


def calculate_efficiency_metrics(df, model, model_name, output_dir):
    """효율성 지표 계산"""
    try:
        # F1 점수 계산
        f1_score = calculate_f1_score(df, model)
        
        # 응답 시간 기반 효율성
        response_times = df[f'response_time_{model}'].values
        f1_per_time = f1_score / (response_times + 1e-6)  # 0으로 나누기 방지
        
        # 효율성 지표를 DataFrame으로 정리
        efficiency_df = pd.DataFrame({
            'model': [model_name],
            'f1_score': [f1_score],
            'avg_response_time': [response_times.mean()],
            'f1_per_time': [f1_per_time.mean()]
        })
        
        # 개별 샘플의 f1_per_time 계산
        sample_f1_per_time = np.full(len(df), f1_per_time.mean())
        
        return efficiency_df, sample_f1_per_time
        
    except Exception as e:
        logger.error(f"Error calculating efficiency metrics for {model}: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        # 에러 발생 시 기본값 반환
        return pd.DataFrame(), np.zeros(len(df))


def count_tokens(text: str, model: str) -> int:
    """모델별 토큰 수를 계산하는 함수"""
    try:
        model = model.lower()
        logger.debug(f"Counting tokens for model: {model}")
        logger.debug(f"Input text length: {len(text)}")
        logger.debug(f"Input text preview: {text[:100]}...")
        
        # OpenAI 모델
        if "gpt" in model:
            encoding = tiktoken.encoding_for_model(model)
            token_count = len(encoding.encode(text))
            logger.debug(f"GPT token count: {token_count}")
            return token_count
            
        # Anthropic Claude 모델
        elif "claude" in model:
            encoding = tiktoken.encoding_for_model("gpt2")
            token_count = len(encoding.encode(text))
            logger.debug(f"Claude token count: {token_count}")
            return token_count
            
        # Qwen 모델
        elif "qwen" in model:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B", token=os.getenv("HUGGINGFACE_TOKEN"))
            token_count = len(tokenizer.encode(text))
            logger.debug(f"Qwen token count: {token_count}")
            return token_count
            
        # Llama 모델
        elif "llama" in model:
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=os.getenv("HUGGINGFACE_TOKEN"))
            token_count = len(tokenizer.encode(text))
            logger.debug(f"Llama token count: {token_count}")
            return token_count
            
        # Mistral 모델
        elif "mistral" in model:
            tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", token=os.getenv("HUGGINGFACE_TOKEN"))
            token_count = len(tokenizer.encode(text))
            logger.debug(f"Mistral token count: {token_count}")
            return token_count
            
        # Gemma 모델
        elif "gemma" in model:
            tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b", token=os.getenv("HUGGINGFACE_TOKEN"))
            token_count = len(tokenizer.encode(text))
            logger.debug(f"Gemma token count: {token_count}")
            return token_count
            
        # 기타 모델은 GPT-2 토크나이저 사용 (기본값)
        else:
            encoding = tiktoken.encoding_for_model("gpt2")
            token_count = len(encoding.encode(text))
            logger.debug(f"Default (GPT-2) token count: {token_count}")
            return token_count
            
    except Exception as e:
        logger.error(f"Error counting tokens for model {model}: {e}")
        logger.error(f"Error details: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return 0