import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import argparse

import re
def sanitize_model_name(model_name: str) -> str:
    """Clean model name for use as file name"""
    return re.sub(r'[^\w\-\.]', '_', model_name.lower())

def get_emotion_colors():
    """감정별 구분되는 색상 정의"""
    return {
        'joy': '#FF0000',       # 빨강
        'fear': '#00FF00',      # 초록
        'anger': '#0000FF',     # 파랑
        'sadness': '#FFD700',   # 골드
        'disgust': '#FF00FF',   # 마젠타
        'shame': '#00FFFF',     # 시안
        'guilt': '#FFA500'      # 주황
    }

def load_prediction_results(prediction_file):
    """예측 결과 파일 로드 및 분석"""
    df = pd.read_csv(prediction_file)
    
    # 예측 컬럼 찾기 (predicted_emotion이 포함된 컬럼)
    pred_cols = [col for col in df.columns if 'predicted_emotion' in col]
    if not pred_cols:
        raise ValueError("No prediction column found in the file")
    
    pred_col = pred_cols[0]  # 첫 번째 예측 컬럼 사용
    
    # 예측 일치/불일치 분석
    df['prediction_match'] = df['emotion'] == df[pred_col]
    
    return df

def load_embeddings(vector_db_path, model_name):
    """벡터 DB에서 임베딩 데이터 로드"""
    sanitized_name = sanitize_model_name(model_name)
    
    # 모든 pkl 파일 찾기
    all_pkl_files = list(Path(vector_db_path).glob("*.pkl"))
    
    # 모델 이름이 포함된 파일 찾기
    matching_files = [f for f in all_pkl_files if sanitized_name in f.name.lower()]
    
    if not matching_files:
        raise FileNotFoundError(f"No embedding file found containing model name: {model_name}")
    
    # 가장 최근 파일 사용
    latest_file = max(matching_files, key=lambda x: x.stat().st_mtime)
    print(f"Found matching file: {latest_file}")
    print(f"Loading embeddings from {latest_file}")
    
    with open(latest_file, 'rb') as f:
        data = pickle.load(f)
    
    # FAISS 인덱스에서 데이터 추출
    if hasattr(data, 'docstore') and hasattr(data, 'index'):
        # texts 추출
        texts = [doc.page_content for doc in data.docstore._dict.values()]
        
        # 임베딩 추출
        embeddings = []
        for i in range(len(texts)):
            vector = data.index.reconstruct(i)
            embeddings.append(vector)
        
        return {
            'texts': texts,
            'embeddings': np.array(embeddings)
        }
    else:
        return data

def load_original_data(data_path):
    """원본 데이터 로드"""
    df = pd.read_csv(data_path)
    return df

def visualize_embeddings(embeddings, labels, save_path, title, prediction_data=None):
    """임베딩 시각화 및 저장
    
    Args:
        embeddings: 임베딩 벡터
        labels: 감정 레이블
        save_path: 저장 경로
        title: 그래프 제목
        prediction_data: 예측 결과 데이터 (optional)
    """
    # t-SNE로 차원 축소
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # 전체 데이터의 범위 계산
    x_min, x_max = embeddings_2d[:, 0].min(), embeddings_2d[:, 0].max()
    y_min, y_max = embeddings_2d[:, 1].min(), embeddings_2d[:, 1].max()
    
    # 여유 공간 추가 (10%)
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    x_lim = [x_min - x_margin, x_max + x_margin]
    y_lim = [y_min - y_margin, y_max + y_margin]
    
    # 감정별 색상 정의
    emotion_colors = get_emotion_colors()
    
    if prediction_data is not None:
        # 예측 결과가 있는 경우, 3개의 subplot 생성
        plt.figure(figsize=(20, 7))
        
        # 1. 전체 데이터 (Match/Mismatch)
        plt.subplot(1, 3, 1)
        match_mask = prediction_data['prediction_match']
        
        # 불일치 데이터 먼저 그리기 (더 큰 마커로)
        plt.scatter(embeddings_2d[~match_mask, 0], embeddings_2d[~match_mask, 1],
                   c='red', marker='x', s=100, alpha=0.6, label='Mismatch')
        
        # 일치 데이터 그리기
        plt.scatter(embeddings_2d[match_mask, 0], embeddings_2d[match_mask, 1],
                   c='blue', marker='o', s=50, alpha=0.3, label='Match')
        
        plt.title(f'{title}\nAll Data - Match/Mismatch')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend()
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        
        # 2. Invalid 데이터만 (감정별 색상)
        plt.subplot(1, 3, 2)
        invalid_mask = ~match_mask
        for label in np.unique(labels):
            mask = (labels == label) & invalid_mask
            if np.any(mask):  # 해당 감정의 invalid 데이터가 있는 경우만
                color = emotion_colors.get(label, '#808080')
                plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                           c=[color], label=label, alpha=0.6)
        
        plt.title(f'{title}\nMismatch Data by Emotion')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        
        # 3. Valid 데이터만 (감정별 색상)
        plt.subplot(1, 3, 3)
        valid_mask = match_mask
        for label in np.unique(labels):
            mask = (labels == label) & valid_mask
            if np.any(mask):  # 해당 감정의 valid 데이터가 있는 경우만
                color = emotion_colors.get(label, '#808080')
                plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                           c=[color], label=label, alpha=0.6)
        
        plt.title(f'{title}\nMatch Data by Emotion')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        
    else:
        # 예측 결과가 없는 경우, 단일 플롯으로 감정별 시각화
        plt.figure(figsize=(10, 7))
        for label in np.unique(labels):
            mask = labels == label
            color = emotion_colors.get(label, '#808080')  # 정의되지 않은 감정은 회색
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                       c=[color], label=label, alpha=0.6)
        
        plt.title(f'{title}\nEmotion Clusters')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xlim(x_lim)
        plt.ylim(y_lim)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 커맨드 라인 인자 파싱
    parser = argparse.ArgumentParser(description='Visualize emotion embeddings')
    parser.add_argument('--vector_db', default='data/vector_db', help='Vector DB path')
    parser.add_argument('--data', default='data/isear/preped/isear_dataset_preprocessed.csv', help='Original data path')
    parser.add_argument('--output', default='outputs/embedding_analysis', help='Output directory')
    parser.add_argument('--prediction_file', default='outputs/done_4/20250314_065109_gemma3_rag_prompt/dataset-isear_model-gemma3.csv', help='Optional prediction results file')
    parser.add_argument('--model', default='bge-m3', help='Model name')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output, exist_ok=True)
    
    # 원본 데이터 로드
    original_data = load_original_data(args.data)
    
    # 예측 결과 로드 (있는 경우)
    prediction_data = None
    if args.prediction_file:
        try:
            prediction_data = load_prediction_results(args.prediction_file)
            print(f"Loaded prediction data from {args.prediction_file}")
        except Exception as e:
            print(f"Warning: Failed to load prediction file: {e}")
            traceback.print_exc()
    
    try:
        print(f"\nProcessing model: {args.model}")
        
        # 임베딩 데이터 로드
        data = load_embeddings(args.vector_db, args.model)
        
        # 임베딩과 레이블 추출
        if 'embeddings' in data and 'texts' in data:
            embeddings = np.array(data['embeddings'])
            texts = data['texts']
            
            # texts를 기반으로 원본 데이터에서 emotion 레이블 매칭
            text_to_emotion = dict(zip(original_data['text'], original_data['emotion']))
            labels = np.array([text_to_emotion[text] for text in texts])
            
            # 저장 경로 설정
            base_save_path = os.path.join(args.output, f"embeddings_{sanitize_model_name(args.model)}")
            
            # 시각화 및 저장
            visualize_embeddings(
                embeddings=embeddings,
                labels=labels,
                save_path=base_save_path + ("_with_predictions.png" if prediction_data is not None else ".png"),
                title=f'Emotion Clusters - {args.model}',
                prediction_data=prediction_data
            )
            
            print(f"Visualization saved to {args.output}")
            
        else:
            print(f"Warning: Invalid data format for model {args.model}")
        
    except Exception as e:
        print(f"Error processing model {args.model}: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 