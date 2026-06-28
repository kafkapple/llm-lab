import torch
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig
import json

class EmotionClassifier:
    def __init__(self, base_model_path: str, lora_path: str, device: str = "cuda"):
        self.device = device
        self.load_model(base_model_path, lora_path)
        
    def load_model(self, base_model_path: str, lora_path: str):
        # LoRA 설정 로드
        peft_config = PeftConfig.from_pretrained(lora_path)
        
        # 레이블 매핑 정보 로드
        lora_path = Path(lora_path)
        with open(lora_path.parent / 'label_mapping.json', 'r') as f:
            label_mapping = json.load(f)
            self.id2label = label_mapping['id_to_label']
            self.label2id = label_mapping['label_to_id']
            num_labels = len(self.label2id)
        
        # 캐시 디렉토리 설정
        cache_directory = Path("models") / base_model_path.split("/")[-1]
        cache_directory.mkdir(parents=True, exist_ok=True)
        
        # 기본 모델 로드
        self.model = AutoModelForSequenceClassification.from_pretrained(
            base_model_path,  # 원본 모델 ID 사용
            cache_dir=cache_directory,  # 캐시 디렉토리 지정
            num_labels=num_labels,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # LoRA 가중치 로드
        self.model = PeftModel.from_pretrained(self.model, lora_path)
        
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        
        self.model.eval()
    
    def predict(self, text: str) -> dict:
        # 입력 텍스트 토크나이징
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        # 모든 입력을 모델과 같은 디바이스로 이동
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # 추론
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=-1)
            predicted_class_id = torch.argmax(predictions, dim=-1).item()
            
        return {
            "predicted_class": self.id2label[str(predicted_class_id)],  # 클래스 ID를 레이블로 변환
            "probabilities": {
                self.id2label[str(i)]: prob.item()  # 각 클래스별 확률
                for i, prob in enumerate(predictions[0])
            }
        }

# 사용 예시
def load_emotion_classifier(base_model_path: str, lora_path: str) -> EmotionClassifier:
    """
    감정 분류 모델을 로드하는 헬퍼 함수
    
    Args:
        base_model_path: 기본 모델 경로 (예: "Qwen/Qwen2.5-7B-Instruct")
        lora_path: 저장된 LoRA 가중치 경로
    
    Returns:
        EmotionClassifier 인스턴스
    """
    return EmotionClassifier(base_model_path, lora_path) 