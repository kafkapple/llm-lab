from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path
from dotenv import load_dotenv
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset

from src.data_prep import load_data
import hydra
import torch

from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from transformers import TrainerCallback

# 토크나이징 함수 (텍스트 컬럼: "SIT")
def tokenize_function(example, tokenizer):
    result = tokenizer(
        example["text"], 
        padding=True,
        truncation=True, 
        max_length=128,
        return_tensors=None  # 중요: None으로 설정
    )
    result["labels"] = example["label_ids"]  # 숫자 레이블 사용
    return result

@hydra.main(version_base="1.2", config_path='config', config_name='llm')
def main(cfg):
    load_dotenv()
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

    model_id = "Qwen/Qwen2.5-7B-Instruct"
    model_name = model_id.split("/")[-1]
    cache_directory = Path("models") / model_name  # 프로젝트 루트의 models 폴더 아래에 모델명으로 저장
    cache_directory.mkdir(parents=True, exist_ok=True)
    
    num_epochs = cfg.model.num_epochs
    cfg.model.template = "none"
    cfg.model.name = model_name

    # 기존 LoRA 모델을 불러올지 여부
    if cfg.model.get('resume_from_lora', False):
        lora_path = cfg.model.get('lora_path', '')
        if lora_path:
            print(f"Loading existing LoRA weights from: {lora_path}")
            # 기본 모델 로드
            model = AutoModelForSequenceClassification.from_pretrained(
                model_id,
                cache_dir=cache_directory,
                token=HUGGINGFACE_TOKEN,
                device_map="auto",
                num_labels=len(cfg.data.datasets[cfg.data.name].labels),
                torch_dtype=torch.bfloat16
            )
            
            # PeftModel.from_pretrained 사용하여 LoRA 모델 로드
            model = PeftModel.from_pretrained(
                model,
                lora_path,
                is_trainable=True
            )
            print("Successfully loaded LoRA weights")
    else:
        # 새로운 LoRA 모델 초기화 (기존 코드)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            cache_dir=cache_directory,
            token=HUGGINGFACE_TOKEN,
            device_map="auto",
            num_labels=len(cfg.data.datasets[cfg.data.name].labels),
            torch_dtype=torch.bfloat16
        )
        
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_CLS",
            inference_mode=False
        )
        model = get_peft_model(model, lora_config)

    # 학습 가능한 파라미터 출력
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(model_id, 
                                              cache_dir=cache_directory, 
                                              token=HUGGINGFACE_TOKEN)

    num_labels = len(cfg.data.datasets[cfg.data.name].labels)  # ISEAR의 7개 감정 클래스
    print(f"Dataset: {cfg.data.name}, Number of labels: {num_labels}")

    # Convert config values to Python basic types first
    model_name = str(cfg.model.name)

    from src.logger import setup_logging
    from src.util import sanitize_model_name, get_output_dir_name
    # Set up output directory first
    model_name = sanitize_model_name(model_name)
    
    output_dir_name = get_output_dir_name(model_name, cfg)
    output_dir = Path('outputs') / 'lora' / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Update cfg output_path
    cfg.general.output_path = str(output_dir)
    
    # Now setup logging with the created output directory
    logger = setup_logging(cfg, output_dir)
    df, labels = load_data(cfg)
    labels = list(map(str, labels))
    
    # Set n_samples
    n_samples = len(df) if cfg.data.n_samples == -1 else min(cfg.data.n_samples, len(df))
    logger.info(f"Using sanitized model name: {model_name}")
    logger.info(f"Processing {n_samples} samples out of {len(df)} total samples")
    df = df.head(n_samples)
    # train (60%), validation (20%), test (20%) 분리 (라벨 분포 유지)
    train_df, temp_df = train_test_split(df, test_size=0.4, random_state=cfg.general.random_state, stratify=df["emotion"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=cfg.general.random_state, stratify=temp_df["emotion"])

    # 레이블을 숫자로 변환하는 함수 추가
    def convert_labels_to_ids(labels, label_to_id):
        return [label_to_id[label] for label in labels]

    # 레이블 매핑 생성
    unique_labels = sorted(list(set(df['emotion'])))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    
    # 데이터프레임에 숫자 레이블 추가
    train_df['label_ids'] = convert_labels_to_ids(train_df['emotion'], label_to_id)
    val_df['label_ids'] = convert_labels_to_ids(val_df['emotion'], label_to_id)

    # 데이터셋 변환 및 전처리
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    tokenized_train = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    tokenized_val = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names
    )

    # TrainingArguments 수정
    training_args = TrainingArguments(
        output_dir=str(output_dir),  # 출력 디렉토리 통일
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=num_epochs,
        eval_strategy="epoch",
        learning_rate=1e-4,
        logging_steps=100,
        save_steps=500,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        report_to="none",
        optim="adamw_torch",
        remove_unused_columns=True
    )

    # 데이터 콜레이터 추가
    from transformers import DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=1)
        
        # config의 레이블 리스트를 일반 Python 리스트로 변환
        label_names = list(cfg.data.datasets[cfg.data.name].labels)
        
        # softmax를 적용하여 확률값으로 변환
        probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
        confidence_scores = np.max(probabilities, axis=1)
        
        # numpy.int64를 일반 정수로 변환하여 인덱싱
        results_df = pd.DataFrame({
            'text': val_df['text'].values,
            'emotion': [label_names[int(l)] for l in labels],  # int()로 변환
            f'predicted_emotion_{model_name}': [label_names[int(p)] for p in predictions],  # int()로 변환
            f'confidence_score_{model_name}': confidence_scores,
        })
        
        # 각 클래스별 확률 추가
        for i, label in enumerate(label_names):
            results_df[f'prob_{label}'] = probabilities[:, i]
        
        # 임시 평가 파일 저장
        eval_file = output_dir / f'{cfg.data.name}_model-{model_name}.csv'
        results_df.to_csv(eval_file, index=False)
        
        from src.eval import save_metrics
        metrics_result, _ = save_metrics(eval_file, cfg)
        
        # 필요한 메트릭 추출
        metrics_dict = {
            'accuracy': metrics_result['overall']['accuracy'],
            'macro_f1': metrics_result['overall']['macro_avg']['f1-score']
        }
        
        # 클래스별 메트릭 추가
        for label, class_metrics in metrics_result['per_class'].items():
            metrics_dict.update({
                f'{label}_precision': class_metrics['precision'],
                f'{label}_recall': class_metrics['recall'],
                f'{label}_f1': class_metrics['f1-score']
            })

        return metrics_dict

    # 학습 곡선을 저장하기 위한 콜백 클래스 추가
    class MetricsPlottingCallback(TrainerCallback):
        def __init__(self, output_dir):
            self.output_dir = Path(output_dir)
            self.train_losses = []
            self.eval_losses = []
            self.eval_metrics = {
                'accuracy': [],
                'macro_f1': []
            }
            self.steps = []
            
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return
            
            # 학습 손실 저장
            if 'loss' in logs:
                self.train_losses.append(logs['loss'])
                self.steps.append(state.global_step)
                
            # 평가 메트릭 저장
            if 'eval_loss' in logs:
                self.eval_losses.append(logs['eval_loss'])
                self.eval_metrics['accuracy'].append(logs.get('eval_accuracy', 0))
                self.eval_metrics['macro_f1'].append(logs.get('eval_macro_f1', 0))
                
        def on_train_end(self, args, state, control, **kwargs):
            # 손실 그래프
            plt.figure(figsize=(10, 6))
            plt.plot(self.steps, self.train_losses, label='Training Loss')
            if self.eval_losses:
                # epoch 기반 평가일 때는 균등하게 분배
                n_epochs = len(self.eval_losses)
                eval_steps = np.linspace(0, len(self.steps), n_epochs+1)[1:]
                plt.plot(eval_steps, self.eval_losses, label='Validation Loss')
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(self.output_dir / 'loss_curve.png')
            plt.close()
            
            # 평가 메트릭 그래프
            if self.eval_metrics['accuracy']:
                plt.figure(figsize=(10, 6))
                n_epochs = len(self.eval_metrics['accuracy'])
                eval_steps = np.linspace(0, len(self.steps), n_epochs+1)[1:]
                plt.plot(eval_steps, self.eval_metrics['accuracy'], label='Accuracy')
                plt.plot(eval_steps, self.eval_metrics['macro_f1'], label='Macro F1')
                plt.xlabel('Steps')
                plt.ylabel('Score')
                plt.title('Evaluation Metrics')
                plt.legend()
                plt.grid(True)
                plt.savefig(self.output_dir / 'metrics_curve.png')
                plt.close()

    # Trainer 수정
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[MetricsPlottingCallback(output_dir)]  # 콜백 추가
    )

    # 파인튜닝 실행
    trainer.train()

    # 최종 평가 수행 및 결과 저장
    final_eval_results = trainer.evaluate()
    
    # 평가 결과를 JSON 형태로 저장
    with open(output_dir / 'eval_results.json', 'w') as f:
        json.dump(final_eval_results, f, indent=4)

    # 파인튜닝 후 모델 저장
    model.save_pretrained(output_dir / "final_model")

    # 레이블 매핑 정보 저장
    label_mapping = {
        'id_to_label': {str(v): k for k, v in label_to_id.items()},
        'label_to_id': label_to_id
    }
    with open(output_dir / 'label_mapping.json', 'w') as f:
        json.dump(label_mapping, f, indent=4)

if __name__ == "__main__":
    main()

