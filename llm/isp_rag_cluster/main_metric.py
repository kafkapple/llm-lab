from omegaconf import OmegaConf
from src.eval import save_metrics
import pandas as pd
from pathlib import Path

def metric():
    cfg = OmegaConf.load("config/llm.yaml")
    
    # Convert path to Path object for correct path construction
    base_path = Path("D:/dev/isp_rag_cluster/outputs")
    
    # Verify actual existence of directory
    output_folders = [f for f in base_path.iterdir() if f.is_dir()]
    if not output_folders:
        raise FileNotFoundError("No output folders found")

    file_path = Path(r"outputs\done_4\20250315_000000_claude-3-7-sonnet-20250219_custom_prompt\dataset-isear_model-claude-3-7-sonnet-20250219.csv") #(r"outputs\20250303_172613_qwen2.5_emotion_prompt\dataset-isear_model-qwen2.5")

    # Save metrics
    save_metrics(file_path, cfg)


if __name__ == "__main__":
    metric()
