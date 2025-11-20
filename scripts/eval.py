"""
Evaluate a fine-tuned DeepSeek model on the validation split.

Usage:
    python scripts/eval.py --model_dir outputs/checkpoints/finetuned_model
"""

import sys
import os
import yaml
from transformers import BitsAndBytesConfig
# Ensure src/ is on path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from transformers import AutoTokenizer, AutoModelForCausalLM
from src.data.load_data import get_struct_data
from src.evaluation.evaluate_model import compute_metrics
from src.utils.logger import get_logger

logger = get_logger(__name__)



def load_quant_config(cfg_dict):
    return BitsAndBytesConfig(**cfg_dict)

def model_config(path="config/model_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():#model_dir: str):
    
    finetuned_cfg = model_config()['SFT_deepseek']
    logger.info(f"Loading model from {finetuned_cfg['pretrained_model_name_or_path']}")
    quant_cfg = load_quant_config(finetuned_cfg["quantization_config"])
    model = AutoModelForCausalLM.from_pretrained(
      finetuned_cfg["pretrained_model_name_or_path"],
      quantization_config=quant_cfg,
      device_map=finetuned_cfg["device_map"])

    tokenizer = AutoTokenizer.from_pretrained(finetuned_cfg['pretrained_model_name_or_path'])
    device = model.device 


    # Load validation split
    dataset = get_struct_data()
    val_dataset = dataset["test"]

    # Generate predictions
    preds_ids, labels,preds = [], [], []
    logger.info("Running evaluation on validation split...")

    for example,i in zip(val_dataset,range(10)):
        prompt = example["prompt"]
        true_response = example["response"]

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model.generate(**inputs, max_new_tokens=128)
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)

        preds_ids.append(outputs)
        preds.append(pred)
        labels.append(true_response)
        if i == 200:   #break the loop when the model is evaluated on 200 samples
            break
  # In your formatting, "text" already contains user+assistant

    # Compute metrics
    results = compute_metrics((preds, labels,preds_ids))
    logger.info("Evaluation complete")
    print("Evaluation Results:", results)
if __name__ == "__main__":
    main()
