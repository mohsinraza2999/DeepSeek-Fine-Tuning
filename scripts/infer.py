"""
Run inference with a fine-tuned DeepSeek model.

Usage:
    python scripts/infer.py --model_dir outputs/checkpoints/finetuned_model --prompt "Write a haiku about mountains"
"""

import sys
import os
import argparse
import yaml
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

# Ensure src/ is on path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.utils.logger import get_logger
logger = get_logger(__name__)

def load_quant_config(cfg_dict):
    return BitsAndBytesConfig(**cfg_dict)

def model_config(path="config/model_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main(prompt: str):
    finetuned_cfg = model_config()['SFT_deepseek']
    logger.info(f"Loading model from {finetuned_cfg['pretrained_model_name_or_path']}")
    quant_cfg = load_quant_config(finetuned_cfg["quantization_config"])
    model = AutoModelForCausalLM.from_pretrained(
      finetuned_cfg["pretrained_model_name_or_path"],
      quantization_config=quant_cfg,
      device_map=finetuned_cfg["device_map"])

    tokenizer = AutoTokenizer.from_pretrained(finetuned_cfg['pretrained_model_name_or_path'])
    device = model.device 

    logger.info(f"Generating response for prompt: {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(**inputs, max_new_tokens=128)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n=== Prompt ===")
    print(prompt)
    print("\n=== Model Response ===")
    print(response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    """parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to fine-tuned model directory")
    parser.add_argument("--max_new_tokens", type=int, default=200,
                        help="Maximum tokens to generate")"""
    
    parser.add_argument("--prompt", type=str, required=True,
                        help="User prompt for inference")
    
    args = parser.parse_args()
    main(args.prompt)
