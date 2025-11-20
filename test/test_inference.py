import sys, os
import pytest
from transformers import AutoTokenizer, AutoModelForCausalLM

# Ensure src/ is on path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

def test_inference_pipeline(tmp_path):
    """Ensure model can load and generate a short response."""
    # For speed, just load tokenizer/model from HF hub (not fine-tuned)
    model_name = "unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    prompt = "Hello, how are you?"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    outputs = model.generate(**inputs, max_new_tokens=10)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    assert isinstance(response, str)
    assert len(response) > 0
