import sys, os
import pytest
import src.model.deepseek_model as deepseek_model
# Ensure src/ is on path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

def test_inference_pipeline(tmp_path):
    """Ensure model can load and generate a short response."""
    # For speed, just load tokenizer/model from HF hub (not fine-tuned)
    model, tokenizer = deepseek_model.load_deepseek()

    prompt = "Hello, how are you?"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    outputs = model.generate(**inputs, max_new_tokens=10)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    assert isinstance(response, str)
    assert len(response) > 0
