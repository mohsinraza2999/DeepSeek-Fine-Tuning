import sys, os
import pytest

# Ensure src/ is on path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data.load_data import get_struct_data

def test_data_format_keys():
    """Check that dataset has prompt/response fields and non-empty splits."""
    dataset = get_struct_data(test_size=0.1)
    assert "train" in dataset and "test" in dataset
    sample = dataset["train"][0]
    assert "prompt" in sample and "response" in sample
    assert isinstance(sample["prompt"], str)
    assert isinstance(sample["response"], str)
