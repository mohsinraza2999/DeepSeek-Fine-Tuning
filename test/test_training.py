import sys, os
import pytest

# Ensure src/ is on path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.training.train_model import define_trainer

def test_trainer_initialization():
    """Smoke test: trainer should initialize without errors."""
    trainer = define_trainer()
    assert trainer is not None
    # Check trainer has model and tokenizer
    assert hasattr(trainer, "model")
    assert hasattr(trainer, "tokenizer")
