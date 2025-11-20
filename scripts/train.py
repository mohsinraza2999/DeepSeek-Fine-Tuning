"""
Thin wrapper to launch training with one command:
    python scripts/train.py
"""

import sys
import os

# Ensure src/ is on path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.trainig.train_save import train_model
from src.utils.logger import get_logger

logger = get_logger(__name__)

if __name__ == "__main__":
    logger.info("Starting training pipeline...")
    train_model()
    logger.info("Training pipeline finished.")
