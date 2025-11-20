from unsloth import FastLanguageModel
import yaml
from src.utils.logger import get_logger
logger = get_logger(__name__)

def model_config(path="config/model_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_deepseek():
    logger.info("🚀 Loading DeepSeek Model")
    return FastLanguageModel.from_pretrained( **model_config()['DeepSeek'] # or use a custom map if needed
    )