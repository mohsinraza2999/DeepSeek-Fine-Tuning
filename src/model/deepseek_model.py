from unsloth import FastLanguageModel
from src.utils.logger import get_logger
logger = get_logger(__name__)

def load_deepseek():
    logger.info("🚀 Loading DeepSeek Model")
    return FastLanguageModel.from_pretrained(
    model_name="unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit",
    quantization_config={
        "load_in_4bit":True,
        "llm_int8_enable_fp32_cpu_offload": True
    },
    device_map="auto"  # or use a custom map if needed
    )