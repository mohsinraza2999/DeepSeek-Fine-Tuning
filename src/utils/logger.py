def load_args():
    return {"model_name":"unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit",
            "max_seq_length":2048,
            "dtype":None,
            "load_in_4bit":True}

def lora_configs():
    return {"r":4,"target_modules":["q_prog","k_proj","v_proj","o_proj"],
            "use_gradient_checkpointing":"unsloth",
            "lora_alpha":16,"lora_dropout":0,
            "bias" :None,"use_rslora":False,"loftq_config":None}
def train_args():
    return{"per_device_train_batch_size":2,
           "gradient_accumulation_steps":4,
           "max_steps":5, "learning_rate":2e-4,
           "optim":"adamw_8bit","weight_decay":0.01}


import logging
from logging.handlers import RotatingFileHandler

def get_logger(name, log_file='app.log'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # File handler with rotation
    file_handler = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3)
    file_handler.setFormatter(formatter)

    # Add handlers only once
    if not logger.hasHandlers():
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger


#Usage Example in Your Script

logger = get_logger(__name__)
logger.info("Starting Fine Tuning LLM APP...")
#logger.warning("Missing 'Experience' section in CV")
#logger.error("Failed to connect to vector DB")
