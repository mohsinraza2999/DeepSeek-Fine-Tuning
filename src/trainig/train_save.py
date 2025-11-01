from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
import os
import torch
import src.data.load_data as load_data
import src.evaluation.evaluate_model as evaluate_model
import src.model.deepseek_model as deepseek_model
from src.utils.logger import get_logger
logger = get_logger(__name__)
save_path = "./finetuned_model"

"""
def load_train_model():

    model, tokenizer=FastLanguageModel.from_pretrained(**config.load_args())

    model=FastLanguageModel.get_peft_model(model,**config.lora_configs())

    dataset=get_data.get_struct_data()

    trainer=SFTTrainer(model=model, tokenizer=tokenizer,
                    train_dataset=dataset,args=TrainngArguments(config.train_args()))

    trainer_status=trainer.train()
"""

def load_training_yml():
    return {"output_dir": "./results",
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "max_steps": 5,
    "learning_rate": 0.0002,
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "save_strategy": "steps",
    "save_steps": 5,
    "report_to": "none"}



def load_lora_yml():
    return {"r": 4,
    "target_modules": ["q_prog", "k_proj", "v_proj", "o_proj"],
    "use_gradient_checkpointing": "unsloth",
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "bias": "none",
    "use_rslora": False,
    "loftq_config": None}


def train_model():
    os.environ["WANDB_MODE"] = "disabled"
    
    trainer=define_trainer()
    logger.info("Fine Tuning Starts")
    results=trainer.train()
    logger.info("Fine Tuned The Model")



    logger.info("Saving Fine Tuned Model")
    trainer.save_model(save_path)
    logger.info("Model Saved")

    torch.cuda.empty_cache()       # clears cached memory
    torch.cuda.reset_peak_memory_stats()  # optional: reset peak usage
    del trainer
    torch.cuda.empty_cache()


def define_lora_config(model):
    return FastLanguageModel.get_peft_model(model,**load_lora_yml())

def define_trainer():
    
    

    dataset=load_data.get_struct_data()

    model, tokenizer = deepseek_model.load_deepseek()
    logger.info("Model Loaded")
    logger.info("Setting Training Configs")
    return SFTTrainer(
        model = define_lora_config(model),
        tokenizer = tokenizer,
        train_dataset = dataset['train'],
        eval_dataset=dataset["test"],
        dataset_text_field = "text",
        compute_metrics=evaluate_model.compute_metrics,
        max_seq_length = 512,
        args = TrainingArguments(**load_training_yml()),
        report_to="none"
    )