from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments,TrainerCallback
import yaml
import os
import torch
import src.data.load_data as load_data
import src.evaluation.evaluate_model as evaluate_model
import src.model.deepseek_model as deepseek_model
from src.utils.logger import get_logger
logger = get_logger(__name__)
SAVE_PATH = "./outputs/checkpoints/finetuned_model"

"""
def load_train_model():

    model, tokenizer=FastLanguageModel.from_pretrained(**config.load_args())

    model=FastLanguageModel.get_peft_model(model,**config.lora_configs())

    dataset=get_data.get_struct_data()

    trainer=SFTTrainer(model=model, tokenizer=tokenizer,
                    train_dataset=dataset,args=TrainngArguments(config.train_args()))

    trainer_status=trainer.train()
"""
class SaveMetricsCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        with open("epoch_metrics.txt", "a") as f:
            if logs is not None:
                f.write(f"Epoch {state.epoch} - Loss: {logs.get('loss')}, Metrics: {logs}\n")
            else:
                f.write(f"Epoch {int(state.epoch)} finished (no logs dict provided)\n")


def load_training_yml(path="config/training_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)



def load_lora_yml(path="config/Lora_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train_model():
    os.environ["WANDB_MODE"] = "disabled"
    
    trainer=define_trainer()
    logger.info("Fine Tuning Starts")
    results=trainer.train()
    logger.info("Fine Tuning Complete")



    logger.info("Saving Fine Tuned Model")
    trainer.save_model(SAVE_PATH)
    logger.info(f"Model Saved at {SAVE_PATH}")

    torch.cuda.empty_cache()       # clears cached memory
    torch.cuda.reset_peak_memory_stats()  # optional: reset peak usage
    del trainer
    torch.cuda.empty_cache()


def define_lora_config(model):
    return FastLanguageModel.get_peft_model(model,**load_lora_yml()['lora'])

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
        args = TrainingArguments(**load_training_yml()['trianing']),
        formatting_func = lambda batch: [f"{p}\n{r}" for p, r in zip(batch["prompt"], batch["response"])],
        callbacks=[SaveMetricsCallback()],
        report_to="none"
    )#formatting_func = lambda batch: [f"{p}\n{r}" for p, r in zip(batch["prompt"], batch["response"])]