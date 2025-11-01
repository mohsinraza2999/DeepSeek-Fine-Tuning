"""import config

def model(id,model_name,max_seq_length,dtype,load_in_4bit):
    print(id,model_name,max_seq_length,dtype,load_in_4bit)


model(1,**config.load_args())

def lora(model,r,target_modules,use_gradient_checkpointing,
         lora_alpha,lora_dropout,bias,use_rslora,loftq_config):
    print(model,r,target_modules,use_gradient_checkpointing,
         lora_alpha,lora_dropout,bias,use_rslora,loftq_config)
lora("deepseek",**config.lora_configs())



import hydra
from omegaconf import DictConfig
from transformers import TrainingArguments
from peft import LoraConfig

@hydra.main(config_path="/content", config_name="training_config.yml")
def main_yml(cfg: DictConfig):
    training_cfg=cfg.trianing
    return TrainingArguments(
        output_dir=training_cfg.output_dir,
        per_device_train_batch_size=training_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
        max_steps=training_cfg.max_steps,
        learning_rate=training_cfg.learning_rate,
        optim=training_cfg.optim,
        weight_decay=training_cfg.weight_decay,
        save_strategy=training_cfg.save_strategy,
        save_steps=training_cfg.save_steps,
        report_to=training_cfg.report_to
    )



    # Example: pass config to training function
@hydra.main(config_path="/content", config_name="Lora_config.yml")
def load_lora_yml(cfg: DictConfig):
    lcfg = cfg.lora
    return LoraConfig(
        r=lcfg.r,
        target_modules=lcfg.target_modules,
        lora_alpha=lcfg.lora_alpha,
        lora_dropout=lcfg.lora_dropout,
        bias=lcfg.bias,
        use_rslora=lcfg.use_rslora,
        loftq_config=lcfg.loftq_config
    )

def prt():
    print(load_lora_yml())
    print(main_yml())

"""
import src.trainig.train_save as train_save
from src.utils.logger import get_logger
logger = get_logger(__name__)

def train():
    logger.info("🚀 Fine Tuning LLM Starts")
    train_save.train_model()






if __name__ == "__main__":
    train()
    

