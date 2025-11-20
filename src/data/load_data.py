from datasets import load_dataset, Dataset
import pandas as pd
from unsloth import to_sharegpt
from unsloth import standardize_sharegpt
from src.utils.logger import get_logger
logger = get_logger(__name__)

def get_struct_data():
    logger.info("Loading Data")
    dataset=load_dataset("vicgalle/alpaca-gpt4", split="train")
    dataset=to_sharegpt(dataset,
                        merged_prompt="{instruction}[[\nYour input is :\n{input}]]",
                        output_column_name="output",
                        conversation_extension=3)
    dataset=standardize_sharegpt(dataset)


    logger.info("Data Loaded")
    split_dataset = formatting_func(dataset).train_test_split(test_size=0.2)
    logger.info("Formatting Done")


    return split_dataset



"""
def formatting_func(example):
    logger.info("🚀 Formatting Data")
    data = example["conversations"]  # example is already a list of dicts
    formatted = []

    # Iterate in pairs: user followed by assistant
    for messages in data:
      i = 0
      while i < len(messages) - 1:
          user_msg = messages[i]
          assistant_msg = messages[i + 1]

          if user_msg["role"] == "user" and assistant_msg["role"] == "assistant":
              combined={"text":f"<|user|>{user_msg['content']} <|assistant|>{assistant_msg['content']}"}
              formatted.append(combined)
              i += 2  # Move to next pair
          else:
              i += 1  # Skip malformed or out-of-order entries
    df = pd.DataFrame(formatted, columns=["text"])
    train_dataset = Dataset.from_pandas(df)

    return train_dataset

"""


def formatting_func(example):
    logger.info("🚀 Formatting Data")
    data = example["conversations"]
    formatted = []

    for messages in data:
        i = 0
        while i < len(messages) - 1:
            user_msg = messages[i]
            assistant_msg = messages[i + 1]

            if user_msg["role"] == "user" and assistant_msg["role"] == "assistant":
                formatted.append({
                    "prompt": f"<|user|>{user_msg['content']}",
                    "response": f"<|assistant|>{assistant_msg['content']}"
                })
                i += 2
            else:
                i += 1

    df = pd.DataFrame(formatted, columns=["prompt", "response"])
    return Dataset.from_pandas(df)
