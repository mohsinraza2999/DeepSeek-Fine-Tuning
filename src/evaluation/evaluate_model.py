from transformers import EvalPrediction
import evaluate
from src.utils.logger import get_logger
logger = get_logger(__name__)

save_path = "./finetuned_model"

# Load a metric, e.g., accuracy
accuracy_metric = evaluate.load("perplexity")

def compute_metrics(eval_preds: EvalPrediction):
    
    predictions, labels = eval_preds
    # Assuming a classification task where predictions are logits
    # You might need to apply argmax or other transformations depending on your task
    predictions = predictions.argmax(axis=1)
    return accuracy_metric.compute(predictions=predictions, references=labels)



def evaluate_model():
    logger.info("🚀 Evaluation Starts")
    pass
    """loaded_model, loaded_tokenizer = FastLanguageModel.from_pretrained(
        save_path,
        load_in_4bit=True,
        device_map="auto"   # ensures model shareds are automatically mapped to GPU
    )"""