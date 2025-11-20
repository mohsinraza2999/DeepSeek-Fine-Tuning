from transformers import EvalPrediction
import evaluate
#from datasets import load_metric
import numpy as np
from src.utils.logger import get_logger
logger = get_logger(__name__)

# Load a metric
perplexity_metric = evaluate.load("perplexity", module_type="metric")
# Load standard metrics
rouge = evaluate.load("rouge")
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_preds: EvalPrediction):
    """
    Compute ROUGE and accuracy for generated vs. reference texts.
    eval_preds: (predictions, labels) from Trainer
    """
    predictions, labels, preds = eval_preds

    perplexity_score=perplexity_metric.compute(predictions=predictions, references=labels, add_start_token=False, model_id='gpt2')
    # Decode if labels are token IDs
    if isinstance(preds[0], (list, np.ndarray)):
        preds = [p for p in preds]
    if isinstance(labels[0], (list, np.ndarray)):
        labels = [l for l in labels]

    # Convert to strings if needed
    preds = [str(p) for p in preds]
    labels = [str(l) for l in labels]

    # Compute metrics
    rouge_scores = rouge.compute(predictions=predictions, references=labels)
    

    # Merge results
    results = {**rouge_scores,"mean_perplexity":perplexity_score['mean_perplexity']}
    return results
