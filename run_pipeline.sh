#!/bin/bash
# Orchestration script for DeepSeek fine-tuning project
# Usage: bash run_pipeline.sh

set -e  # stop on first error

#echo "=== Step 1: Preparing data ==="
#python scripts/prepare_data.py --dataset vicgalle/alpaca-gpt4 \
#    --train_out data/train.jsonl \
#    --eval_out data/val.jsonl \
#    --val_ratio 0.2

echo "=== Step 2: Training model ==="
python scripts/train.py --config configs/train.yaml

echo "=== Step 3: Evaluating model ==="
python scripts/eval.py --config configs/eval.yaml

echo "=== Step 4: Inference demo ==="
python scripts/infer.py --model_dir outputs/checkpoints/finetuned_model \
    --prompt "Write a short poem about the stars."
