# DeepSeek-Fine-Tuning with Unsloth & Alpaca-GPT4🔧

![DeepSeek Fine-Tuning](https://img.shields.io/badge/LLM-Fine%20Tuning-blue)
![Python](https://img.shields.io/badge/Python-3.9%2B-green)
![License](https://img.shields.io/badge/License-MIT-orange)

## 🧠 Overview

This repository demonstrates instruction fine-tuning of the DeepSeek Large Language Model using the vicgalle/alpaca-gpt4 dataset and Unsloth’s FastLanguageModel.from_pretrained for highly efficient training.

The project focuses on fast, memory-optimized LLM fine-tuning, enabling large-model experimentation on limited hardware while maintaining strong instruction-following performance.

---

## 🚀 Key Highlights

⚡ Unsloth FastLanguageModel for optimized model loading

🧠 Instruction fine-tuning with alpaca-gpt4

🔧 Modular, extensible training pipeline

💾 Reduced VRAM usage and faster training

📦 Clean separation of configs, scripts, core logic, and tests

## 📘 Dataset
vicgalle/alpaca-gpt4

A high-quality instruction dataset generated using GPT-4, designed to align open-source LLMs with human-like instruction following.

Sample format:
```json
{
  "instruction": "Explain bias vs variance.",
  "input": "",
  "output": "Bias refers to systematic error..."
}
```

This dataset improves:

Instruction adherence

Reasoning quality

Response clarity and structure

## 🧠 Model Loading (Unsloth)

The model is loaded using Unsloth’s optimized API, enabling:

Faster initialization

Memory-efficient training

Seamless PEFT / LoRA integration

    ```python
    from unsloth import FastLanguageModel

    def load_deepseek():
        logger.info("🚀 Loading DeepSeek Model")
        return FastLanguageModel.from_pretrained( **model_config()['DeepSeek'] # or use a custom map if needed
        )
    ```

This approach allows fine-tuning large models on consumer-grade GPUs.

## 🏗️ Project Structure

    ```text
    DeepSeek-Fine-Tuning/
    │
    ├── config/               # Training & model configs
    ├── scripts/              # Training and inference scripts
    ├── src/                  # Core fine-tuning logic
    ├── test/                 # Validation tests
    ├── run_pipeline.sh       # End-to-end pipeline execution
    ├── README.md
    └── LICENSE
    ```
## 🛠️ Tech Stack

Python

PyTorch

Hugging Face Transformers & Datasets

Unsloth

DeepSeek LLM

CUDA

## ⚙️ Setup & Installation

    ```bash
    git clone https://github.com/mohsinraza2999/DeepSeek-Fine-Tuning.git
    cd DeepSeek-Fine-Tuning
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```


Requirements

Python 3.9+

NVIDIA GPU (CUDA recommended)

Minimum VRAM depends on sequence length and batch size

## 🚀 Training

Run the complete fine-tuning pipeline:

 
    ```bash
    run_pipeline.sh
    ```


This performs:

Dataset loading (alpaca-gpt4)

Optimized model initialization (Unsloth)

Instruction fine-tuning

Checkpoint saving

## 🔍 Inference
    ```bash
    python scripts/infer.py \ --prompt "Explain gradient descent in simple terms."
    ```


The fine-tuned model produces:

More structured answers

Better reasoning

Improved instruction compliance vs base model

## 📊 Evaluation

Current evaluation:

Qualitative comparison with base DeepSeek

Instruction-following accuracy

Coherence and relevance of responses

Planned improvements:

Automated benchmarks

Human preference evaluation

Task-specific metrics

## 📈 Results & Observations

⚡ Faster training compared to standard HF loading

💾 Significant VRAM reduction using Unsloth

📈 Improved instruction adherence after fine-tuning

🧠 Stable convergence on alpaca-gpt4

💡 Use Cases

Instruction-following AI assistants

Technical and educational Q&A systems

Foundation for RAG-based LLM applications

Cost-efficient enterprise LLM adaptation

## 🔮 Future Enhancements

LoRA hyperparameter experiments

RAG integration with vector databases

MLflow / W&B experiment tracking

FastAPI inference service

Quantized deployment (4-bit / 8-bit)

## 👨‍💻 Author

Mohsin Raza
AI / ML Engineer
🔗 https://github.com/mohsinraza2999