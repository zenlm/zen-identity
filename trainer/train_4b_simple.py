#!/usr/bin/env python3
"""
Train zen-coder-4b with MLX (Apple Silicon optimized)

For Apple Silicon Macs with 32GB+ RAM.

Usage:
    python trainer/train_4b_simple.py
"""

import subprocess

MODEL_ID = "Qwen/Qwen3-Coder-4B"
OUTPUT_DIR = "./output/zen-coder-4b-mlx"

# First download and prepare dataset
PREPARE_CMD = """
# Download dataset from HuggingFace
python -c "
from datasets import load_dataset
import json
import os

os.makedirs('./output/zen-coder-4b-mlx', exist_ok=True)

dataset = load_dataset('zenlm/zen-identity', split='train')

# Convert to mlx_lm format
train_data = []
for item in dataset:
    text = f'<|im_start|>user\\n{item[\"instruction\"]}<|im_end|>\\n<|im_start|>assistant\\n{item[\"output\"]}<|im_end|>'
    train_data.append({'text': text})

# 90/10 split
split_idx = int(len(train_data) * 0.9)
with open('./output/zen-coder-4b-mlx/train.jsonl', 'w') as f:
    for item in train_data[:split_idx]:
        f.write(json.dumps(item) + '\\n')

with open('./output/zen-coder-4b-mlx/valid.jsonl', 'w') as f:
    for item in train_data[split_idx:]:
        f.write(json.dumps(item) + '\\n')

print(f'Prepared {len(train_data[:split_idx])} train, {len(train_data[split_idx:])} valid samples')
"
"""


def main():
    print("=" * 60)
    print("zen-coder-4b Training - MLX (Apple Silicon)")
    print(f"Base: {MODEL_ID}")
    print("=" * 60)

    # Prepare dataset
    print("\nPreparing dataset...")
    subprocess.run(PREPARE_CMD, shell=True, check=True)

    # Run MLX training
    cmd = [
        "python", "-m", "mlx_lm", "lora",
        "--model", MODEL_ID,
        "--data", OUTPUT_DIR,
        "--train",
        "--batch-size", "4",
        "--num-layers", "32",
        "--learning-rate", "2e-4",
        "--iters", "5000",
        "--steps-per-report", "10",
        "--steps-per-eval", "100",
        "--save-every", "500",
        "--adapter-path", OUTPUT_DIR,
        "--max-seq-length", "2048",
        "--grad-checkpoint",
    ]

    print("\nRunning:", " ".join(cmd))
    print("=" * 60)
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
