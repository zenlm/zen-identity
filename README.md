# Zen Identity

**The Soul, Datasets, and Training Infrastructure for Zen AI**

This repository contains everything needed to train Zen model identity:

## Structure

```
zen-identity/
├── soul.md              # Core identity document (Zen SOUL)
├── datasets/            # Identity training data (JSONL)
│   ├── train.jsonl      # Full dataset (all models)
│   ├── zen-nano_train.jsonl
│   ├── zen-eco_train.jsonl
│   ├── zen-coder_train.jsonl
│   ├── zen-omni_train.jsonl
│   └── zen-next_train.jsonl
├── training/            # HuggingFace Spaces training app
│   ├── app.py           # Gradio UI for cloud training
│   └── requirements.txt
└── trainer/             # Python training framework
    ├── zen_trainer/     # Core package
    ├── train_4b.py      # 4B model training script
    └── pyproject.toml   # Package config
```

## Zen SOUL

The `soul.md` document defines:

- **Core Mission**: Clarity through intelligence
- **Value Hierarchy**: Safety > Ethics > Guidelines > Helpfulness
- **Operator/User Model**: Trust levels and interaction patterns
- **Default Behaviors**: Hardcoded vs softcoded behaviors
- **Code Quality Standards**: Correctness, minimalism, idiomatics

## Identity Datasets

Training data for instilling Zen identity across model sizes:

| Model | File | Use Case |
|-------|------|----------|
| zen-nano | `zen-nano_train.jsonl` | 0.6B edge model |
| zen-eco | `zen-eco_train.jsonl` | 4B balanced model |
| zen-coder | `zen-coder_train.jsonl` | Code-focused models |
| zen-omni | `zen-omni_train.jsonl` | Multimodal model |
| zen-next | `zen-next_train.jsonl` | 32B frontier model |

## Training Options

### 1. HuggingFace Spaces (Cloud)

Deploy `training/` to a GPU Space:

```bash
# Upload to HF Space
cd training
# Create Space with A10G GPU
# Upload app.py and requirements.txt
```

### 2. Python Framework (Local/Cloud)

```bash
pip install -e trainer/

# Train 4B model
python trainer/train_4b.py

# Or use the library
from zen_trainer import ZenTrainer
trainer = ZenTrainer(model_key="qwen3-4b", dataset_path="datasets/")
trainer.train()
```

### 3. zen-coder-flash Training

For the flagship model, see [zen-coder-flash](https://github.com/zenlm/zen-coder-flash):
- MLX (Apple Silicon)
- CUDA (Local GPU)
- 8x H200 (Cloud)

## Links

- **Models**: [huggingface.co/zenlm](https://huggingface.co/zenlm)
- **Training Data**: [zen-agentic-dataset](https://huggingface.co/datasets/hanzoai/zen-agentic-dataset-private)
- **zen-coder-flash**: [github.com/zenlm/zen-coder-flash](https://github.com/zenlm/zen-coder-flash)
- **Docs**: [github.com/zenlm/docs](https://github.com/zenlm/docs)

## License

Apache 2.0

---

*Developed by [Hanzo AI](https://hanzo.ai) & [Zoo Labs Foundation](https://zoo.ngo)*
