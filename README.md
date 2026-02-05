# zen-identity

Identity training for Zen AI models.

## Contents

```
├── soul.md              # Core identity document
├── datasets/            # Training JSONL per model
├── training/            # HuggingFace Spaces app
└── trainer/             # Python training package
```

## Datasets

| File | Model |
|------|-------|
| `zen-nano_train.jsonl` | 0.6B |
| `zen-eco_train.jsonl` | 4B |
| `zen-coder_train.jsonl` | 7B-31B |
| `zen-omni_train.jsonl` | 7B multimodal |
| `zen-next_train.jsonl` | 32B |

## Training

**HF Spaces:**
```bash
# Deploy training/ to GPU Space (A10G)
```

**Local:**
```bash
pip install -e trainer/
python trainer/train_4b.py
```

**zen-coder-flash (flagship):**
```bash
git clone https://github.com/zenlm/zen-coder-flash
python training/train_mlx.py   # Apple Silicon
python training/train_cuda.py  # NVIDIA
```

## Links

- Models: https://huggingface.co/zenlm
- Docs: https://github.com/zenlm/docs
- Dataset: https://huggingface.co/datasets/hanzoai/zen-agentic-dataset-private
