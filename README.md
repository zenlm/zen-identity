# zen-identity

Identity training for Zen AI models.

## Zen Coder Family

| Model | Base | Size | Script |
|-------|------|------|--------|
| zen-coder-4b | Zen Coder 4B | 4B | `train_4b.py` |
| zen-coder | Zen Coder 24B base | 24B | - |
| **zen-coder-flash** ⭐ | **Zen Coder Flash** | **31B MoE (3B active)** | `train_zen_coder_flash.py` |
| zen-coder-max | Zen Coder MAX | 671B MoE (14B active) | `train_zen_coder_max.py` |

## Training

### zen-coder-flash (Flagship)

```bash
# CUDA
pip install torch transformers peft bitsandbytes datasets
python trainer/train_zen_coder_flash.py

# MLX (Apple Silicon)
git clone https://github.com/zenlm/zen-coder-flash
python training/train_mlx.py
```

### zen-coder-max (Frontier)

Requires 4x A100 80GB or 8x H200:

```bash
python trainer/train_zen_coder_max.py

# Multi-GPU
torchrun --nproc_per_node 4 trainer/train_zen_coder_max.py
```

## Contents

```
├── soul.md                      # Core identity document
├── datasets/                    # Training JSONL
├── trainer/
│   ├── train_zen_coder_flash.py # Zen Coder Flash training
│   ├── train_zen_coder_max.py   # Zen Coder MAX training
│   └── train_4b.py              # Zen Coder 4B training
└── training/
    └── app.py                   # HF Spaces Gradio app
```

## Links

- Models: https://huggingface.co/zenlm
- Dataset: https://huggingface.co/datasets/zenlm/zen-identity
- zen-coder-flash repo: https://github.com/zenlm/zen-coder-flash
