# zen-identity

Identity training for Zen AI models.

## Zen Coder Family

| Model | Base | Size | Script |
|-------|------|------|--------|
| zen-coder-4b | Qwen3-Coder-4B | 4B | `train_4b.py` |
| zen-coder | Devstral-Small-2-24B | 24B | - |
| **zen-coder-flash** ⭐ | **GLM-4.7-Flash** | **31B MoE (3B active)** | `train_zen_coder_flash.py` |
| zen-coder-max | Kimi-K2 | 671B MoE (14B active) | `train_zen_coder_max.py` |

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
│   ├── train_zen_coder_flash.py # GLM-4.7-Flash training
│   ├── train_zen_coder_max.py   # Kimi-K2 training
│   └── train_4b.py              # Qwen3-4B training
└── training/
    └── app.py                   # HF Spaces Gradio app
```

## Links

- Models: https://huggingface.co/zenlm
- Dataset: https://huggingface.co/datasets/zenlm/zen-identity
- zen-coder-flash repo: https://github.com/zenlm/zen-coder-flash
