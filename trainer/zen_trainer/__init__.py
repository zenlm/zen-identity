"""
Zen Trainer - Fine-tune Zen Coder models

Models:
  zen-coder-4b      Qwen3-Coder-4B         4B      8GB VRAM
  zen-coder         Devstral-Small-2-24B   24B    24GB VRAM
  zen-coder-flash   GLM-4.7-Flash          31B MoE (3B active)  24GB VRAM  ⭐
  zen-coder-max     Kimi-K2                671B MoE (14B active) 160GB VRAM

Usage:
    from zen_trainer import ZenTrainer, get_model_config

    trainer = ZenTrainer("zen-coder-flash", "zenlm/zen-identity")
    trainer.train()
"""

__version__ = "0.1.0"

from .models import ZEN_CODER_MODELS, get_model_config, list_models

__all__ = [
    "ZEN_CODER_MODELS",
    "get_model_config",
    "list_models",
]
