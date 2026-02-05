"""
Zen Coder Model Configurations

Correct model mappings:
- zen-coder: Devstral 2 (24B dense)
- zen-coder-flash: GLM-4.7-Flash (31B MoE, 3B active) - FLAGSHIP
- zen-coder-max: Kimi K2 (671B MoE, 14B active)
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class ModelConfig:
    name: str
    hf_id: str
    base_model: str
    size_total: str
    size_active: str
    architecture: str
    vram_qlora: int
    context_window: int
    license: str
    lora_r: int
    lora_alpha: int
    batch_size: int
    learning_rate: float
    supports_mlx: bool
    supports_unsloth: bool


ZEN_CODER_MODELS: Dict[str, ModelConfig] = {
    # Edge model
    "zen-coder-4b": ModelConfig(
        name="Zen Coder 4B",
        hf_id="zenlm/zen-coder",
        base_model="Qwen/Qwen3-Coder-4B",
        size_total="4B",
        size_active="4B",
        architecture="dense",
        vram_qlora=8,
        context_window=32768,
        license="Apache 2.0",
        lora_r=64,
        lora_alpha=128,
        batch_size=4,
        learning_rate=2e-4,
        supports_mlx=True,
        supports_unsloth=True,
    ),

    # Main production model
    "zen-coder": ModelConfig(
        name="Zen Coder",
        hf_id="zenlm/zen-coder-24b",
        base_model="mistralai/Devstral-Small-2-24B-Instruct-2512",
        size_total="24B",
        size_active="24B",
        architecture="dense",
        vram_qlora=24,
        context_window=262144,
        license="Apache 2.0",
        lora_r=32,
        lora_alpha=64,
        batch_size=2,
        learning_rate=1e-4,
        supports_mlx=True,
        supports_unsloth=True,
    ),

    # FLAGSHIP - Best balance of performance and efficiency
    "zen-coder-flash": ModelConfig(
        name="Zen Coder Flash",
        hf_id="zenlm/zen-coder-flash",
        base_model="zai-org/GLM-4.7-Flash",
        size_total="31B MoE",
        size_active="3B",
        architecture="moe",
        vram_qlora=24,
        context_window=131072,
        license="MIT",
        lora_r=64,
        lora_alpha=128,
        batch_size=2,
        learning_rate=1e-4,
        supports_mlx=True,
        supports_unsloth=True,
    ),

    # Frontier model
    "zen-coder-max": ModelConfig(
        name="Zen Coder Max",
        hf_id="zenlm/zen-max",
        base_model="moonshotai/Kimi-K2-Instruct",
        size_total="671B MoE",
        size_active="14B",
        architecture="moe",
        vram_qlora=160,
        context_window=262144,
        license="Apache 2.0",
        lora_r=16,
        lora_alpha=32,
        batch_size=1,
        learning_rate=5e-5,
        supports_mlx=False,
        supports_unsloth=False,
    ),
}


def get_model_config(model_key: str) -> ModelConfig:
    if model_key not in ZEN_CODER_MODELS:
        available = ", ".join(ZEN_CODER_MODELS.keys())
        raise ValueError(f"Unknown model: {model_key}. Available: {available}")
    return ZEN_CODER_MODELS[model_key]


def list_models():
    """Print model summary."""
    print("Zen Coder Model Family:")
    print("-" * 70)
    for key, cfg in ZEN_CODER_MODELS.items():
        flag = " ⭐" if key == "zen-coder-flash" else ""
        print(f"{key}{flag}")
        print(f"  Base: {cfg.base_model}")
        print(f"  Size: {cfg.size_total} ({cfg.size_active} active)")
        print(f"  Context: {cfg.context_window:,} tokens")
        print()


if __name__ == "__main__":
    list_models()
