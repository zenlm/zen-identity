"""
Zen Identity Training - HuggingFace Spaces
Fine-tune Zen models with identity data.
"""

import os
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# Zen Coder Model Family - CORRECT MAPPINGS
MODELS = {
    "zen-coder-4b": {
        "base": "Qwen/Qwen3-Coder-4B",
        "size": "4B",
        "arch": "dense",
        "context": "32K",
    },
    "zen-coder": {
        "base": "mistralai/Devstral-Small-2-24B-Instruct-2512",
        "size": "24B",
        "arch": "dense",
        "context": "256K",
    },
    "zen-coder-flash": {
        "base": "zai-org/GLM-4.7-Flash",
        "size": "31B MoE (3B active)",
        "arch": "moe",
        "context": "131K",
        "flagship": True,
    },
    "zen-coder-max": {
        "base": "moonshotai/Kimi-K2-Instruct",
        "size": "671B MoE (14B active)",
        "arch": "moe",
        "context": "256K",
    },
}

OUTPUT_DIR = "./zen-identity-lora"


def train_model(model_key: str, lr: float, epochs: int, batch_size: int, lora_rank: int, progress=gr.Progress()):
    if model_key not in MODELS:
        return f"Unknown model: {model_key}"

    cfg = MODELS[model_key]
    base_model = cfg["base"]

    progress(0.1, desc=f"Loading {model_key}...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        return "No GPU available. Use a GPU Space."

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    progress(0.3, desc="Setting up LoRA...")

    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    progress(0.4, desc="Loading identity data...")

    # Load identity dataset from this repo
    dataset = load_dataset("zenlm/zen-identity", split="train")

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")

    tokenized = dataset.map(tokenize, batched=True)

    progress(0.5, desc="Training...")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        logging_steps=10,
        save_steps=100,
        bf16=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
    )

    trainer.train()

    progress(0.9, desc="Saving...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    return f"Done. Saved to {OUTPUT_DIR}"


with gr.Blocks(title="Zen Identity Training") as demo:
    gr.Markdown("""
    # Zen Identity Training

    Fine-tune Zen models with identity data.

    | Model | Base | Size |
    |-------|------|------|
    | zen-coder-4b | Qwen3-Coder-4B | 4B |
    | zen-coder | Devstral-Small-2-24B | 24B |
    | **zen-coder-flash** | **GLM-4.7-Flash** | **31B MoE** ⭐ |
    | zen-coder-max | Kimi-K2 | 671B MoE |
    """)

    with gr.Row():
        model_select = gr.Dropdown(
            choices=list(MODELS.keys()),
            value="zen-coder-flash",
            label="Model"
        )

    with gr.Row():
        lr = gr.Slider(1e-5, 1e-3, value=1e-4, label="Learning Rate")
        epochs = gr.Slider(1, 10, value=3, step=1, label="Epochs")

    with gr.Row():
        batch = gr.Slider(1, 4, value=1, step=1, label="Batch Size")
        rank = gr.Slider(8, 64, value=16, step=8, label="LoRA Rank")

    train_btn = gr.Button("Train", variant="primary")
    output = gr.Textbox(label="Status", lines=3)

    train_btn.click(train_model, [model_select, lr, epochs, batch, rank], output)


if __name__ == "__main__":
    demo.launch()
