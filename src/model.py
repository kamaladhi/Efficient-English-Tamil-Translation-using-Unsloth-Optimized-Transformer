# model.py
"""
Model loading module for Tamil-English translation.
Loads Llama 3.1 8B (4-bit) with Unsloth and applies LoRA adapters.
"""

import torch
from unsloth import FastLanguageModel


class TamilTranslationModel:
    """Handles loading the base model and applying LoRA for fine-tuning."""

    @staticmethod
    def get_device():
        """Detect available compute device."""
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @staticmethod
    def from_pretrained(config):
        """Load the base model and apply LoRA adapters via Unsloth."""
        device = TamilTranslationModel.get_device()
        print(f"🔧 Loading model: {config.model_name} (device: {device})")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model_name,
            max_seq_length=config.max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora_r,
            target_modules=config.target_modules,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

        print(f"   ✅ Model loaded with LoRA (r={config.lora_r}, "
              f"alpha={config.lora_alpha}, dropout={config.lora_dropout})")

        return model, tokenizer

    @staticmethod
    def load_base_model(config):
        """Load the base model WITHOUT LoRA (for zero-shot baseline)."""
        print(f"🔧 Loading base model (no LoRA): {config.model_name}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model_name,
            max_seq_length=config.max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        return model, tokenizer


def load_model(config):
    """Convenience function to load model with LoRA."""
    return TamilTranslationModel.from_pretrained(config)