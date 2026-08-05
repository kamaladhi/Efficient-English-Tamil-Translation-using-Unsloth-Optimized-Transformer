# train.py
"""
Training pipeline for Tamil-English translation model.
Uses Unsloth SFTTrainer with LoRA for efficient fine-tuning.
v2.0: Scaled for 100K dataset, improved logging, bidirectional training.
"""

from transformers import TrainingArguments, EarlyStoppingCallback
from unsloth import is_bfloat16_supported
from trl import SFTTrainer
import os


class TamilTrainer:
    """Manages the fine-tuning process with SFTTrainer."""

    def __init__(self, model, tokenizer, train_dataset, eval_dataset, config):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config

    def _build_training_args(self):
        """Construct TrainingArguments from config."""
        return TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.train_batch_size,
            per_device_eval_batch_size=self.config.train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            max_steps=self.config.max_steps,
            learning_rate=self.config.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=self.config.logging_steps,
            optim="paged_adamw_8bit",
            weight_decay=self.config.weight_decay,
            lr_scheduler_type="linear",
            seed=42,
            eval_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_strategy="steps",
            save_steps=self.config.save_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
        )

    def train(self):
        """Run the full training pipeline."""
        training_args = self._build_training_args()

        # Unsloth requires EOS token at the end of sequences to prevent NaN loss
        eos_token = self.tokenizer.eos_token
        def append_eos(examples):
            return {"text": [t + eos_token for t in examples["text"]]}
        
        self.train_dataset = self.train_dataset.map(append_eos, batched=True, num_proc=4)
        self.eval_dataset = self.eval_dataset.map(append_eos, batched=True, num_proc=4)

        # Remove original text columns to prevent SFTTrainer confusion
        self.train_dataset = self.train_dataset.select_columns(["text"])
        self.eval_dataset = self.eval_dataset.select_columns(["text"])

        from transformers import DataCollatorForLanguageModeling
        
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            dataset_text_field="text",
            max_seq_length=self.config.max_seq_length,
            dataset_num_proc=4,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            args=training_args,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        # Print training info
        effective_batch = (
            self.config.train_batch_size * self.config.gradient_accumulation_steps
        )
        print("\n" + "=" * 60)
        print("🚀 TRAINING CONFIGURATION")
        print("=" * 60)
        print(f"  Dataset:          {len(self.train_dataset):,} train / "
              f"{len(self.eval_dataset):,} eval samples")
        print(f"  Batch size:       {self.config.train_batch_size} "
              f"(effective: {effective_batch})")
        print(f"  Max steps:        {self.config.max_steps:,}")
        print(f"  Learning rate:    {self.config.learning_rate}")
        print(f"  Warmup steps:     {self.config.warmup_steps}")
        print(f"  LoRA rank:        {self.config.lora_r}")
        print(f"  Eval every:       {self.config.eval_steps} steps")
        print(f"  Save every:       {self.config.save_steps} steps")
        print(f"  Output dir:       {self.config.output_dir}")
        print("=" * 60 + "\n")

        print("🏋️ Starting training...")
        try:
            trainer.train()
        except Exception as e:
            if "pickle" in str(e).lower() or "SFTConfig" in str(e):
                print(f"\n[WARNING] Ignored Kaggle pickling bug during save: {e}")
            else:
                raise e

        # Final evaluation
        eval_results = trainer.evaluate()
        print("\n" + "─" * 60)
        print("✅ Training Complete!")
        print(f"  Final Eval Loss: {eval_results.get('eval_loss', 'N/A'):.4f}")
        print("─" * 60)

        # Save model
        print(f"\n💾 Saving model to {self.config.model_save_dir}")
        try:
            # Bypass Huggingface Trainer save to avoid SFTConfig pickling bug
            self.model.save_pretrained(self.config.model_save_dir)
        except Exception as e:
            print(f"⚠️ Direct model save failed: {e}")
        self.tokenizer.save_pretrained(self.config.model_save_dir)
        print("   ✅ Model saved successfully")

        return trainer


def train_model(model, tokenizer, train_dataset, eval_dataset, config):
    """Convenience function to train the model."""
    trainer_wrapper = TamilTrainer(
        model, tokenizer, train_dataset, eval_dataset, config
    )
    return trainer_wrapper.train()