"""
Configuration module for Tamil-English translation model.
Contains all hyperparameters and training settings.
Updated for v2.0: 100K dataset, bidirectional translation, FLORES-200 benchmark.
"""


class FastConfig:
    """Configuration class for model training and inference."""

    # ── Model ──────────────────────────────────────────────────────────
    model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
    max_seq_length = 512

    # ── Dataset ────────────────────────────────────────────────────────
    target_samples = 100_000        # Total real parallel pairs (was 2000)
    opus_samples = 50_000           # Pairs from OPUS-100
    samanantar_samples = 50_000     # Pairs from AI4Bharat Samanantar
    train_split = 0.99

    # Data quality filters
    min_length = 5                  # Min chars per sentence
    max_length = 300                # Max chars per sentence
    max_length_ratio = 3.0          # Max ratio between source/target lengths

    # ── Training ───────────────────────────────────────────────────────
    train_batch_size = 2
    gradient_accumulation_steps = 8     # Effective batch = 2 * 8 = 16
    learning_rate = 2e-5                # Lower LR for larger dataset (was 1.5e-4)
    max_steps = 1500                    # Scaled for 100K data (was 600)
    eval_steps = 1500                   # (was 100)
    save_steps = 1500                   # (was 200)
    logging_steps = 50                  # (was 25)
    warmup_steps = 300                  # (was 100)
    weight_decay = 0.01

    # ── LoRA ───────────────────────────────────────────────────────────
    lora_r = 64
    lora_alpha = 32
    lora_dropout = 0
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    # ── Evaluation ─────────────────────────────────────────────────────
    test_sample_size = 200              # Random samples from train-split test set
    flores_benchmark = True             # Evaluate on FLORES-200 devtest
    flores_sample_size = 500            # Subset of FLORES-200 for faster eval

    # ── Paths ──────────────────────────────────────────────────────────
    output_dir = "./tamil-translation-fast"
    model_save_dir = "./tamil-translation-model"
    results_file = "./evaluation_results.txt"

    def __repr__(self):
        return (
            f"FastConfig(samples={self.target_samples}, "
            f"opus={self.opus_samples}, samanantar={self.samanantar_samples}, "
            f"steps={self.max_steps}, lr={self.learning_rate}, "
            f"batch_size={self.train_batch_size})"
        )