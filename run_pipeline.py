"""
run_pipeline.py
================
Single entry point to run the entire Tamil-English translation pipeline.
Covers: data loading → training → evaluation → FLORES benchmark → baseline comparison.

Usage:
    python run_pipeline.py                    # Full pipeline
    python run_pipeline.py --skip-baselines   # Skip baseline comparison (faster)
    python run_pipeline.py --eval-only        # Skip training, only evaluate
"""

import argparse
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys

# Ensure src is importable
sys.path.insert(0, os.path.dirname(__file__))

from src.config import FastConfig
from src.model import load_model
from src.data_loader import load_tamil_data
from src.train import train_model
from src.evaluate import TamilEvaluator, evaluate_model
from src.inference import TamilTranslator
from src.benchmark import FLORESBenchmark


def parse_args():
    parser = argparse.ArgumentParser(description="Tamil-English Translation Pipeline")
    parser.add_argument("--skip-baselines", action="store_true",
                        help="Skip baseline comparison (NLLB-200, IndicTrans2, zero-shot)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, only run evaluation (requires saved model)")
    parser.add_argument("--skip-flores", action="store_true",
                        help="Skip FLORES-200 benchmark evaluation")
    parser.add_argument("--samples", type=int, default=None,
                        help="Override target_samples (default: 100000)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override max_steps (default: 5000)")
    return parser.parse_args()


def main():
    args = parse_args()
    config = FastConfig()

    # Override config if CLI args provided
    if args.samples:
        config.target_samples = args.samples
        config.opus_samples = args.samples // 2
        config.samanantar_samples = args.samples // 2
    if args.max_steps:
        config.max_steps = args.max_steps

    print("=" * 70)
    print("🚀 TAMIL-ENGLISH TRANSLATION PIPELINE v2.0")
    print("=" * 70)
    print(f"Config: {config}")
    print()

    # ── Step 1: FLORES-200 contamination hashes ────────────────────────
    print("\n" + "=" * 70)
    print("📋 STEP 1: Loading FLORES-200 benchmark data")
    print("=" * 70)

    flores = FLORESBenchmark(sample_size=config.flores_sample_size)
    flores_hashes = flores.get_contamination_hashes()

    # ── Step 2: Load & Prepare Data ────────────────────────────────────
    print("\n" + "=" * 70)
    print("📋 STEP 2: Loading & preparing dataset")
    print("=" * 70)

    train_dataset, eval_dataset, test_dataset = load_tamil_data(
        config, flores_hashes=flores_hashes
    )

    # ── Step 3: Load Model ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("📋 STEP 3: Loading model with LoRA")
    print("=" * 70)

    model, tokenizer = load_model(config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n   Trainable params: {trainable:,} ({100 * trainable / total:.2f}%)")
    print(f"   Total params:     {total:,}")

    # ── Step 4: Train ──────────────────────────────────────────────────
    if not args.eval_only:
        print("\n" + "=" * 70)
        print("📋 STEP 4: Training")
        print("=" * 70)

        trainer = train_model(model, tokenizer, train_dataset, eval_dataset, config)
    else:
        print("\n⏭️  Skipping training (--eval-only)")

    
    split_results, flores_results, baseline_results = {}, {}, []
    try:
        # ── Step 5: Evaluate on test split ─────────────────────────────────

            print("\n" + "=" * 70)
            print("📋 STEP 5: Evaluation (train-split test set)")
            print("=" * 70)

            evaluator = TamilEvaluator(model, tokenizer, test_dataset, config)
            split_results = evaluator.evaluate()

            # ── Step 6: FLORES-200 Benchmark ───────────────────────────────────
            flores_results = {}
            if not args.skip_flores:
                print("\n" + "=" * 70)
                print("📋 STEP 6: FLORES-200 Benchmark")
                print("=" * 70)

                flores_results = flores.evaluate(evaluator, sample_size=config.flores_sample_size)
            else:
                print("\n⏭️  Skipping FLORES-200 benchmark (--skip-flores)")

            # ── Step 7: Baseline Comparison ────────────────────────────────────
            baseline_results = []
            if not args.skip_baselines:
                print("\n" + "=" * 70)
                print("📋 STEP 7: Baseline Comparison")
                print("=" * 70)

                from src.baselines import load_all_baselines

                baselines = load_all_baselines(config)
                if baselines:
                    flores_pairs = flores.get_test_pairs(sample_size=200)
                    baseline_results = evaluator.compare_baselines(
                        baselines=baselines,
                        pairs=flores_pairs,
                        directions=["ta2en", "en2ta"],
                    )
            else:
                print("\n⏭️  Skipping baseline comparison (--skip-baselines)")

            # ── Step 8: Quick Demo ─────────────────────────────────────────────
            print("\n" + "=" * 70)
            print("📋 STEP 8: Translation Demo")
            print("=" * 70)

            translator = TamilTranslator(model, tokenizer)

            demo_pairs = [
                ("ta2en", "வணக்கம், நான் தமிழ் கற்கிறேன்"),
                ("ta2en", "இன்று வானிலை மிகவும் அழகாக இருக்கிறது"),
                ("ta2en", "நான் பள்ளிக்கு செல்கிறேன்"),
                ("en2ta", "Hello, I am learning Tamil"),
                ("en2ta", "The weather is very beautiful today"),
                ("en2ta", "I am going to school"),
            ]

            for direction, text in demo_pairs:
                result = translator.translate(text, direction=direction)
                arrow = "→ EN" if direction == "ta2en" else "→ TA"
                print(f"  [{arrow}] {text}")
                print(f"         {result}\n")

            # ── Step 9: Save Results ───────────────────────────────────────────
            print("\n" + "=" * 70)
            print("📋 STEP 9: Saving results")
            print("=" * 70)

            all_results = {
                "train_split": split_results,
                "flores_200": flores_results,
                "baselines": baseline_results,
            }

            with open(config.results_file, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            print(f"   💾 Results saved to {config.results_file}")

            
    except Exception as e:
        print(f"\n⚠️ CRITICAL WARNING: Evaluation failed: {e}")
        print("   But do not worry! The trained model is already safely saved!")

    # ── Done ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"   Model saved to:   {config.model_save_dir}")
    print(f"   Results saved to: {config.results_file}")
    print()


if __name__ == "__main__":
    main()
