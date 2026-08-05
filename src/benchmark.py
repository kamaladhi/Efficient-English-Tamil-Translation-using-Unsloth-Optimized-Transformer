# benchmark.py
"""
FLORES-200 benchmark loader and evaluator.
Provides a standardized, held-out test set for fair evaluation.
FLORES-200 devtest contains 1012 parallel sentences across 200+ languages.
"""

import random
import logging
from datasets import load_dataset

logger = logging.getLogger(__name__)


class FLORESBenchmark:
    """Load and manage FLORES-200 devtest data for Tamil-English evaluation."""

    def __init__(self, sample_size=500):
        self.sample_size = sample_size
        self._pairs = None

    def load_flores(self):
        """
        Load FLORES-200 devtest splits for Tamil and English.
        Sentences are aligned by index across languages.

        Returns:
            List of dicts: [{"tamil": ..., "english": ...}, ...]
        """
        if self._pairs is not None:
            return self._pairs

        print("📥 Loading FLORES-200 devtest benchmark...")

        try:
            flores_ta = load_dataset("facebook/flores", "tam_Taml", split="devtest")
            flores_en = load_dataset("facebook/flores", "eng_Latn", split="devtest")

            self._pairs = [
                {
                    "tamil": flores_ta[i]["sentence"],
                    "english": flores_en[i]["sentence"],
                }
                for i in range(min(len(flores_ta), len(flores_en)))
            ]

            print(f"   ✅ Loaded {len(self._pairs):,} FLORES-200 parallel pairs")
            return self._pairs

        except Exception as e:
            logger.error(f"FLORES-200 loading failed: {e}")
            print(f"   ❌ FLORES-200 failed: {e}")
            print("   ℹ️  Try: pip install datasets")
            return []

    def get_contamination_hashes(self):
        """
        Return a set of hashes for all FLORES-200 sentences.
        Used to exclude these from training data to prevent data leakage.
        """
        pairs = self.load_flores()
        hashes = set()
        for p in pairs:
            hashes.add(hash(p["tamil"].strip().lower()))
            hashes.add(hash(p["english"].strip().lower()))
        print(f"   🔒 Generated {len(hashes):,} contamination hashes for FLORES-200")
        return hashes

    def get_test_pairs(self, sample_size=None):
        """
        Get a subset of FLORES-200 pairs for evaluation.

        Args:
            sample_size: Number of pairs to return. If None, uses self.sample_size.
                         If >= total pairs, returns all.

        Returns:
            List of dicts: [{"tamil": ..., "english": ...}, ...]
        """
        pairs = self.load_flores()
        if not pairs:
            return []

        n = sample_size or self.sample_size
        if n >= len(pairs):
            return pairs

        return random.sample(pairs, n)

    def evaluate(self, evaluator, sample_size=None):
        """
        Run full evaluation on FLORES-200 benchmark in both directions.

        Args:
            evaluator: TamilEvaluator instance with evaluate_pairs() method.
            sample_size: Number of FLORES pairs to evaluate on.

        Returns:
            Dict with 'ta2en' and 'en2ta' evaluation results.
        """
        pairs = self.get_test_pairs(sample_size)
        if not pairs:
            print("   ⚠️  No FLORES-200 data available. Skipping benchmark.")
            return {}

        print(f"\n🏆 FLORES-200 Benchmark Evaluation ({len(pairs)} pairs)")
        print("=" * 60)

        # Tamil → English
        print("\n📌 Direction: Tamil → English")
        ta2en_results = evaluator.evaluate_pairs(pairs, direction="ta2en")

        # English → Tamil
        print("\n📌 Direction: English → Tamil")
        en2ta_results = evaluator.evaluate_pairs(pairs, direction="en2ta")

        return {
            "ta2en": ta2en_results,
            "en2ta": en2ta_results,
            "num_pairs": len(pairs),
        }
