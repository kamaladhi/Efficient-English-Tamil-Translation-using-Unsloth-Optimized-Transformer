# data_loader.py
"""
Data loading pipeline for Tamil-English translation.
v2.0: Loads 100K real parallel pairs from OPUS-100 + Samanantar.
      No synthetic data. Bidirectional formatting. Deduplication.
"""

import re
import logging
from datasets import load_dataset, Dataset

logger = logging.getLogger(__name__)


class TamilDataLoader:
    """Loads, filters, deduplicates, and formats parallel Tamil-English data."""

    def __init__(self, config, flores_hashes=None):
        self.config = config
        self.flores_hashes = flores_hashes or set()

    # ── Data Collection ────────────────────────────────────────────────

    def _is_valid_pair(self, ta, en):
        """Quality filter for a single translation pair."""
        if not ta or not en:
            return False
        len_ta, len_en = len(ta), len(en)

        # Length filter
        if not (self.config.min_length < len_ta < self.config.max_length):
            return False
        if not (self.config.min_length < len_en < self.config.max_length):
            return False

        # Length ratio filter (catches misaligned pairs)
        ratio = max(len_ta, len_en) / max(min(len_ta, len_en), 1)
        if ratio > self.config.max_length_ratio:
            return False

        # Must contain Tamil script
        if not re.search(r'[\u0B80-\u0BFF]', ta):
            return False

        return True

    def _is_contaminated(self, ta, en):
        """Check if a pair overlaps with FLORES-200 benchmark data."""
        ta_hash = hash(ta.strip().lower())
        en_hash = hash(en.strip().lower())
        return ta_hash in self.flores_hashes or en_hash in self.flores_hashes

    def _collect_opus(self):
        """Collect parallel pairs from OPUS-100 (Helsinki-NLP)."""
        pairs = []
        seen = set()
        print(f"📥 Loading OPUS-100 (target: {self.config.opus_samples:,} pairs)...")

        try:
            dataset = load_dataset(
                "Helsinki-NLP/opus-100", "en-ta", split="train", streaming=True
            )
            for item in dataset:
                if len(pairs) >= self.config.opus_samples:
                    break
                if 'translation' not in item:
                    continue

                ta = item['translation'].get('ta', '').strip()
                en = item['translation'].get('en', '').strip()

                if not self._is_valid_pair(ta, en):
                    continue

                # Deduplication
                pair_key = (ta.lower(), en.lower())
                if pair_key in seen:
                    continue
                seen.add(pair_key)

                # FLORES contamination check
                if self._is_contaminated(ta, en):
                    continue

                pairs.append({'tamil': ta, 'english': en})

            print(f"   ✅ Collected {len(pairs):,} pairs from OPUS-100")
        except Exception as e:
            logger.error(f"OPUS-100 loading failed: {e}")
            print(f"   ❌ OPUS-100 failed: {e}")

        return pairs, seen

    def _collect_samanantar(self, seen):
        """Collect parallel pairs from AI4Bharat Samanantar corpus."""
        pairs = []
        print(f"📥 Loading Samanantar (target: {self.config.samanantar_samples:,} pairs)...")

        try:
            dataset = load_dataset(
                "ai4bharat/samanantar", "ta", split="train", streaming=True
            )
            for item in dataset:
                if len(pairs) >= self.config.samanantar_samples:
                    break

                ta = item.get('tgt', '').strip()  # Samanantar uses 'src'(en) / 'tgt'(ta)
                en = item.get('src', '').strip()

                if not self._is_valid_pair(ta, en):
                    continue

                # Cross-source deduplication
                pair_key = (ta.lower(), en.lower())
                if pair_key in seen:
                    continue
                seen.add(pair_key)

                # FLORES contamination check
                if self._is_contaminated(ta, en):
                    continue

                pairs.append({'tamil': ta, 'english': en})

            print(f"   ✅ Collected {len(pairs):,} pairs from Samanantar")
        except Exception as e:
            logger.error(f"Samanantar loading failed: {e}")
            print(f"   ❌ Samanantar failed: {e}")

        return pairs

    def collect_dataset(self):
        """Collect and merge data from all sources."""
        opus_pairs, seen = self._collect_opus()
        samanantar_pairs = self._collect_samanantar(seen)

        all_data = opus_pairs + samanantar_pairs
        print(f"\n📊 Total collected: {len(all_data):,} unique, clean parallel pairs")
        print(f"   OPUS-100:    {len(opus_pairs):,}")
        print(f"   Samanantar:  {len(samanantar_pairs):,}")

        return all_data

    # ── Formatting ─────────────────────────────────────────────────────

    def _format_bidirectional(self, raw_data):
        """
        Format each pair in BOTH directions for bidirectional training.
        Each raw pair produces two training samples.
        """
        ta2en_template = "Translate from Tamil to English.\n\nTamil: {tamil}\nEnglish: {english}"
        en2ta_template = "Translate from English to Tamil.\n\nEnglish: {english}\nTamil: {tamil}"

        formatted_texts = []
        for item in raw_data:
            formatted_texts.append(ta2en_template.format(**item))
            formatted_texts.append(en2ta_template.format(**item))

        return formatted_texts

    # ── Main Pipeline ──────────────────────────────────────────────────

    def prepare_data(self):
        """
        Full data pipeline: collect → filter → deduplicate → format → split.
        Returns (train_dataset, eval_dataset, test_dataset_original).
        """
        raw_data = self.collect_dataset()

        # Format bidirectionally
        formatted_texts = self._format_bidirectional(raw_data)
        dataset = Dataset.from_dict({"text": formatted_texts})
        print(f"\n📝 Formatted {len(dataset):,} training samples (bidirectional)")

        # Keep original pairs for evaluation (unformatted)
        original_dataset = Dataset.from_dict({
            'tamil': [item['tamil'] for item in raw_data],
            'english': [item['english'] for item in raw_data],
        })

        # Split formatted data for training
        split = dataset.train_test_split(
            test_size=1 - self.config.train_split, seed=42
        )
        train_dataset = split["train"]
        eval_dataset = split["test"]

        # Split original pairs for testing (raw format, for metric computation)
        original_split = original_dataset.train_test_split(test_size=0.1, seed=42)
        test_dataset_original = original_split["test"]

        print(f"📊 Train: {len(train_dataset):,} | Eval: {len(eval_dataset):,} | "
              f"Test (original pairs): {len(test_dataset_original):,}")

        return train_dataset, eval_dataset, test_dataset_original


def load_tamil_data(config, flores_hashes=None):
    """Convenience function to load and prepare all data."""
    loader = TamilDataLoader(config, flores_hashes=flores_hashes)
    return loader.prepare_data()