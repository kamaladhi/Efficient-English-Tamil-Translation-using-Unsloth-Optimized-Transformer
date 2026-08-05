# evaluate.py
"""
Comprehensive evaluation module for Tamil-English translation.
v2.0: BLEU + chrF++ + COMET + Sentence-BERT, bidirectional, baseline comparison.
"""

import numpy as np
import random
import logging
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sacrebleu.metrics import BLEU, CHRF
from .inference import TamilTranslator

logger = logging.getLogger(__name__)


class TamilEvaluator:
    """
    Evaluates translation quality using multiple metrics:
    - BLEU (corpus-level, n-gram overlap)
    - chrF++ (character-level, better for morphologically rich languages)
    - COMET (neural, state-of-the-art MT metric)
    - Sentence-BERT similarity (semantic similarity)
    """

    def __init__(self, model, tokenizer, test_dataset, config):
        self.translator = TamilTranslator(model, tokenizer)
        self.test_dataset = test_dataset
        self.config = config

        # Metrics
        self.bleu_metric = BLEU()
        self.chrf_metric = CHRF(word_order=2)  # chrF++ (with bigram word order)
        self.sentence_model = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2"
        )

        # COMET (loaded lazily — it's a large model)
        self._comet_model = None

    def _load_comet(self):
        """Lazy-load COMET model (downloads on first use)."""
        if self._comet_model is None:
            try:
                from comet import download_model, load_from_checkpoint

                print("📥 Loading COMET model (first time may take a while)...")
                model_path = download_model("Unbabel/wmt22-comet-da")
                self._comet_model = load_from_checkpoint(model_path)
                print("   ✅ COMET model loaded")
            except Exception as e:
                logger.warning(f"COMET loading failed: {e}")
                print(f"   ⚠️  COMET unavailable: {e}")
                print("   ℹ️  Install with: pip install unbabel-comet")
                self._comet_model = None
        return self._comet_model

    # ── Metric Computation ─────────────────────────────────────────────

    def compute_similarity(self, text1, text2):
        """Compute cosine similarity using Sentence-BERT embeddings."""
        emb1 = self.sentence_model.encode(text1, convert_to_tensor=True).cpu().numpy()
        emb2 = self.sentence_model.encode(text2, convert_to_tensor=True).cpu().numpy()
        cos_sim = np.dot(emb1, emb2) / (
            np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8
        )
        return float(cos_sim)

    def compute_comet(self, sources, predictions, references):
        """
        Compute COMET score (neural MT metric).
        Takes source, hypothesis, and reference — more informative than BLEU.
        """
        comet_model = self._load_comet()
        if comet_model is None:
            return None

        comet_data = [
            {"src": s, "mt": p, "ref": r}
            for s, p, r in zip(sources, predictions, references)
        ]

        try:
            comet_output = comet_model.predict(comet_data, batch_size=8, gpus=1)
            return comet_output.system_score
        except Exception as e:
            logger.warning(f"COMET computation failed: {e}")
            # Try CPU fallback
            try:
                comet_output = comet_model.predict(comet_data, batch_size=4, gpus=0)
                return comet_output.system_score
            except Exception as e2:
                logger.warning(f"COMET CPU fallback also failed: {e2}")
                return None

    # ── Core Evaluation ────────────────────────────────────────────────

    def evaluate_pairs(self, pairs, direction="ta2en"):
        """
        Evaluate translation on a list of parallel pairs in a given direction.

        Args:
            pairs: List of dicts with 'tamil' and 'english' keys.
            direction: "ta2en" or "en2ta"

        Returns:
            Dict with all metric scores.
        """
        similarities = []
        high_quality = 0
        semantic_matches = 0
        exact_matches = 0

        sources = []
        predictions = []
        references = []

        for pair in tqdm(pairs, desc=f"Evaluating ({direction})"):
            if direction == "ta2en":
                source_text = pair["tamil"]
                reference = pair["english"]
            else:
                source_text = pair["english"]
                reference = pair["tamil"]

            predicted = self.translator.translate(source_text, direction=direction)

            sim = self.compute_similarity(predicted, reference)
            similarities.append(sim)

            if sim > 0.7:
                high_quality += 1
            if sim > 0.5:
                semantic_matches += 1
            if predicted.lower().strip() == reference.lower().strip():
                exact_matches += 1

            sources.append(source_text)
            predictions.append(predicted)
            references.append(reference)

        n = len(similarities)
        if n == 0:
            return {"error": "No samples evaluated"}

        # BLEU (corpus-level)
        bleu_refs = [[r] for r in references]  # sacrebleu expects list of lists
        bleu_result = self.bleu_metric.corpus_score(predictions, bleu_refs)

        # chrF++ (character-level, better for Tamil)
        chrf_result = self.chrf_metric.corpus_score(predictions, bleu_refs)

        # COMET (neural)
        comet_score = self.compute_comet(sources, predictions, references)

        # Sentence-BERT statistics
        stats = {
            "direction": direction,
            "num_samples": n,
            # BLEU
            "bleu": bleu_result.score,
            "bleu_1": bleu_result.precisions[0] if bleu_result.precisions else None,
            "bleu_2": bleu_result.precisions[1] if len(bleu_result.precisions) > 1 else None,
            "bleu_3": bleu_result.precisions[2] if len(bleu_result.precisions) > 2 else None,
            "bleu_4": bleu_result.precisions[3] if len(bleu_result.precisions) > 3 else None,
            "brevity_penalty": bleu_result.bp,
            # chrF++
            "chrf": chrf_result.score,
            # COMET
            "comet": comet_score,
            # Sentence-BERT similarity
            "sbert_mean": float(np.mean(similarities)),
            "sbert_median": float(np.median(similarities)),
            "sbert_std": float(np.std(similarities)),
            "sbert_min": float(np.min(similarities)),
            "sbert_max": float(np.max(similarities)),
            # Quality buckets
            "excellent_pct": (high_quality / n) * 100,
            "good_pct": (semantic_matches / n) * 100,
            "exact_match_pct": (exact_matches / n) * 100,
        }

        # Combined score (40% BLEU + 30% COMET + 30% Sentence-BERT)
        if comet_score is not None:
            stats["combined_score"] = (
                0.4 * (stats["bleu"] / 100)    # Normalize BLEU to 0-1
                + 0.3 * comet_score             # COMET is already 0-1
                + 0.3 * stats["sbert_mean"]     # SBERT is already 0-1
            ) * 100  # Scale back to 0-100
        else:
            stats["combined_score"] = (
                0.4 * (stats["bleu"] / 100)
                + 0.6 * stats["sbert_mean"]
            ) * 100

        self._print_results(stats)
        return stats

    def evaluate(self):
        """
        Run evaluation on the test dataset in both directions.

        Returns:
            Dict with 'ta2en' and 'en2ta' results.
        """
        print(f"\n🔍 Evaluating on {self.config.test_sample_size} test samples...")

        # Select random samples
        if len(self.test_dataset) > self.config.test_sample_size:
            indices = random.sample(
                range(len(self.test_dataset)), self.config.test_sample_size
            )
            pairs = [self.test_dataset[i] for i in indices]
        else:
            pairs = [self.test_dataset[i] for i in range(len(self.test_dataset))]

        # Evaluate both directions
        print("\n" + "=" * 70)
        print("📌 TAMIL → ENGLISH")
        print("=" * 70)
        ta2en_results = self.evaluate_pairs(pairs, direction="ta2en")

        print("\n" + "=" * 70)
        print("📌 ENGLISH → TAMIL")
        print("=" * 70)
        en2ta_results = self.evaluate_pairs(pairs, direction="en2ta")

        return {"ta2en": ta2en_results, "en2ta": en2ta_results}

    # ── Baseline Comparison ────────────────────────────────────────────

    def evaluate_baseline(self, baseline, pairs, direction="ta2en"):
        """
        Evaluate a single baseline model on the given pairs.

        Args:
            baseline: A BaselineTranslator instance with translate(text, direction).
            pairs: List of dicts with 'tamil' and 'english' keys.
            direction: "ta2en" or "en2ta"

        Returns:
            Dict with metric scores for this baseline.
        """
        similarities = []
        sources = []
        predictions = []
        references = []

        for pair in tqdm(pairs, desc=f"{baseline.name} ({direction})"):
            if direction == "ta2en":
                source_text = pair["tamil"]
                reference = pair["english"]
            else:
                source_text = pair["english"]
                reference = pair["tamil"]

            predicted = baseline.translate(source_text, direction=direction)

            sim = self.compute_similarity(predicted, reference)
            similarities.append(sim)
            sources.append(source_text)
            predictions.append(predicted)
            references.append(reference)

        n = len(similarities)
        if n == 0:
            return {"error": "No samples evaluated"}

        bleu_refs = [[r] for r in references]
        bleu_result = self.bleu_metric.corpus_score(predictions, bleu_refs)
        chrf_result = self.chrf_metric.corpus_score(predictions, bleu_refs)
        comet_score = self.compute_comet(sources, predictions, references)

        return {
            "model": baseline.name,
            "direction": direction,
            "bleu": bleu_result.score,
            "chrf": chrf_result.score,
            "comet": comet_score,
            "sbert_mean": float(np.mean(similarities)),
        }

    def compare_baselines(self, baselines, pairs, directions=None):
        """
        Compare your model against all baselines on the same test data.

        Args:
            baselines: List of BaselineTranslator instances.
            pairs: Test pairs to evaluate on.
            directions: List of directions (default: both).

        Returns:
            List of result dicts for tabular display.
        """
        if directions is None:
            directions = ["ta2en", "en2ta"]

        all_results = []

        # Evaluate your fine-tuned model
        for direction in directions:
            result = self.evaluate_pairs(pairs, direction=direction)
            result["model"] = "Your Model (LoRA fine-tuned)"
            all_results.append(result)

        # Evaluate each baseline
        for baseline in baselines:
            for direction in directions:
                try:
                    result = self.evaluate_baseline(baseline, pairs, direction)
                    all_results.append(result)
                except Exception as e:
                    logger.warning(f"Baseline {baseline.name} failed on {direction}: {e}")
                    print(f"⚠️  {baseline.name} ({direction}) failed: {e}")

        self._print_comparison_table(all_results)
        return all_results

    # ── Pretty Printing ────────────────────────────────────────────────

    def _print_results(self, stats):
        """Print evaluation results in a clean format."""
        direction = stats.get("direction", "unknown")
        print(f"\n{'─' * 60}")
        print(f"  Direction:        {direction}")
        print(f"  Samples:          {stats['num_samples']}")
        print(f"{'─' * 60}")
        print(f"  BLEU:             {stats['bleu']:.2f}")
        print(f"  chrF++:           {stats['chrf']:.2f}")
        comet_str = f"{stats['comet']:.4f}" if stats['comet'] is not None else "N/A"
        print(f"  COMET:            {comet_str}")
        print(f"  Sentence-BERT:    {stats['sbert_mean']:.4f} "
              f"(median: {stats['sbert_median']:.4f}, std: {stats['sbert_std']:.4f})")
        print(f"{'─' * 60}")
        print(f"  Excellent (>0.7): {stats['excellent_pct']:.1f}%")
        print(f"  Good (>0.5):      {stats['good_pct']:.1f}%")
        print(f"  Exact Matches:    {stats['exact_match_pct']:.1f}%")
        print(f"{'─' * 60}")
        print(f"  Combined Score:   {stats['combined_score']:.2f}")
        print(f"{'─' * 60}\n")

    def _print_comparison_table(self, results):
        """Print a comparison table of all models."""
        print("\n" + "=" * 80)
        print("📊 MODEL COMPARISON TABLE")
        print("=" * 80)
        header = f"{'Model':<35} {'Dir':<6} {'BLEU':>7} {'chrF++':>7} {'COMET':>7} {'SBERT':>7}"
        print(header)
        print("─" * 80)
        for r in results:
            comet_str = f"{r['comet']:.3f}" if r.get('comet') is not None else "  N/A"
            line = (
                f"{r.get('model', 'Unknown'):<35} "
                f"{r.get('direction', '??'):<6} "
                f"{r.get('bleu', 0):>7.2f} "
                f"{r.get('chrf', 0):>7.2f} "
                f"{comet_str:>7} "
                f"{r.get('sbert_mean', 0):>7.4f}"
            )
            print(line)
        print("=" * 80)


def evaluate_model(model, tokenizer, test_dataset, config):
    """Convenience function to run full evaluation."""
    evaluator = TamilEvaluator(model, tokenizer, test_dataset, config)
    return evaluator.evaluate()