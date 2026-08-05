# inference.py
"""
Inference module for Tamil-English translation.
v2.0: Bidirectional translation with auto-detection, dynamic device, robust parsing.
"""

import re
import torch
from unsloth import FastLanguageModel


class TamilTranslator:
    """Handles bidirectional Tamil ↔ English translation at inference time."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        FastLanguageModel.for_inference(self.model)

    def detect_direction(self, text):
        """
        Auto-detect translation direction based on script.
        If Tamil characters are found → Tamil-to-English, else English-to-Tamil.
        """
        if re.search(r'[\u0B80-\u0BFF]', text):
            return "ta2en"
        return "en2ta"

    def _build_prompt(self, text, direction):
        """Build the instruction prompt for the given direction."""
        if direction == "ta2en":
            return f"Translate from Tamil to English.\n\nTamil: {text}\nEnglish:"
        else:
            return f"Translate from English to Tamil.\n\nEnglish: {text}\nTamil:"

    def _parse_output(self, result, direction):
        """
        Robustly extract the translation from model output.
        Handles cases where the model generates extra content after the translation.
        """
        split_key = "English:" if direction == "ta2en" else "Tamil:"

        if split_key in result:
            # Take the last occurrence of the key and extract the first line after it
            parts = result.split(split_key)
            if len(parts) >= 2:
                translation = parts[-1].strip()
                # Take only the first line (stop at newline or next instruction)
                translation = translation.split("\n")[0].strip()
                # Remove any trailing prompt artifacts
                for artifact in ["Translate from", "Tamil:", "English:"]:
                    if artifact in translation:
                        translation = translation.split(artifact)[0].strip()
                return translation

        # Fallback: return the raw output stripped
        return result.strip()

    def translate(self, text, direction=None):
        """
        Translate text between Tamil and English.

        Args:
            text: Input text to translate.
            direction: "ta2en" or "en2ta". If None, auto-detected from script.

        Returns:
            Translated text as a string.
        """
        if not text or not text.strip():
            return ""

        if direction is None:
            direction = self.detect_direction(text)

        prompt = self._build_prompt(text, direction)

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.1,
                do_sample=False,
            )

        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._parse_output(result, direction)

    def translate_batch(self, texts, direction=None, batch_size=8):
        """
        Translate a batch of texts.

        Args:
            texts: List of input texts.
            direction: Translation direction (applied to all). If None, auto-detected per text.
            batch_size: Number of texts to process at once.

        Returns:
            List of translated texts.
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            for text in batch:
                results.append(self.translate(text, direction=direction))
        return results


def translate(model, tokenizer, text, direction=None):
    """Convenience function for single translation."""
    translator = TamilTranslator(model, tokenizer)
    return translator.translate(text, direction=direction)