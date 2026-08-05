# baselines.py
"""
Baseline translation models for comparison.
Implements NLLB-200, IndicTrans2, and zero-shot Llama 3.1 baselines.
All baselines share a common interface: translate(text, direction).
"""

import torch
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaselineTranslator(ABC):
    """Abstract base class for all baseline translators."""

    @abstractmethod
    def translate(self, text, direction="ta2en"):
        """Translate text in the given direction."""
        pass

    @abstractmethod
    def name(self):
        """Return the display name of this baseline."""
        pass


class NLLBBaseline(BaselineTranslator):
    """
    Meta's NLLB-200 (No Language Left Behind) multilingual translation model.
    Uses the distilled 600M parameter variant for efficiency.
    Supports 200+ languages including Tamil.
    """

    LANG_CODES = {
        "ta": "tam_Taml",
        "en": "eng_Latn",
    }

    def __init__(self, model_name="facebook/nllb-200-distilled-600M"):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"📥 Loading NLLB-200 baseline: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print(f"   ✅ NLLB-200 loaded on {self.device}")

    @property
    def name(self):
        return "NLLB-200-distilled-600M"

    def translate(self, text, direction="ta2en"):
        """Translate using NLLB-200."""
        if direction == "ta2en":
            src_lang, tgt_lang = self.LANG_CODES["ta"], self.LANG_CODES["en"]
        else:
            src_lang, tgt_lang = self.LANG_CODES["en"], self.LANG_CODES["ta"]

        self.tokenizer.src_lang = src_lang
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(tgt_lang),
                max_new_tokens=128,
            )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)


class IndicTransBaseline(BaselineTranslator):
    """
    AI4Bharat's IndicTrans2 — state-of-the-art for Indian language translation.
    Uses separate models for en→indic and indic→en directions.
    """

    def __init__(self):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # IndicTrans2 uses separate models per direction
        self.models = {}
        self.tokenizers = {}

        # Load indic→en model
        print("📥 Loading IndicTrans2 (indic→en)...")
        try:
            model_name_i2e = "ai4bharat/indictrans2-indic-en-dist-200M"
            self.tokenizers["ta2en"] = AutoTokenizer.from_pretrained(
                model_name_i2e, trust_remote_code=True
            )
            self.models["ta2en"] = AutoModelForSeq2SeqLM.from_pretrained(
                model_name_i2e, trust_remote_code=True
            ).to(self.device)
            self.models["ta2en"].eval()
            print("   ✅ IndicTrans2 (indic→en) loaded")
        except Exception as e:
            logger.warning(f"IndicTrans2 indic→en failed: {e}")
            print(f"   ⚠️  IndicTrans2 (indic→en) failed: {e}")

        # Load en→indic model
        print("📥 Loading IndicTrans2 (en→indic)...")
        try:
            model_name_e2i = "ai4bharat/indictrans2-en-indic-dist-200M"
            self.tokenizers["en2ta"] = AutoTokenizer.from_pretrained(
                model_name_e2i, trust_remote_code=True
            )
            self.models["en2ta"] = AutoModelForSeq2SeqLM.from_pretrained(
                model_name_e2i, trust_remote_code=True
            ).to(self.device)
            self.models["en2ta"].eval()
            print("   ✅ IndicTrans2 (en→indic) loaded")
        except Exception as e:
            logger.warning(f"IndicTrans2 en→indic failed: {e}")
            print(f"   ⚠️  IndicTrans2 (en→indic) failed: {e}")

    @property
    def name(self):
        return "IndicTrans2-dist-200M"

    def translate(self, text, direction="ta2en"):
        """Translate using IndicTrans2."""
        if direction not in self.models:
            return f"[IndicTrans2 {direction} not available]"

        tokenizer = self.tokenizers[direction]
        model = self.models[direction]

        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=128)

        return tokenizer.decode(output[0], skip_special_tokens=True)


class ZeroShotLlamaBaseline(BaselineTranslator):
    """
    Zero-shot Llama 3.1 8B baseline — same model, same prompt, NO LoRA fine-tuning.
    Shows the improvement gained from fine-tuning.
    """

    def __init__(self, config):
        from unsloth import FastLanguageModel

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"📥 Loading zero-shot Llama baseline: {config.model_name}")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model_name,
            max_seq_length=config.max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        # NO LoRA applied — raw base model
        FastLanguageModel.for_inference(self.model)
        print("   ✅ Zero-shot Llama loaded (no fine-tuning)")

    @property
    def name(self):
        return "Llama-3.1-8B (zero-shot)"

    def translate(self, text, direction="ta2en"):
        """Translate using zero-shot Llama with the same prompt format."""
        if direction == "ta2en":
            prompt = f"Translate from Tamil to English.\n\nTamil: {text}\nEnglish:"
            split_key = "English:"
        else:
            prompt = f"Translate from English to Tamil.\n\nEnglish: {text}\nTamil:"
            split_key = "Tamil:"

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

        if split_key in result:
            parsed = result.split(split_key)[-1].strip().split("\n")[0].strip()
            return parsed
        return result.strip()


def load_all_baselines(config):
    """
    Load all baseline models. Returns a list of BaselineTranslator instances.
    Gracefully handles failures — returns only successfully loaded baselines.
    """
    baselines = []

    # 1. NLLB-200
    try:
        baselines.append(NLLBBaseline())
    except Exception as e:
        print(f"⚠️  NLLB-200 baseline failed to load: {e}")

    # 2. IndicTrans2
    try:
        baselines.append(IndicTransBaseline())
    except Exception as e:
        print(f"⚠️  IndicTrans2 baseline failed to load: {e}")

    # 3. Zero-shot Llama
    try:
        baselines.append(ZeroShotLlamaBaseline(config))
    except Exception as e:
        print(f"⚠️  Zero-shot Llama baseline failed to load: {e}")

    print(f"\n📊 Loaded {len(baselines)} baseline(s): "
          f"{', '.join(b.name for b in baselines)}")

    return baselines
