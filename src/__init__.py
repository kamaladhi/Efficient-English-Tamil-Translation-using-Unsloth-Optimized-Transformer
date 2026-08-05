"""
Tamil-English Translation Model Package
Bidirectional translation using fine-tuned LLaMA 3.1 8B with Unsloth + LoRA.
v2.0: 100K dataset, bidirectional, FLORES-200 benchmark, baselines, COMET + chrF++.
"""

__version__ = "2.0.0"
__author__ = "Team B5 - Amrita Vishwa Vidyapeetham"

from .config import FastConfig
from .model import TamilTranslationModel, load_model
from .data_loader import TamilDataLoader, load_tamil_data
from .train import TamilTrainer, train_model
from .inference import TamilTranslator, translate
from .evaluate import TamilEvaluator, evaluate_model
from .benchmark import FLORESBenchmark
from .baselines import (
    NLLBBaseline,
    IndicTransBaseline,
    ZeroShotLlamaBaseline,
    load_all_baselines,
)

__all__ = [
    # Config
    'FastConfig',
    # Model
    'TamilTranslationModel',
    'load_model',
    # Data
    'TamilDataLoader',
    'load_tamil_data',
    # Training
    'TamilTrainer',
    'train_model',
    # Inference
    'TamilTranslator',
    'translate',
    # Evaluation
    'TamilEvaluator',
    'evaluate_model',
    # Benchmark
    'FLORESBenchmark',
    # Baselines
    'NLLBBaseline',
    'IndicTransBaseline',
    'ZeroShotLlamaBaseline',
    'load_all_baselines',
]