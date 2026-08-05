from setuptools import setup, find_packages

setup(
    name='tamil-translation-unsloth',
    version='2.0.0',
    description='Bidirectional Tamil-English translation using Unsloth + LoRA fine-tuned Llama 3.1 8B',
    author='Team B5 - Amrita Vishwa Vidyapeetham',
    license='MIT',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'unsloth>=2024.0',
        'transformers>=4.40.0',
        'peft>=0.10.0',
        'accelerate>=0.30.0',
        'bitsandbytes>=0.43.0',
        'datasets>=2.19.0',
        'sentence-transformers>=2.7.0',
        'sacrebleu>=2.4.0',
        'torch>=2.2.0',
        'tqdm>=4.66.0',
        'trl>=0.8.0',
        'numpy>=1.26.0',
        'unbabel-comet>=2.2.0',
    ],
    python_requires='>=3.8',
)