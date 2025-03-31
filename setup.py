from setuptools import setup, find_packages

setup(
    name="mlx_support_model",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "mlx>=0.2.0",
        "mlx-lm>=0.0.7",
        "huggingface_hub>=0.16.0",
        "safetensors>=0.3.1",
        "sentencepiece>=0.1.99"
    ],
    python_requires=">=3.8",
) 