"""
Configuration module for MLX language models.
Contains settings and constants for model initialization and features.
"""

import os

# Default model - only load this when explicitly requested
# DEFAULT_MODEL = "mlx-community/Qwen2.5-Coder-32B-Instruct-8bit"
DEFAULT_MODEL = "mlx-community/Qwen2.5-Coder-32B-Instruct-6bit"
# Additional models available
ADDITIONAL_MODELS = {
    "mistral": "mlx-community/OpenHermes-2.5-Mistral-7B-4bit-mlx",
    "llama": "mlx-community/Llama-3-8B-Instruct-4bit"
}

# Generation parameters
DEFAULT_GENERATION_PARAMS = {
    "max_tokens": 4000,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "repetition_penalty": 1.1
}

# Cache settings
CACHE_SETTINGS = {
    "enable_cache": True,
    "cache_dir": os.path.join(os.path.expanduser("~"), ".mlx_cache"),
    "max_models_cached": 3,  # Maximum number of models to keep in memory
    "tokenizer_cache_size": 5000,  # Cache size for tokenizer
    "max_file_cache_entries": 100,  # Maximum number of file results to cache
    "max_response_cache_entries": 1000  # Maximum number of response results to cache
}

# Context window settings
CONTEXT_SETTINGS = {
    "default_max_length": 8192,  # Default maximum context length
    "qwen_max_length": 32768,    # Maximum context length for Qwen models
    "dynamic_scaling": True      # Dynamically scale context based on input size
}

# Chat settings
CHAT_SETTINGS = {
    "system_prompt": "You are a helpful, accurate AI assistant with expertise in programming. Answer questions, explain concepts, and help with code.",
    "chat_template": "default",   # Template for chat formatting
    "save_history": True,         # Whether to save chat history
    "history_file": os.path.join(os.path.expanduser("~"), ".mlx_chat_history"),
    "max_history_turns": 20       # Maximum number of turns to keep in history
}

# Fill-in-Middle settings 
FIM_SETTINGS = {
    "prefix_suffix_separator": "<fim_separator>",  # Separator for FIM mode
    "enable_auto_fim": True,   # Auto-detect and enable FIM for code completion
    "fim_prefix_length": 500,  # Characters to include from prefix in auto-FIM
    "fim_suffix_length": 300   # Characters to include from suffix in auto-FIM
}

# Logging settings
LOG_SETTINGS = {
    "level": "INFO",  # Default log level
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": None  # Set to a path to enable file logging
}