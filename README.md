# MLX Code Model with Ollama Support

A modular framework for language models optimized for Apple Silicon, now with support for both MLX and Ollama backends. This tool provides sophisticated features for working with models like Qwen2.5-Coder and various Ollama models, including Fill-in-Middle (FIM) support, chat functionality, code completion, and intelligent caching.

## New Feature: Ollama Support

This project now supports two LLM providers:

- 🦙 **Ollama**: Fast, easy-to-use, open-source framework that runs models locally
- 🍎 **MLX**: Native language models optimized for Apple Silicon

By default, Ollama is now the main backend, making the framework more accessible to users across different platforms.

## Features

- 🧠 Fill-in-Middle (FIM) support for code completion
- 💬 Full chat mode with conversation history
- 🔄 Dynamic context window sizing based on input length
- ⚡ Fast response with intelligent caching system
- 📄 Sophisticated file processing
- 🖥️ Dedicated code completion mode
- 🚀 Interactive chat and code completion interfaces
- ♻️ Seamless switching between MLX and Ollama backends

## Requirements

### For Ollama Backend (Default)
- macOS, Linux, or Windows (with WSL2)
- [Ollama](https://ollama.com/download) installed and running
- Python 3.8 or higher

### For MLX Backend
- macOS with Apple Silicon (M1/M2/M3/M4 series chip)
- Python 3.8 or higher
- MLX and MLX-LM libraries

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/mlx-code-model.git
cd mlx-code-model
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the installation script to set up Ollama (optional):
```bash
python install_ollama.py
```

## Usage

### Basic Usage

```bash
# Using Ollama (default)
python main.py --prompt "Explain binary search"

# Explicitly specify Ollama
python main.py --provider ollama --model llama3 --prompt "Explain binary search"

# Using MLX
python main.py --provider mlx --prompt "Explain binary search"
```

### Chat Mode

Start a chat conversation:

```bash
# With Ollama
python main.py --interactive --chat

# With MLX
python main.py --provider mlx --interactive --chat
```

Or chat with a single prompt:

```bash
python main.py --prompt "Explain binary search in C++" --chat
```

### Code Completion

Complete code with Fill-in-Middle (FIM):

```bash
python main.py --complete-code "def fibonacci(n):\n    if n <= 1:\n        return n\n    " 
```

Use interactive code completion:

```bash
python main.py --interactive --code-mode
```

### File Processing

Process a file:

```bash
python main.py --file path/to/your/code.py
```

Process a file in chat mode:

```bash
python main.py --file path/to/your/code.py --chat
```

Convert a file to a different format:

```bash
python main.py --file path/to/your/data.json --convert-to yaml
```

### Output Control

Save the output to a file:

```bash
python main.py --prompt "Write a quicksort algorithm" --output algorithms/quicksort.py
```

### Advanced Options

Use a specific model:

```bash
# Use a specific Ollama model
python main.py --model codellama --prompt "Hello"

# Use a specific MLX model
python main.py --provider mlx --model mlx-community/Qwen2.5-Coder-32B-Instruct-8bit --prompt "Hello"
```

Customize generation parameters:

```bash
python main.py --prompt "Write a short story" --max-tokens 2000 --temperature 0.9 --repetition-penalty 1.3
```

Clear and manage cache:

```bash
python main.py --clear-cache --prompt "Hello"
```

## Ollama Models

This framework works with any model available through Ollama. You can pull models with:

```bash
ollama pull llama3
ollama pull codellama
ollama pull mistral
```

Some recommended models:
- **llama3**: Good all-purpose model
- **codellama**: Specialized for code generation
- **phi3**: Microsoft's high-performance small model
- **mistral**: Balanced performance and size
- **mixtral**: Larger mixture-of-experts model

## Project Structure

```
mlx_code_model/
├── config.py                 # Configuration settings
├── main.py                   # Slim main entry point
├── requirements.txt          # Dependencies
├── README.md                 # Documentation
├── install_ollama.py         # Installation script for Ollama
└── services/                 # Services directory
    ├── __init__.py           # Package initialization
    ├── application.py        # Main application class
    ├── cache_service.py      # Caching functionality
    ├── chat_service.py       # Chat functionality
    ├── cli_service.py        # Command-line parsing
    ├── code_service.py       # Code completion and FIM
    ├── command_service.py    # Command pattern implementation 
    ├── file_service.py       # File operations
    ├── interactive_service.py # Interactive mode handlers
    ├── model_service.py      # MLX model operations
    ├── ollama_service.py     # Ollama integration
    ├── model_service_factory.py # Provider selection logic
    └── utils/                # Utilities
        ├── __init__.py       # Package initialization
        ├── code_utils.py     # Code analysis utilities
        ├── file_utils.py     # File handling utilities
        ├── token_utils.py    # Tokenization utilities
        └── prompt_utils.py   # Prompt formatting utilities
```

## Configuration

You can customize settings by editing the configuration in `config.py`, including:

- LLM provider (ollama or mlx)
- Default models for each provider
- Cache behavior
- Context window sizing
- FIM settings
- Default generation parameters
- Chat behavior

## Troubleshooting

### Ollama Issues

1. If you see "Error: No model loaded", make sure Ollama server is running
2. Run `ollama list` to see available models
3. If needed, pull the required model with `ollama pull <model_name>`
4. Check Ollama is running on the default port (11434)

### MLX Issues

1. MLX requires macOS with Apple Silicon (M1/M2/M3/M4)
2. Ensure MLX and MLX-LM are properly installed: `pip install -U mlx mlx-lm`
3. If memory issues occur, try using smaller models (4-bit or 8-bit quantized versions)

### General Issues

1. Use `--verbose` flag for detailed logging: `python main.py --verbose ...`
2. Check Python version (3.8+ required)
3. Ensure all dependencies are installed: `pip install -r requirements.txt`

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.