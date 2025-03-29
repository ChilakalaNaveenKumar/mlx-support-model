# MLX Code Model

A modular framework for MLX-based language models optimized for Apple Silicon. This tool provides sophisticated features for working with models like Qwen2.5-Coder, including Fill-in-Middle (FIM) support, chat functionality, code completion, and intelligent caching.

## Features

- 🧠 Fill-in-Middle (FIM) support for code completion
- 💬 Full chat mode with conversation history
- 🔄 Dynamic context window sizing based on input length
- ⚡ Fast response with intelligent caching system
- 📄 Sophisticated file processing
- 🖥️ Dedicated code completion mode
- 🚀 Interactive chat and code completion interfaces

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4 series chip)
- Python 3.8 or higher
- MLX and MLX-LM libraries

## Installation

1. Clone this repository:
```

## Usage

### Chat Mode

Start a chat conversation:

```bash
python main.py --interactive --chat
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
python main.py --model mlx-community/Qwen2.5-Coder-32B-Instruct-8bit --prompt "Hello"
```

Customize generation parameters:

```bash
python main.py --prompt "Write a short story" --max-tokens 2000 --temperature 0.9 --repetition-penalty 1.3
```

Clear and manage cache:

```bash
python main.py --clear-cache --prompt "Hello"
```

## Modular Service Architecture

The framework employs a modular service-based architecture for better organization and maintainability:

### Core Services

- **ModelService**: Handles core model operations like loading and generation
- **CacheService**: Manages caching for models, responses, and files
- **ChatService**: Provides chat functionality with history management
- **CodeService**: Handles code-specific operations like FIM and analysis
- **FileService**: Manages file-related operations and processing
- **cli_service.py**: Handles command-line arguments
- **interactive_service.py**: Manages interactive modes
- **command_service.py**: Implements the Command pattern

### Utility Modules

- **code_utils**: Tools for code analysis and manipulation
- **file_utils**: File handling utilities
- **token_utils**: Token counting and management
- **prompt_utils**: Prompt formatting and optimization

## Features in Detail

### Model Loading and Caching

The framework intelligently manages model loading:

- Models are only loaded when needed
- Default model is `mlx-community/Qwen2.5-Coder-32B-Instruct-8bit`
- Model caching keeps models in memory for faster access
- Response caching avoids regenerating identical responses

### Fill-in-Middle (FIM) Support

FIM allows code completion within existing code:

- Automatic detection of code completion contexts
- Manual FIM with prefix and suffix inputs
- Optimized context handling for large files

### Chat Mode

Full-featured chat interface:

- Conversation history tracking
- Custom system prompts
- Interactive chat interface

### Cache System

The caching system offers several benefits:

- Model caching: Keeps loaded models in memory
- Response caching: Stores generated responses based on input+parameters
- File caching: Efficiently handles file content
- Persistence: Response cache persists between sessions

## Customization

You can customize settings by editing the configuration in `config.py`, including:

- Cache behavior
- Context window sizing
- FIM settings
- Default generation parameters
- Chat behavior

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.bash
git clone https://github.com/yourusername/mlx-code-model.git
cd mlx-code-model
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

```
mlx_code_model/
├── config.py                 # Configuration settings
├── main.py                   # Slim main entry point
├── requirements.txt          # Dependencies
├── README.md                 # Documentation
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
    ├── model_service.py      # Core model operations
    └── utils/                # Utilities
        ├── __init__.py       # Package initialization
        ├── code_utils.py     # Code analysis utilities
        ├── file_utils.py     # File handling utilities
        ├── token_utils.py    # Tokenization utilities
        └── prompt_utils.py   # Prompt formatting utilities
```