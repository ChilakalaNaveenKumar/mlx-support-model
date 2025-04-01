I'll create a readme for a code analysis tool that can work with any programming language, not just Vue.js.

# Code Structure Analyzer

A tool for analyzing and visualizing code component relationships across any programming language project.

## Overview

This tool scans your codebase and generates a hierarchical mind map showing the relationships between different files, functions, classes, and other code components. It uses a local MLX-based language model to understand code structure and connections.

## Features

- 🔍 Analyzes code components across any programming language
- 🌲 Generates hierarchical mind maps of project structure
- 🔄 Identifies dependencies and relationships between components
- 📊 Provides insights into code functionality and architecture
- 🧠 Uses local LLM for intelligent code analysis

## How It Works

1. The tool scans your project directory for all code files
2. Each file is analyzed to extract:
   - Component names (functions, classes, modules)
   - Import/include statements
   - Dependencies and relationships
   - Primary functionality
3. The local LLM processes each file to understand component relationships
4. A hierarchical mind map is built showing how components connect
5. Components are tracked by both name and path to avoid collisions

## Example Output

```
project_name/
├── config.py                 # Configuration settings
├── main.py                   # Main entry point
├── services/                 # Services directory
│   ├── application.py        # Core application class
│   │   ├── command_service.py # Command pattern implementation
│   │   └── cli_service.py     # Command-line interface
│   ├── model_service.py      # Model management
│   │   └── cache_service.py   # Model caching functionality
│   └── utils/                # Utility functions
│       ├── file_utils.py     # File handling utilities
│       └── token_utils.py    # Token processing utilities
```

## Usage

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the analyzer on your project:
   ```
   python analyzer.py /path/to/your/project
   ```
4. View the generated mind map in the output directory

## Requirements

- Python 3.8+
- MLX framework
- MLX language models

## Limitations

- Performance depends on the size of your project and available resources
- Complex or highly specialized codebases may require additional configuration

## Future Improvements

- Interactive visualization options
- Code complexity metrics
- Architecture pattern recognition
- Multi-repository analysis

## License

MIT