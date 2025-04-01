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

### Directory Structure Mind Map

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

### Detailed File Analysis

```
File: services/router/index.js
========================

Component: router
Description: Configures and exports a Vue Router instance for the application

Imports:
  - createRouter, createWebHistory from 'vue-router' (lines 1-2)
  - ResumeMaker from '@/components/resume-maker.vue' (line 3)

Key Functionality:
  - Defines routes for the application (lines 6-15)
  - Configures the router to use HTML5 history mode (line 18)
  - Exports the configured router instance (line 21)

Components/Functions Defined:
  - router (Vue Router instance) (lines 6-21): Configured router with defined routes

File size: 428 bytes
Line count: 24 lines
```

### Consolidated Analysis

The tool also generates a consolidated analysis file containing the analyses of all files in the project, providing a comprehensive view of the entire codebase.

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Analyzing a Project

```
python main.py /path/to/your/project
```

### Analyzing a Single File

```
python main.py --file /path/to/your/file.py
```

### Analyzing Directory Structure Only

If you only want to analyze the directory structure without processing file contents:

```
python main.py --structure-only /path/to/your/project
```

### Optional Arguments

- `--output` or `-o`: Specify output directory for analysis files
- `--model` or `-m`: Specify MLX model to use for analysis
- `--verbose` or `-v`: Enable verbose logging
- `--structure-only` or `-s`: Only analyze directory structure, skip file content analysis
- `--file` or `-f`: Analyze a single file instead of the entire project

## Requirements

- Python 3.8+
- MLX framework
- MLX language models

## Project Structure

```
code_analyzer/
├── __init__.py        # Package initialization
├── scanner.py         # Project file scanner
├── processor.py       # LLM analyzer for code files
├── generator.py       # Mind map generator
├── analyzer.py        # Main analysis orchestrator
└── main.py            # Command-line interface
```

## Limitations

- Performance depends on the size of your project and available resources
- Complex or highly specialized codebases may require additional configuration

## License

MIT