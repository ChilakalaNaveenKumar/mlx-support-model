# File to Q&A System

A tool for generating comprehensive question-answer pairs from files to fine-tune local language models. This system enables you to teach your local LLM about internal files by creating detailed Q&A pairs that cover every aspect of the file content.

## Overview

This system processes files (code, documentation, text, etc.) and uses a local LLM to generate exhaustive Q&A pairs that cover all information in those files. These Q&A pairs can then be used to fine-tune the LLM, enabling it to "learn" about your internal files and answer questions about them without needing a RAG (Retrieval-Augmented Generation) system.

## Files in this Package

- `main.py`: Main script for processing files and generating Q&A pairs
- `qa_generator.py`: Core functions for generating Q&A pairs using a local LLM
- `file_processor.py`: Utility functions for file operations
- `README.md`: This documentation file

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3 chip)
- Python 3.8 or higher
- Local LLM framework (MLX) installed

## Installation

1. Ensure you have the MLX Support Model framework installed
2. Clone or download this repository to your local machine
3. Navigate to the folder containing the scripts

## Basic Usage

### Process a Single File

```bash
python main.py --file path/to/your/file.py --output qa_output
```

This will:
1. Read the file
2. Generate comprehensive Q&A pairs about its content
3. Save the pairs to the `qa_output` directory

### Process an Entire Directory

```bash
python main.py --directory path/to/your/codebase --output qa_output
```

This processes all supported files in the directory (and subdirectories), generating Q&A pairs for each.

### Create a Fine-Tuning Dataset and Script

```bash
python main.py --directory path/to/your/codebase --output qa_dataset --create-finetune
```

This processes all files, creates train/test datasets, and generates a fine-tuning script.

## Advanced Options

### Prioritize Keywords

```bash
python main.py --file path/to/your/file.py --priority-keywords "important_function" "key_class" "critical_term"
```

### Specify Number of Q&A Pairs

```bash
python main.py --file path/to/your/file.py --min-pairs 100
```

### Specify Output Format

```bash
python main.py --file path/to/your/file.py --output-format json
```

Supported formats: `jsonl` (default), `json`, `csv`

### Filter Files by Extension

```bash
python main.py --directory path/to/your/codebase --extensions .py .js .html
```

### Use Specific LLM Model

```bash
python main.py --file path/to/your/file.py --llm-model "mlx-community/Llama-3-8B-Instruct-4bit"
```

## Complete Command Reference

```
usage: main.py [-h] (--file FILE | --directory DIRECTORY) [--output OUTPUT]
               [--output-format {jsonl,json,csv}] [--min-pairs MIN_PAIRS]
               [--priority-keywords PRIORITY_KEYWORDS [PRIORITY_KEYWORDS ...]]
               [--llm-model LLM_MODEL]
               [--extensions EXTENSIONS [EXTENSIONS ...]] [--max-files MAX_FILES]
               [--create-finetune] [--train-split TRAIN_SPLIT] [--verbose]

Generate comprehensive Q&A pairs from files for model fine-tuning

options:
  -h, --help            show this help message and exit
  --file FILE, -f FILE  Path to a single file to process
  --directory DIRECTORY, -d DIRECTORY
                        Path to a directory of files to process
  --output OUTPUT, -o OUTPUT
                        Output directory for Q&A pairs (default: qa_output)
  --output-format {jsonl,json,csv}, -of {jsonl,json,csv}
                        Output format for Q&A pairs (default: jsonl)
  --min-pairs MIN_PAIRS, -m MIN_PAIRS
                        Minimum number of Q&A pairs to generate per file (default: 50)
  --priority-keywords PRIORITY_KEYWORDS [PRIORITY_KEYWORDS ...], -pk PRIORITY_KEYWORDS [PRIORITY_KEYWORDS ...]
                        Keywords to prioritize in Q&A generation (e.g., file names, key terms)
  --llm-model LLM_MODEL, -lm LLM_MODEL
                        Local LLM model to use (default: use the system's default model)
  --extensions EXTENSIONS [EXTENSIONS ...], -e EXTENSIONS [EXTENSIONS ...]
                        File extensions to process (default: common code and text files)
  --max-files MAX_FILES, -mf MAX_FILES
                        Maximum number of files to process (0 = no limit, default: 0)
  --create-finetune, -cf
                        Create a fine-tuning script after generating Q&A pairs
  --train-split TRAIN_SPLIT, -ts TRAIN_SPLIT
                        Train/test split ratio (default: 0.8)
  --verbose, -v         Enable verbose logging
```

## Fine-Tuning Process

After generating Q&A pairs, you can fine-tune your local LLM with these steps:

1. Generate the dataset:
```bash
python main.py --directory your_files --output qa_dataset --create-finetune
```

2. Run the generated fine-tuning script:
```bash
python qa_dataset/finetune.py
```

3. The fine-tuned model will be saved in the specified output directory.

## Q&A Pair Format

For each file, Q&A pairs will be generated in the following format:

```json
{"question": "What is the purpose of the read_file function in file_processor.py?", "answer": "The read_file function in file_processor.py is designed to read content from a file and return it as a string. It handles different encodings by first attempting to open the file with UTF-8 encoding, and if that fails, it tries with latin-1 encoding. The function also includes error handling to catch and log any exceptions that occur during file reading."}
```

## How It Works

1. **File Processing**: The system reads files and organizes their content
2. **Prompt Construction**: A specialized prompt is created for the local LLM
3. **Q&A Generation**: The LLM generates detailed question-answer pairs
4. **Keyword Prioritization**: Specified keywords receive special attention
5. **Comprehensive Coverage**: The system ensures all content is covered
6. **Dataset Creation**: Q&A pairs are organized into training/testing datasets
7. **Fine-tuning Script**: A script is generated for fine-tuning the model

## Supported File Types

This system supports many file types, including:
- Python (.py)
- JavaScript/TypeScript (.js, .ts)
- HTML/CSS (.html, .css)
- Markdown (.md)
- Text (.txt)
- JSON (.json)
- YAML (.yml, .yaml)
- And many more

## Troubleshooting

If you encounter issues:

1. **LLM Not Found**: Ensure the MLX Support Model framework is properly installed
2. **File Reading Errors**: Check file permissions and encodings
3. **Memory Issues**: Process fewer files at once or reduce min-pairs
4. **Fine-tuning Errors**: Check the MLX dependencies and model compatibility

## Example Workflow

```bash
# Step 1: Process a single important file
python main.py --file important_module.py --min-pairs 100 --priority-keywords "critical_function" "key_class"

# Step 2: Process an entire codebase
python main.py --directory ./my_project --output project_qa --create-finetune

# Step 3: Fine-tune the model
python project_qa/finetune.py --epochs 5 --learning-rate 1e-5
```

Once fine-tuning is complete, your local LLM will have the knowledge from your files embedded in its parameters.