# code_questions_generator.py
"""
Code Questions Generator - Creates targeted Q&A pairs from code blocks
and prepares them for fine-tuning Ollama models using OllamaSFTTrainer.

This tool is designed to work with the LLM Code Analyzer.
"""

import os
import json
import argparse
import re
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# Standard question templates for different code constructs
QUESTION_TEMPLATES = {
    "function_definition": [
        "What is the purpose of the {name} function?",
        "What operations are executed when the {name} function runs?",
        "What are the inputs and outputs of the {name} function?",
        "How does the {name} function handle errors?",
        "What would happen if {parameter} is None in the {name} function?"
    ],
    "method_definition": [
        "What does the {name} method do in the {class_name} class?",
        "How does the {name} method interact with other parts of the {class_name} class?",
        "What are the inputs and outputs of the {name} method?",
        "When would you call the {name} method?"
    ],
    "class_definition": [
        "What is the purpose of the {name} class?",
        "What attributes and methods does the {name} class have?",
        "How would you instantiate and use the {name} class?",
        "What functionality does the {name} class provide?"
    ]
}

def extract_name_and_params(code_block: Dict) -> Dict[str, str]:
    """Extract function/class name and parameters from code block."""
    code = code_block["code"]
    block_type = code_block["block_type"]
    result = {"name": "this", "class_name": "this", "parameter": "the input"}
    
    if block_type == "function_definition" or block_type == "method_definition":
        # Extract function/method name
        match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code)
        if match:
            result["name"] = match.group(1)
        
        # Extract parameter names
        params_match = re.search(r'def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\((.*?)\)', code, re.DOTALL)
        if params_match:
            params = params_match.group(1).split(',')
            if params and params[0].strip() and 'self' not in params[0]:
                result["parameter"] = params[0].split(':')[0].strip()
            elif len(params) > 1:
                result["parameter"] = params[1].split(':')[0].strip()
    
    elif block_type == "class_definition":
        # Extract class name
        match = re.search(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)', code)
        if match:
            result["name"] = match.group(1)
            result["class_name"] = match.group(1)
    
    # Handle method in class - find parent class
    if block_type == "method_definition":
        class_lines = code.split('\n')
        indentation = len(class_lines[0]) - len(class_lines[0].lstrip())
        for line in reversed(code.split('\n')[:code_block["start_line"]]):
            if line.strip().startswith("class ") and (len(line) - len(line.lstrip())) < indentation:
                match = re.search(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)', line)
                if match:
                    result["class_name"] = match.group(1)
                    break
    
    return result

def generate_targeted_questions(blocks_file: str) -> List[Dict[str, str]]:
    """Generate targeted questions for code blocks based on their type and content."""
    if not os.path.exists(blocks_file):
        print(f"File not found: {blocks_file}")
        return []
    
    # Load blocks from file
    with open(blocks_file, 'r', encoding='utf-8') as f:
        blocks = json.load(f)
    
    qa_pairs = []
    
    for block in blocks:
        # Skip blocks that are too small (likely not significant)
        if len(block["code"].strip().split('\n')) < 3:
            continue
        
        # Get metadata for template population
        metadata = extract_name_and_params(block)
        
        # Get templates for this block type
        templates = QUESTION_TEMPLATES.get(block["block_type"], 
                                          QUESTION_TEMPLATES["function_definition"])
        
        # Generate questions from templates
        for template in templates:
            # Fill in the template with extracted metadata
            question = template.format(**metadata)
            
            # Add to QA pairs
            qa_pairs.append({
                "question": question,
                "code": block["code"],
                "block_type": block["block_type"],
                "lines": f"{block['start_line']}-{block['end_line']}"
            })
        
        # Generate keyword-based questions
        keywords = extract_keywords(block["code"])
        for keyword in keywords[:3]:  # Limit to 3 keywords
            if keyword != metadata.get("name") and len(keyword) > 3:
                qa_pairs.append({
                    "question": f"How is {keyword} used in this code?",
                    "code": block["code"],
                    "block_type": block["block_type"],
                    "lines": f"{block['start_line']}-{block['end_line']}"
                })
    
    return qa_pairs

def extract_keywords(code: str) -> List[str]:
    """Extract important keywords from code."""
    # Extract function and class names
    function_pattern = re.compile(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(')
    class_pattern = re.compile(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[:\(]')
    
    functions = function_pattern.findall(code)
    classes = class_pattern.findall(code)
    
    # Extract variable assignments
    var_pattern = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*')
    variables = var_pattern.findall(code)
    
    # Extract method calls
    method_pattern = re.compile(r'\.([a-zA-Z_][a-zA-Z0-9_]*)\(')
    methods = method_pattern.findall(code)
    
    # Combine all keywords and remove duplicates
    keywords = set(functions + classes + variables + methods)
    
    # Filter out common Python keywords
    common_words = {'self', 'None', 'True', 'False', 'if', 'else', 'for', 'while', 'try', 'except', 'with'}
    keywords = [k for k in keywords if k not in common_words and len(k) > 2]
    
    return list(keywords)

def create_ollama_dataset(qa_pairs: List[Dict[str, str]], output_file: str) -> None:
    """Create a dataset file for use with Ollama fine-tuning."""
    # Convert QA pairs to the format expected by OllamaSFTTrainer
    formatted_pairs = []
    
    for pair in qa_pairs:
        formatted_pair = {
            "question": f"```python\n{pair['code']}\n```\n\n{pair['question']}",
            "answer": f"This is a {pair['block_type']} (lines {pair['lines']}).\n\nAnalyzing this code: " + 
                      "I need to determine the functionality and purpose based on the implementation."
        }
        formatted_pairs.append(formatted_pair)
    
    # Write to JSONL format
    with open(output_file, 'w', encoding='utf-8') as f:
        for pair in formatted_pairs:
            f.write(json.dumps(pair) + '\n')
    
    print(f"Created dataset with {len(formatted_pairs)} examples in {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Code Questions Generator")
    parser.add_argument("--blocks-file", required=True, help="JSON file containing code blocks")
    parser.add_argument("--output", default="code_qa_dataset.jsonl", help="Output JSONL file for OllamaSFTTrainer")
    
    args = parser.parse_args()
    
    # Generate questions
    qa_pairs = generate_targeted_questions(args.blocks_file)
    
    if qa_pairs:
        # Create dataset
        create_ollama_dataset(qa_pairs, args.output)
        print(f"Generated {len(qa_pairs)} question-answer pairs")
        print(f"You can now use this dataset with OllamaSFTTrainer:")
        print(f"  python ollama_sft_trainer.py --train-data {args.output} --model llama3 --output-dir ./tuning_output")
    else:
        print("No question-answer pairs generated. Check your input file.")

if __name__ == "__main__":
    main()