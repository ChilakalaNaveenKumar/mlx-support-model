# llm_code_analyzer.py
"""
LLM Code Analyzer - Uses tree-sitter for AST parsing and local LLM for code understanding
Focussed on block identification and QA generation for fine-tuning
"""

import os
import json
import argparse
import re
import time
import requests
from typing import List, Dict, Any, Optional
from tree_sitter_languages import get_parser
from tqdm import tqdm

# Language mappings
EXT_TO_LANG = {
    '.py': 'python', '.js': 'javascript', '.ts': 'typescript', 
    '.java': 'java', '.c': 'c', '.cpp': 'cpp', '.go': 'go', 
    '.rb': 'ruby', '.php': 'php'
}

class CodeBlock:
    """Represents a functional block of code with metadata."""
    
    def __init__(self, start_line: int, end_line: int, code: str, block_type: str):
        self.start_line = start_line
        self.end_line = end_line
        self.code = code
        self.block_type = block_type
        self.description = ""
        self.questions_answers = []
    
    def to_dict(self):
        return {
            "start_line": self.start_line,
            "end_line": self.end_line,
            "code": self.code,
            "block_type": self.block_type,
            "description": self.description,
            "questions_answers": self.questions_answers
        }

class LocalLLMClient:
    """Client for interacting with a local LLM API."""
    
    def __init__(self, api_url="http://localhost:11434/api/chat"):
        self.api_url = api_url
        self.model_name = "gemma3:27b"  # Default model
        
    def set_model(self, model_name):
        """Change the model used for queries."""
        self.model_name = model_name
        
    def query(self, prompt: str, system_prompt: str = "You are a helpful assistant specializing in code analysis.") -> str:
        """Send a query to the local LLM and get a response."""
        try:
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "options": {
                    "temperature": 0.1
                }
            }
            
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            if 'message' in result and 'content' in result['message']:
                return result['message']['content']
            else:
                return f"Error: Unexpected response structure from LLM API"
        except Exception as e:
            print(f"Error querying local LLM: {e}")
            return f"Error: {e}"
    
    def generate_block_description(self, code_block: CodeBlock) -> str:
        """Generate a description for a code block."""
        prompt = f"""
Analyze this code block and provide a concise description of what it does:

```{code_block.block_type}
{code_block.code}
```

Respond with a single paragraph that explains the purpose and functionality of this code.
"""
        description = self.query(prompt)
        code_block.description = description
        return description
    
    def generate_questions_answers(self, code_block: CodeBlock) -> List[Dict[str, str]]:
        """Generate question-answer pairs for a code block."""
        standard_questions = [
            "What is the purpose of this code block?",
            "What happens when this code runs?",
            "What operations are executed in this block?",
            "What are the inputs and outputs of this code block?"
        ]
        
        qa_pairs = []
        
        # Generate answers for standard questions
        for question in standard_questions:
            prompt = f"""
Analyze this code block:

```{code_block.block_type}
{code_block.code}
```

Question: {question}

Provide a detailed and accurate answer based solely on the code shown.
"""
            answer = self.query(prompt)
            qa_pairs.append({"question": question, "answer": answer})
            time.sleep(0.5)  # Small delay to avoid overwhelming the LLM service
        
        code_block.questions_answers = qa_pairs
        return qa_pairs

def extract_code_blocks(file_path: str, lang: str) -> List[CodeBlock]:
    """Extract code blocks from a file using tree-sitter."""
    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()
    
    code_lines = code.split('\n')
    blocks = []
    
    try:
        parser = get_parser(lang)
        tree = parser.parse(code.encode('utf8'))
        root = tree.root_node
        
        # Identify functions, classes, methods
        for node in root.children:
            if node.type in ['function_definition', 'class_definition', 'method_definition']:
                start_line = node.start_point[0]
                end_line = node.end_point[0]
                block_code = "\n".join(code_lines[start_line:end_line+1])
                blocks.append(CodeBlock(start_line, end_line, block_code, node.type))
            
            # Look for nested functions and methods in classes
            if node.type == 'class_definition':
                for child in node.children:
                    if child.type in ['function_definition', 'method_definition']:
                        start_line = child.start_point[0]
                        end_line = child.end_point[0]
                        block_code = "\n".join(code_lines[start_line:end_line+1])
                        blocks.append(CodeBlock(start_line, end_line, block_code, child.type))
    
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        # Fallback: use regex to identify functions and classes
        blocks = extract_blocks_with_regex(code)
    
    return blocks

def extract_blocks_with_regex(code: str) -> List[CodeBlock]:
    """Fallback method to extract code blocks using regex."""
    blocks = []
    lines = code.split('\n')
    
    # Pattern for Python functions and methods
    func_pattern = re.compile(r'^\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(')
    # Pattern for Python classes
    class_pattern = re.compile(r'^\s*class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[:\(]')
    
    in_block = False
    block_start = 0
    block_type = ""
    indent_level = 0
    
    for i, line in enumerate(lines):
        if not in_block:
            if func_pattern.match(line):
                in_block = True
                block_start = i
                block_type = "function_definition"
                indent_level = len(line) - len(line.lstrip())
            elif class_pattern.match(line):
                in_block = True
                block_start = i
                block_type = "class_definition"
                indent_level = len(line) - len(line.lstrip())
        else:
            # Check if we've exited the block (indentation level decreased)
            if line.strip() and len(line) - len(line.lstrip()) <= indent_level:
                in_block = False
                block_code = "\n".join(lines[block_start:i])
                blocks.append(CodeBlock(block_start, i-1, block_code, block_type))
    
    # Handle case where the last block extends to the end of the file
    if in_block:
        block_code = "\n".join(lines[block_start:])
        blocks.append(CodeBlock(block_start, len(lines)-1, block_code, block_type))
    
    return blocks

def generate_finetune_dataset(blocks: List[CodeBlock], output_file: str) -> None:
    """Generate a fine-tuning dataset from code blocks."""
    dataset = []
    
    for block in blocks:
        for qa_pair in block.questions_answers:
            # Create a training example with code context
            example = {
                "messages": [
                    {"role": "system", "content": "You are a helpful code analysis assistant."},
                    {"role": "user", "content": f"```\n{block.code}\n```\n\n{qa_pair['question']}"},
                    {"role": "assistant", "content": qa_pair['answer']}
                ]
            }
            dataset.append(example)
            
            # Also create a keyword-based example
            keywords = extract_keywords(block.code)
            for keyword in keywords[:2]:  # Limit to first 2 keywords to avoid too many examples
                example = {
                    "messages": [
                        {"role": "system", "content": "You are a helpful code analysis assistant."},
                        {"role": "user", "content": f"How is {keyword} used in this code?"},
                        {"role": "assistant", "content": f"Looking at lines {block.start_line}-{block.end_line}, {generate_keyword_answer(block, keyword)}"}
                    ]
                }
                dataset.append(example)
    
    # Write dataset to file
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item) + '\n')
    
    print(f"Generated dataset with {len(dataset)} examples and saved to {output_file}")

def extract_keywords(code: str) -> List[str]:
    """Extract important keywords from code."""
    # Extract function and class names
    function_pattern = re.compile(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(')
    class_pattern = re.compile(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[:\(]')
    
    functions = function_pattern.findall(code)
    classes = class_pattern.findall(code)
    
    # Extract imported modules
    import_pattern = re.compile(r'import\s+([a-zA-Z_][a-zA-Z0-9_]*)')
    from_import_pattern = re.compile(r'from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+import')
    
    imports = import_pattern.findall(code)
    from_imports = from_import_pattern.findall(code)
    
    # Extract variable assignments
    var_pattern = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*')
    variables = var_pattern.findall(code)
    
    # Combine all keywords
    keywords = set(functions + classes + imports + from_imports + variables)
    
    return list(keywords)

def generate_keyword_answer(block: CodeBlock, keyword: str) -> str:
    """Generate an explanation about a keyword in the code."""
    if keyword in block.description:
        return f"this code uses '{keyword}' as described in the block: {block.description}"
    elif keyword in block.code:
        return f"this code contains '{keyword}', which appears to be used in this block."
    else:
        return f"this code does not directly use '{keyword}'."

def process_file(file_path: str, llm_client: LocalLLMClient) -> List[CodeBlock]:
    """Process a single file, extracting and analyzing code blocks."""
    ext = os.path.splitext(file_path)[1]
    lang = EXT_TO_LANG.get(ext)
    
    if not lang:
        print(f"Unsupported file type: {file_path}")
        return []
    
    print(f"Processing {file_path} ({lang})")
    blocks = extract_code_blocks(file_path, lang)
    print(f"Found {len(blocks)} code blocks")
    
    for i, block in enumerate(blocks):
        print(f"Analyzing block {i+1}/{len(blocks)}: {block.block_type} (lines {block.start_line}-{block.end_line})")
        llm_client.generate_block_description(block)
        llm_client.generate_questions_answers(block)
    
    # Save blocks to JSON
    output_path = f"{file_path}.blocks.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump([block.to_dict() for block in blocks], f, indent=2)
    
    return blocks

def main():
    parser = argparse.ArgumentParser(description="LLM Code Analyzer")
    parser.add_argument("--file", required=True, help="Path to the code file to analyze")
    parser.add_argument("--model", default="gemma3:27b", help="Name of the local LLM model to use")
    parser.add_argument("--llm-url", default="http://localhost:11434/api/chat", help="URL for local LLM API")
    parser.add_argument("--output", default="finetune_dataset.jsonl", help="Output file for fine-tuning dataset")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"File not found: {args.file}")
        return 1
    
    # Initialize LLM client
    llm_client = LocalLLMClient(api_url=args.llm_url)
    llm_client.set_model(args.model)
    
    # Process the file
    blocks = process_file(args.file, llm_client)
    
    # Generate fine-tuning dataset
    if blocks:
        generate_finetune_dataset(blocks, args.output)
        
        # Create an Ollama-compatible fine-tuning file
        ollama_output = f"{os.path.splitext(args.output)[0]}_ollama.jsonl"
        with open(args.output, 'r', encoding='utf-8') as f, open(ollama_output, 'w', encoding='utf-8') as out:
            for line in f:
                data = json.loads(line)
                messages = data.get('messages', [])
                if len(messages) >= 3:
                    user_message = messages[1].get('content', '')
                    assistant_message = messages[2].get('content', '')
                    
                    ollama_format = {
                        "prompt": user_message + "\n",
                        "response": assistant_message
                    }
                    
                    out.write(json.dumps(ollama_format) + '\n')
        
        print(f"Generated Ollama-compatible dataset: {ollama_output}")
        
        # Write a Modelfile
        modelfile_path = "Modelfile"
        with open(modelfile_path, 'w', encoding='utf-8') as f:
            f.write(f"""FROM {args.model}
PARAMETER temperature 0.7
PARAMETER repetition_penalty 1.1

# Fine-tuning data
TRAIN {ollama_output}
""")
        print(f"Created Modelfile for Ollama fine-tuning")
    
    return 0

if __name__ == "__main__":
    main()