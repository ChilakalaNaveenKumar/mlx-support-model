"""
Utility functions for working with MLX models.
Includes file handling, tokenization, and code parsing helpers.
"""

import os
import re
import logging
import json
import hashlib
from typing import Dict, Any, Optional, List, Tuple, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("utils")


def detect_language(file_path: str) -> str:
    """
    Detect programming language from file extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Programming language name
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    language_map = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.jsx': 'jsx',
        '.tsx': 'tsx',
        '.html': 'html',
        '.css': 'css',
        '.scss': 'scss',
        '.less': 'less',
        '.c': 'c',
        '.cpp': 'cpp',
        '.cs': 'csharp',
        '.java': 'java',
        '.go': 'go',
        '.rb': 'ruby',
        '.php': 'php',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.rs': 'rust',
        '.sh': 'bash',
        '.ps1': 'powershell',
        '.sql': 'sql',
        '.R': 'r',
        '.dart': 'dart',
        '.m': 'objective-c',
        '.scala': 'scala',
        '.pl': 'perl',
        '.lua': 'lua',
        '.ex': 'elixir',
        '.exs': 'elixir',
        '.elm': 'elm',
        '.erl': 'erlang',
        '.fs': 'fsharp',
        '.hs': 'haskell',
        '.json': 'json',
        '.xml': 'xml',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.md': 'markdown',
        '.dockerfile': 'dockerfile',
        '.clj': 'clojure',
        '.lisp': 'lisp',
        '.coffee': 'coffeescript'
    }
    
    return language_map.get(ext, 'text')


def find_cursor_position(code: str) -> int:
    """
    Find cursor position in code, marked with a special token.
    
    Args:
        code: Code with cursor position marker
        
    Returns:
        Cursor position
    """
    # Common cursor markers
    cursor_markers = ["<cursor>", "█", "|", "[]", "<|>", "/*cursor*/", "#cursor#"]
    
    for marker in cursor_markers:
        if marker in code:
            return code.find(marker)
    
    # Default to end of code if no marker found
    return len(code)


def split_at_cursor(code: str, cursor_pos: Optional[int] = None) -> Tuple[str, str]:
    """
    Split code at cursor position into prefix and suffix.
    
    Args:
        code: Code to split
        cursor_pos: Optional cursor position
        
    Returns:
        Tuple of (prefix, suffix)
    """
    if cursor_pos is None:
        cursor_pos = find_cursor_position(code)
        
        # Remove the cursor marker if present
        cursor_markers = ["<cursor>", "█", "|", "[]", "<|>", "/*cursor*/", "#cursor#"]
        for marker in cursor_markers:
            if marker in code:
                code = code.replace(marker, "")
                cursor_pos = find_cursor_position(code)
                break
    
    # Split at cursor position
    prefix = code[:cursor_pos]
    suffix = code[cursor_pos:]
    
    return prefix, suffix


def parse_code_structure(code: str, language: str) -> Dict[str, Any]:
    """
    Parse code to extract structure and important elements.
    
    Args:
        code: Code to parse
        language: Programming language
        
    Returns:
        Dict with code structure information
    """
    structure = {
        "language": language,
        "imports": [],
        "functions": [],
        "classes": [],
        "variables": [],
        "total_lines": code.count("\n") + 1
    }
    
    # Simple regex-based parsing for different languages
    if language == "python":
        # Extract imports
        import_pattern = r"^(?:from\s+[\w.]+\s+)?import\s+[\w.,\s*]+(?:\s+as\s+[\w]+)?$"
        structure["imports"] = re.findall(import_pattern, code, re.MULTILINE)
        
        # Extract function definitions
        function_pattern = r"def\s+(\w+)\s*\("
        structure["functions"] = re.findall(function_pattern, code)
        
        # Extract class definitions
        class_pattern = r"class\s+(\w+)"
        structure["classes"] = re.findall(class_pattern, code)
        
    elif language in ["javascript", "typescript"]:
        # Extract imports
        import_pattern = r"(?:import|require)\s*\(?[\w\s,{}*]+"
        structure["imports"] = re.findall(import_pattern, code)
        
        # Extract function definitions (including arrow functions)
        function_pattern = r"(?:function\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s*)?(?:\([^)]*\)|\w+)\s*=>)"
        matches = re.findall(function_pattern, code)
        structure["functions"] = [m[0] or m[1] for m in matches if m[0] or m[1]]
        
        # Extract class definitions
        class_pattern = r"class\s+(\w+)"
        structure["classes"] = re.findall(class_pattern, code)
    
    # Extract trailing whitespace or incomplete expressions
    last_line = code.split('\n')[-1] if '\n' in code else code
    structure["trailing_whitespace"] = last_line.endswith(' ') or last_line.endswith('\t')
    
    # Check for incomplete structures
    open_parens = code.count('(') - code.count(')')
    open_brackets = code.count('[') - code.count(']')
    open_braces = code.count('{') - code.count('}')
    structure["incomplete_structures"] = {
        "parentheses": open_parens,
        "brackets": open_brackets,
        "braces": open_braces
    }
    
    return structure


def optimize_prompt_for_completion(prefix: str, suffix: str) -> Tuple[str, str]:
    """
    Optimize prompt for code completion by focusing on relevant context.
    
    Args:
        prefix: Code before cursor
        suffix: Code after cursor
        
    Returns:
        Tuple of (optimized_prefix, optimized_suffix)
    """
    # For very long prefixes, keep the most relevant parts
    if len(prefix) > 5000:
        # Keep the beginning for context
        beginning = prefix[:1000]
        
        # Keep the end as it's most relevant for completion
        ending = prefix[-4000:]
        
        # Create summary marker
        marker = "\n# ... [middle code omitted for brevity] ...\n"
        
        # Combine to create optimized prefix
        optimized_prefix = beginning + marker + ending
    else:
        optimized_prefix = prefix
    
    # For very long suffixes, keep the most relevant parts
    if len(suffix) > 2000:
        # Keep the beginning of the suffix as it's most relevant
        beginning = suffix[:1500]
        
        # Keep some of the end for context
        ending = suffix[-500:]
        
        # Create summary marker
        marker = "\n# ... [remaining code omitted for brevity] ...\n"
        
        # Combine to create optimized suffix
        optimized_suffix = beginning + marker + ending
    else:
        optimized_suffix = suffix
    
    return optimized_prefix, optimized_suffix


def count_tokens(text: str, tokenizer) -> int:
    """
    Count tokens in text using the provided tokenizer.
    
    Args:
        text: Text to tokenize
        tokenizer: Tokenizer to use
        
    Returns:
        Number of tokens
    """
    try:
        tokens = tokenizer.encode(text)
        return len(tokens)
    except Exception as e:
        logger.warning(f"Error counting tokens: {e}")
        # Fallback to simple approximation (avg. 4 chars per token)
        return len(text) // 4


def create_file_hash(file_path: str) -> str:
    """
    Create a hash of a file's contents.
    
    Args:
        file_path: Path to the file
        
    Returns:
        SHA-256 hash of file contents
    """
    hash_sha256 = hashlib.sha256()
    
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        logger.error(f"Error creating file hash: {e}")
        return ""