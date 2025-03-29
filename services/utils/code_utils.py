"""
Code utility functions.
Provides tools for code analysis and manipulation.
"""

import os
import re
import logging
from typing import Dict, Any, Optional, Tuple, List, Union

logger = logging.getLogger(__name__)


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
        '.r': 'r',
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


def get_indentation(code: str) -> str:
    """
    Detect the indentation style used in the code.
    
    Args:
        code: Code to analyze
        
    Returns:
        Indentation string (spaces or tab)
    """
    # Look for indentation in the code
    lines = code.split("\n")
    indentations = []
    
    for line in lines:
        if line.strip():  # Skip empty lines
            # Count leading spaces/tabs
            leading_whitespace = len(line) - len(line.lstrip())
            if leading_whitespace > 0:
                indentations.append(line[:leading_whitespace])
    
    if not indentations:
        # Default to 4 spaces if no indentation detected
        return "    "
    
    # Find the most common indentation
    from collections import Counter
    counter = Counter(indentations)
    return counter.most_common(1)[0][0]


def extract_function_context(code: str, function_name: str, language: str = "python") -> Optional[str]:
    """
    Extract a function and its context from the code.
    
    Args:
        code: Code to extract from
        function_name: Name of the function to extract
        language: Programming language
        
    Returns:
        Function code with context or None if not found
    """
    if language == "python":
        # For Python, extract the function and any docstring/comments above it
        pattern = r'((?:^\s*#[^\n]*\n)*)\s*def\s+' + re.escape(function_name) + r'\s*\([^)]*\):\s*(?:\n(?:\s+[^\n]*\n?)+)*'
        match = re.search(pattern, code, re.MULTILINE)
        if match:
            return match.group(0)
    elif language in ["javascript", "typescript"]:
        # For JS/TS, extract function and any JSDoc/comments
        pattern = r'((?:^\s*//[^\n]*\n)*|(?:^\s*/\*[\s\S]*?\*/\s*\n)*)?\s*(?:function\s+' + re.escape(function_name) + r'\s*\([^)]*\)|const\s+' + re.escape(function_name) + r'\s*=\s*(?:async\s*)?\([^)]*\)\s*=>)\s*\{(?:\n(?:[^{}]*|(?:\{[^{}]*\}))*\n?)*\}'
        match = re.search(pattern, code, re.MULTILINE)
        if match:
            return match.group(0)
    
    return None