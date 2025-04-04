"""
File processor module for handling file operations.
Provides functions for reading files, saving Q&A pairs, and finding files.
"""

import os
import json
import csv
import logging
from typing import List, Dict, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def read_file(file_path: str) -> str:
    """
    Read content from a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File content as string
    """
    if not os.path.exists(file_path):
        logger.error(f"File does not exist: {file_path}")
        return ""
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except UnicodeDecodeError:
        # Try with a different encoding
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
            logger.warning(f"File {file_path} opened with latin-1 encoding")
            return content
        except Exception as e:
            logger.error(f"Failed to read file {file_path} with latin-1 encoding: {e}")
            return ""
    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {e}")
        return ""


def get_file_extension(file_path: str) -> str:
    """
    Get the extension of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File extension (including the dot)
    """
    _, ext = os.path.splitext(file_path)
    return ext


def save_qa_pairs(qa_pairs: List[Dict[str, str]], output_path: str, format_type: str) -> bool:
    """
    Save Q&A pairs to a file.
    
    Args:
        qa_pairs: List of Q&A pairs to save
        output_path: Path to save the file
        format_type: Output format (jsonl, json, csv)
        
    Returns:
        Boolean indicating success
    """
    if not qa_pairs:
        logger.warning("No Q&A pairs to save")
        return False
        
    try:
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        if format_type.lower() == "jsonl":
            with open(output_path, 'w', encoding='utf-8') as f:
                for pair in qa_pairs:
                    f.write(json.dumps(pair) + '\n')
                    
        elif format_type.lower() == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(qa_pairs, f, indent=2)
                
        elif format_type.lower() == "csv":
            with open(output_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["question", "answer"])
                for pair in qa_pairs:
                    writer.writerow([pair.get("question", ""), pair.get("answer", "")])
        else:
            logger.error(f"Unsupported format: {format_type}")
            return False
        
        logger.info(f"Saved {len(qa_pairs)} Q&A pairs to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save Q&A pairs to {output_path}: {e}")
        return False


def find_files(directory: str, extensions: List[str]) -> List[str]:
    """
    Find all files in a directory with the specified extensions.
    
    Args:
        directory: Directory to search in
        extensions: List of file extensions to include
        
    Returns:
        List of file paths
    """
    if not os.path.isdir(directory):
        logger.error(f"Directory does not exist: {directory}")
        return []
        
    file_paths = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            # Skip hidden files
            if file.startswith('.'):
                continue
            
            # Check if file has one of the specified extensions
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
    
    return file_paths


def get_file_stat(file_path: str) -> Dict[str, Any]:
    """
    Get file statistics.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary of file statistics
    """
    if not os.path.exists(file_path):
        logger.error(f"File does not exist: {file_path}")
        return {}
        
    try:
        stat = os.stat(file_path)
        
        return {
            "size": stat.st_size,
            "modified": stat.st_mtime,
            "created": stat.st_ctime,
            "accessed": stat.st_atime
        }
    except Exception as e:
        logger.error(f"Failed to get stats for {file_path}: {e}")
        return {}


def check_file_size(file_path: str, max_size_mb: float = 5.0) -> bool:
    """
    Check if a file is within the maximum size limit.
    
    Args:
        file_path: Path to the file
        max_size_mb: Maximum file size in megabytes
        
    Returns:
        Boolean indicating if the file is within the size limit
    """
    if not os.path.exists(file_path):
        logger.error(f"File does not exist: {file_path}")
        return False
        
    try:
        size_bytes = os.path.getsize(file_path)
        size_mb = size_bytes / (1024 * 1024)
        
        return size_mb <= max_size_mb
    except Exception as e:
        logger.error(f"Failed to check file size for {file_path}: {e}")
        return False


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Human-readable size string
    """
    if size_bytes < 0:
        return "0 B"
        
    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    size = float(size_bytes)
    unit_index = 0
    
    while size >= 1024.0 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1
        
    return f"{size:.2f} {units[unit_index]}"


def is_text_file(file_path: str) -> bool:
    """
    Check if a file is a text file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Boolean indicating if the file is a text file
    """
    # List of common text file extensions
    text_extensions = [
        '.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml',
        '.yaml', '.yml', '.csv', '.ini', '.cfg', '.conf', '.sh', '.bat',
        '.ps1', '.java', '.c', '.cpp', '.h', '.cs', '.go', '.rb', '.php',
        '.ts', '.jsx', '.tsx', '.sql', '.r', '.scala', '.kt', '.swift'
    ]
    
    # Check extension
    _, ext = os.path.splitext(file_path)
    if ext.lower() in text_extensions:
        return True
        
    # If extension doesn't match, try reading a small portion of the file
    try:
        # Read the first few bytes
        with open(file_path, 'rb') as f:
            content = f.read(1024)
            
        # Check if content is printable text
        try:
            content.decode('utf-8')
            return True
        except UnicodeDecodeError:
            pass
            
        # Try another common encoding
        try:
            content.decode('latin-1')
            return True
        except UnicodeDecodeError:
            pass
            
        return False
    except Exception:
        return False