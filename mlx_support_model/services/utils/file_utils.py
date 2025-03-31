"""
File utility functions.
Provides tools for file handling operations.
"""

import os
import logging
import hashlib
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)


def read_file(file_path: str) -> str:
    """
    Read content from a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File content as string
        
    Raises:
        FileNotFoundError: If file doesn't exist
        IOError: If file can't be read
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        logger.debug(f"Read {len(content)} characters from {file_path}")
        return content
    except UnicodeDecodeError:
        logger.warning(f"UTF-8 decode error for {file_path}, trying with latin-1")
        # Try with a different encoding
        with open(file_path, 'r', encoding='latin-1') as f:
            content = f.read()
        return content
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        raise IOError(f"Error reading file: {str(e)}")


def write_file(content: str, file_path: str) -> None:
    """
    Write content to a file.
    
    Args:
        content: Content to write
        file_path: Path to the output file
        
    Raises:
        IOError: If file can't be written
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.debug(f"Wrote {len(content)} characters to {file_path}")
    except Exception as e:
        logger.error(f"Error writing to file {file_path}: {e}")
        raise IOError(f"Error writing to file: {str(e)}")


def create_file_hash(file_path: str) -> str:
    """
    Create a hash of a file's contents.
    
    Args:
        file_path: Path to the file
        
    Returns:
        SHA-256 hash of file contents
        
    Raises:
        FileNotFoundError: If file doesn't exist
        IOError: If file can't be read
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    hash_sha256 = hashlib.sha256()
    
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        logger.error(f"Error creating file hash for {file_path}: {e}")
        raise IOError(f"Error creating file hash: {str(e)}")


def get_file_size(file_path: str) -> int:
    """
    Get file size in bytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Size in bytes
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    return os.path.getsize(file_path)


def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get comprehensive file information.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dict with file information
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    stat_info = os.stat(file_path)
    
    file_info = {
        "path": os.path.abspath(file_path),
        "name": os.path.basename(file_path),
        "directory": os.path.dirname(os.path.abspath(file_path)),
        "extension": os.path.splitext(file_path)[1],
        "size_bytes": stat_info.st_size,
        "size_human": format_file_size(stat_info.st_size),
        "modified": stat_info.st_mtime,
        "accessed": stat_info.st_atime,
        "created": stat_info.st_ctime
    }
    
    # Try to add hash
    try:
        file_info["hash"] = create_file_hash(file_path)
    except Exception as e:
        logger.warning(f"Could not create hash for {file_path}: {e}")
        file_info["hash"] = None
    
    return file_info


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Human-readable size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def list_files(directory: str, pattern: Optional[str] = None) -> list:
    """
    List files in a directory, optionally filtered by pattern.
    
    Args:
        directory: Directory to list files from
        pattern: Optional glob pattern to filter files
        
    Returns:
        List of file paths
        
    Raises:
        FileNotFoundError: If directory doesn't exist
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
        
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"Not a directory: {directory}")
    
    if pattern:
        import glob
        return glob.glob(os.path.join(directory, pattern))
    else:
        return [os.path.join(directory, f) for f in os.listdir(directory)
                if os.path.isfile(os.path.join(directory, f))]


def ensure_directory(directory: str) -> bool:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path
        
    Returns:
        True if directory exists or was created successfully
    """
    try:
        os.makedirs(directory, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Error creating directory {directory}: {e}")
        return False