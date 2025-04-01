import os
import logging
from typing import List, Dict, Any, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("code_analyzer")

class ProjectScanner:
    """
    Scans a project directory for code files and reads their contents.
    """
    
    # Common directories to ignore
    IGNORED_DIRS = [
        'node_modules', 'venv', '.git', '.idea', '.vscode', '__pycache__', 
        'build', 'dist', 'target', '.gradle', 'bin', 'obj', 'out'
    ]
    
    # Common file patterns to ignore
    IGNORED_FILE_PATTERNS = [
        '*.pyc', '*.pyo', '*.class', '*.jar', '*.dll', '*.exe', '*.o', '*.so',
        '*.zip', '*.tar', '*.gz', '*.log', '*.bak', '*.swp', '*.tmp', '*~'
    ]
    
    def __init__(self, project_path: str):
        """Initialize with the path to the project directory."""
        self.project_path = os.path.abspath(project_path)
        logger.info(f"Initialized project scanner for: {self.project_path}")
    
    def should_ignore_path(self, path: str) -> bool:
        """Check if a path should be ignored during scanning."""
        path_basename = os.path.basename(path)
        
        # Ignore hidden files/dirs except some configuration files
        if path_basename.startswith('.') and path_basename not in ['.gitignore', '.dockerignore']:
            return True
            
        # Ignore common build/temp directories
        if os.path.isdir(path) and path_basename in self.IGNORED_DIRS:
            return True
            
        # Ignore files matching ignored patterns
        if os.path.isfile(path):
            for pattern in self.IGNORED_FILE_PATTERNS:
                if self._match_pattern(path_basename, pattern):
                    return True
                    
        return False
    
    def _match_pattern(self, filename: str, pattern: str) -> bool:
        """Simple pattern matching for file names."""
        if pattern.startswith('*'):
            return filename.endswith(pattern[1:])
        elif pattern.endswith('*'):
            return filename.startswith(pattern[:-1])
        else:
            return filename == pattern
    
    def scan_project(self) -> List[Tuple[str, str]]:
        """
        Scan the project directory for code files.
        
        Returns:
            List of tuples containing (file_path, file_content)
        """
        if not os.path.exists(self.project_path):
            logger.error(f"Project path does not exist: {self.project_path}")
            return []
            
        if not os.path.isdir(self.project_path):
            logger.error(f"Project path is not a directory: {self.project_path}")
            return []
        
        project_files = []
        
        # Walk through all directories and files
        for root, dirs, files in os.walk(self.project_path):
            # Modify dirs in-place to skip ignored directories
            dirs[:] = [d for d in dirs if not self.should_ignore_path(os.path.join(root, d))]
            
            for file in files:
                file_path = os.path.join(root, file)
                
                # Skip ignored files
                if self.should_ignore_path(file_path):
                    continue
                
                try:
                    # Read file content
                    content = self._read_file(file_path)
                    project_files.append((file_path, content))
                    logger.debug(f"Read file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to read file {file_path}: {e}")
        
        logger.info(f"Scanned {len(project_files)} files from project")
        return project_files
    
    def _read_file(self, file_path: str) -> str:
        """Read content from a file with error handling."""
        try:
            # Try to read as text with UTF-8 encoding
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                return f.read()
        except UnicodeDecodeError:
            # If UTF-8 fails, try binary mode and return placeholder
            with open(file_path, 'rb') as f:
                binary_content = f.read()
                return f"[BINARY CONTENT: {len(binary_content)} bytes]"
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise