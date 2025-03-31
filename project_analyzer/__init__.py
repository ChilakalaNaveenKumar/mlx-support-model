import os
import fnmatch
import logging
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)

class ProjectReaderService:
    """
    Service for reading all files in a project directory structure.
    Traverses directories recursively and reads all file contents.
    """
    
    # Common directories to ignore
    IGNORED_DIRS = [
        'target', 'build', 'dist', 'node_modules', 'venv', '.git', 
        '.idea', '.vscode', '__pycache__', '.gradle', 'bin', 'obj',
        'out', 'Debug', 'Release', '.svn', '.hg', '.DS_Store',
        'bower_components', 'coverage', 'tmp', 'temp'
    ]
    
    # Common file patterns to ignore
    IGNORED_FILE_PATTERNS = [
        '*.pyc', '*.pyo', '*.class', '*.jar', '*.war', '*.dll', '*.exe',
        '*.o', '*.so', '*.a', '*.lib', '*.zip', '*.tar', '*.gz', '*.rar',
        '*.log', '*.bak', '*.swp', '*.tmp', '*~', 'Thumbs.db', 'desktop.ini',
        '*.min.js', '*.min.css'
    ]
    
    # Important config files to keep
    IMPORTANT_CONFIG_FILES = [
        'package.json', 'requirements.txt', 'pom.xml', 'build.gradle',
        'Gemfile', 'Cargo.toml', 'go.mod', 'composer.json', 'setup.py',
        '.gitignore', 'Dockerfile', 'docker-compose.yml', 'tsconfig.json',
        'webpack.config.js', '.env.example', 'Makefile', 'CMakeLists.txt'
    ]
    
    def __init__(self, project_path: str):
        """
        Initialize the project reader service.
        
        Args:
            project_path: Path to the project directory
        """
        self.project_path = os.path.abspath(project_path)
        logger.info(f"Initialized project reader for path: {self.project_path}")
    
    def should_ignore_path(self, path: str) -> bool:
        """
        Check if a path should be ignored.
        
        Args:
            path: Path to check
            
        Returns:
            True if path should be ignored, False otherwise
        """
        path_basename = os.path.basename(path)
        
        # Always keep important config files
        if path_basename in self.IMPORTANT_CONFIG_FILES:
            return False
        
        # Ignore hidden directories/files (starting with '.')
        if path_basename.startswith('.') and path_basename not in ['.gitignore', '.dockerignore']:
            return True
            
        # Ignore common build/temporary directories
        if os.path.isdir(path) and path_basename in self.IGNORED_DIRS:
            return True
            
        # Ignore files matching ignored patterns
        if os.path.isfile(path):
            for pattern in self.IGNORED_FILE_PATTERNS:
                if fnmatch.fnmatch(path_basename, pattern):
                    return True
                    
        return False
    
    def read_project(self) -> List[Tuple[str, str]]:
        """
        Read all files in the project directory and subdirectories.
        
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
        
        logger.info(f"Read {len(project_files)} files from project")
        return project_files
    
    def _read_file(self, file_path: str) -> str:
        """
        Read content from a file with error handling.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File content as string
            
        Raises:
            Exception: If file cannot be read
        """
        try:
            # Try to read as text with UTF-8 encoding
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                return f.read()
        except UnicodeDecodeError:
            # If UTF-8 fails, try binary mode
            try:
                with open(file_path, 'rb') as f:
                    binary_content = f.read()
                    # For binary files, just return a placeholder
                    return f"[BINARY CONTENT: {len(binary_content)} bytes]"
            except Exception as e:
                logger.error(f"Error reading binary file {file_path}: {e}")
                raise
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise