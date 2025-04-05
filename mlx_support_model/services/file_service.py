"""
File service for language models.
Handles file operations including reading, writing, and format detection.
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple, List, Union

from mlx_support_model.services.model_service import ModelService
from mlx_support_model.services.ollama_service import OllamaService
from mlx_support_model.services.code_service import CodeService
from mlx_support_model.services.utils.file_utils import read_file, write_file, create_file_hash, get_file_size
from mlx_support_model.services.utils.code_utils import detect_language
from mlx_support_model.services.cache_service import CacheService

logger = logging.getLogger(__name__)


class FileService:
    """
    Handles file operations for the model interface.
    Manages reading, writing, and processing files.
    """
    
    def __init__(self, 
                model_service: Union[ModelService, OllamaService],
                code_service: Optional[CodeService] = None,
                cache_service: Optional[CacheService] = None):
        """
        Initialize the file service.
        
        Args:
            model_service: ModelService or OllamaService instance for processing
            code_service: Optional CodeService for code-specific operations
            cache_service: Optional CacheService for caching
        """
        self.model_service = model_service
        self.code_service = code_service
        self.cache_service = cache_service or CacheService()
        logger.info("File service initialized")
    
    def process_file(self, 
                   file_path: str, 
                   params: Optional[Dict[str, Any]] = None) -> str:
        """
        Process a file and generate a response.
        
        Args:
            file_path: Path to the file
            params: Optional generation parameters
            
        Returns:
            Generated response
        """
        if not self.model_service.is_loaded():
            return "Error: Model not loaded. Please load a model first."
        
        # Check cache first
        cached_file = self.cache_service.get_file(file_path) if self.cache_service else None
        
        if cached_file:
            content, _ = cached_file
            logger.info(f"Using cached file content for: {file_path}")
        else:
            # Read file
            try:
                content = read_file(file_path)
                
                # Add to cache
                if self.cache_service:
                    self.cache_service.add_file(file_path, content)
                    
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
                return f"Error reading file: {str(e)}"
        
        # Detect file type
        file_type = detect_language(file_path)
        file_name = os.path.basename(file_path)
        
        # Create appropriate prompt based on file type
        if file_type in ["python", "javascript", "typescript", "java", "cpp", "c", "csharp", "go", "ruby", "rust", "php"]:
            # Code file
            if self.code_service:
                # Try to use code service for code files
                analysis = self.code_service.analyze_code(content, file_type)
                return analysis["analysis"]
            else:
                # Fallback to general prompt
                prompt = f"""Analyze the following {file_type} code file:

Filename: {file_name}

```{file_type}
{content}
```

Provide a comprehensive analysis including:
1. Structure and organization
2. Potential bugs or issues
3. Performance considerations
4. Best practices and improvements
"""
        elif file_type in ["json", "yaml", "xml"]:
            # Data file
            prompt = f"""Analyze the following {file_type} data file:

Filename: {file_name}

```{file_type}
{content}
```

Provide a comprehensive analysis of this data structure.
"""
        elif file_type in ["markdown", "text"]:
            # Text file
            prompt = f"""Process the following {file_type} file:

Filename: {file_name}

```{file_type}
{content}
```

Provide a comprehensive analysis, summary and feedback on this content.
"""
        else:
            # Generic file
            prompt = f"""Process the following file:

Filename: {file_name}

```
{content}
```

Provide an analysis and interpretation of this content.
"""
        
        # Generate response
        logger.info(f"Processing file: {file_path} ({file_type})")
        return self.model_service.generate_text(prompt, params)
    
    def convert_file(self, 
                   file_path: str, 
                   target_format: str,
                   params: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        """
        Convert a file to a different format.
        
        Args:
            file_path: Path to the file
            target_format: Target format for conversion
            params: Optional generation parameters
            
        Returns:
            Tuple of (success, content)
        """
        if not self.model_service.is_loaded():
            return False, "Error: Model not loaded. Please load a model first."
        
        # Read file
        try:
            content = read_file(file_path)
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return False, f"Error reading file: {str(e)}"
        
        # Detect source format
        source_format = detect_language(file_path)
        file_name = os.path.basename(file_path)
        
        # Create conversion prompt
        prompt = f"""Convert the following {source_format} file to {target_format} format:

Source file: {file_name}

```{source_format}
{content}
```

Converted {target_format}:
"""
        
        # Set parameters optimized for conversion
        generation_params = {
            "temperature": 0.2,
            "top_p": 0.95,
            "repetition_penalty": 1.1,
            "max_tokens": 4000
        }
        
        # Override with user params if provided
        if params:
            generation_params.update(params)
        
        # Generate conversion
        logger.info(f"Converting {file_path} from {source_format} to {target_format}")
        result = self.model_service.generate_text(prompt, generation_params)
        
        # Process result for Ollama which might include markdown formatting
        if isinstance(self.model_service, OllamaService) and "```" in result:
            try:
                # Try to extract the converted content from code blocks
                parts = result.split("```")
                if len(parts) > 1:
                    # Check if the first line is just a format identifier
                    content_part = parts[1].strip()
                    lines = content_part.split("\n")
                    if len(lines) > 1 and lines[0].strip() in [target_format, target_format.lower()]:
                        # Remove the format identifier line
                        content_part = "\n".join(lines[1:])
                    return True, content_part
            except Exception as e:
                logger.warning(f"Error extracting converted content: {e}")
                # Fall back to original result
                return True, result
        
        return True, result
    
    def save_file(self, 
                content: str, 
                file_path: str, 
                overwrite: bool = False) -> bool:
        """
        Save content to a file.
        
        Args:
            content: Content to save
            file_path: Path to save to
            overwrite: Whether to overwrite existing file
            
        Returns:
            Boolean indicating success
        """
        # Check if file exists and should not be overwritten
        if os.path.exists(file_path) and not overwrite:
            logger.warning(f"File exists and overwrite is False: {file_path}")
            return False
        
        try:
            # Create directories if needed
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # Write file
            write_file(content, file_path)
            logger.info(f"File saved: {file_path}")
            
            # Update cache if available
            if self.cache_service:
                self.cache_service.add_file(file_path, content)
                
            return True
        except Exception as e:
            logger.error(f"Error saving file {file_path}: {e}")
            return False