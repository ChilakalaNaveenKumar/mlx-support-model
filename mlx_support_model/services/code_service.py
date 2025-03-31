"""
Code service for MLX models.
Handles code-specific functionality including FIM and code completion.
"""

import logging
from typing import Dict, Any, Optional, Tuple, List, Union

from mlx_support_model.config import FIM_SETTINGS
from mlx_support_model.services.model_service import ModelService
from mlx_support_model.services.utils.code_utils import (
    detect_language, 
    find_cursor_position,
    split_at_cursor,
    parse_code_structure
)
from mlx_support_model.services.utils.prompt_utils import (
    optimize_prompt_for_completion,
    format_fim_prompt
)

logger = logging.getLogger(__name__)


class CodeService:
    """
    Handles code-specific model interactions.
    Provides code completion and FIM (Fill-in-Middle) functionality.
    """
    
    def __init__(self, model_service: ModelService):
        """
        Initialize the code service.
        
        Args:
            model_service: ModelService instance to use for generation
        """
        self.model_service = model_service
        logger.info("Code service initialized")
    
    def detect_code_completion_context(self, code: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Detect if code needs completion and extract context.
        
        Args:
            code: Code to analyze
            
        Returns:
            Tuple of (is_code_completion, prefix, suffix)
        """
        # Simple heuristics to detect code
        code_indicators = [
            "{", "}", "def ", "class ", "function", "import ", "from ", 
            "public ", "private ", "interface ", "async", "await", "#include",
            "for ", "while ", "if ", "else ", "return ", "//", "/*", "*/", "#"
        ]
        
        is_code = any(indicator in code for indicator in code_indicators)
        if not is_code:
            return False, None, None
        
        # Check if there's a clear completion point
        # Look for incomplete code structures
        incomplete_indicators = [
            # Incomplete brackets/parentheses
            "{" if code.count("{") > code.count("}") else None,
            "(" if code.count("(") > code.count(")") else None,
            "[" if code.count("[") > code.count("]") else None,
            
            # Incomplete blocks
            "if " if "if " in code and ":" in code and not "else" in code.split("if ")[-1] else None,
            "for " if "for " in code and ":" in code else None,
            "def " if "def " in code and ":" in code else None,
            "class " if "class " in code and ":" in code else None,
            
            # Line continuation
            "\\" if code.rstrip().endswith("\\") else None,
            
            # Incomplete string
            "\"" if code.count("\"") % 2 != 0 else None,
            "'" if code.count("'") % 2 != 0 else None,
        ]
        
        is_incomplete = any(indicator for indicator in incomplete_indicators if indicator)
        
        if is_incomplete and FIM_SETTINGS['enable_auto_fim']:
            # Find a suitable split point - use the end of the input as the split
            # For real code completion, you'd typically use the cursor position
            prefix = code
            suffix = ""
            return True, prefix, suffix
        
        return False, None, None
    
    def complete_code(self, 
                    code: str, 
                    suffix: Optional[str] = None,
                    language: Optional[str] = None,
                    cursor_pos: Optional[int] = None,
                    params: Optional[Dict[str, Any]] = None) -> str:
        """
        Complete code using FIM (Fill-in-Middle) mode.
        
        Args:
            code: Code to complete (prefix)
            suffix: Optional code after the cursor/gap
            language: Programming language (detected if not provided)
            cursor_pos: Optional cursor position (end of code if not provided)
            params: Optional generation parameters
            
        Returns:
            Completed code
        """
        if not self.model_service.is_loaded():
            return "Error: Model not loaded. Please load a model first."
        
        # Detect language if not provided
        if not language:
            language = detect_language("dummy." + language if language else "dummy.py")
        
        # Determine cursor position and split code if needed
        if cursor_pos is None and suffix is None:
            cursor_pos = len(code)
            prefix = code
        elif cursor_pos is not None:
            prefix, suffix_from_cursor = split_at_cursor(code, cursor_pos)
            suffix = suffix or suffix_from_cursor
        else:
            prefix = code
        
        # Optimize prefix and suffix if they're too long
        prefix, suffix = optimize_prompt_for_completion(prefix, suffix or "")
        
        # Format for FIM
        formatted_prompt = format_fim_prompt(prefix, suffix or "")
        
        # Set parameters optimized for code completion
        generation_params = {
            "temperature": 0.2,
            "top_p": 0.95,
            "repetition_penalty": 1.2,
            "max_tokens": 2000
        }
        
        # Override with user params if provided
        if params:
            generation_params.update(params)
        
        # Generate completion
        logger.info(f"Generating code completion for {language} code")
        completion = self.model_service.generate_text(formatted_prompt, generation_params)
        
        # Extract the middle part (the completion)
        if FIM_SETTINGS['prefix_suffix_separator'] in completion:
            # Split on separator and take the part before the suffix
            completion_parts = completion.split(FIM_SETTINGS['prefix_suffix_separator'])
            completion_part = completion_parts[0]
            
            # Remove the original prefix if it's repeated in the response
            if completion_part.startswith(prefix):
                completion_part = completion_part[len(prefix):]
                
            return completion_part
        else:
            # If no separator in response, return the full completion
            # minus the prefix if it's repeated
            if completion.startswith(prefix):
                completion = completion[len(prefix):]
            return completion
    
    def analyze_code(self, 
                   code: str, 
                   language: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze code structure and provide insights.
        
        Args:
            code: Code to analyze
            language: Programming language (detected if not provided)
            
        Returns:
            Dict containing code analysis
        """
        # Detect language if not provided
        if not language:
            language = detect_language("dummy." + language if language else "dummy.py")
        
        # Parse code structure
        structure = parse_code_structure(code, language)
        
        # Generate prompts for the model to analyze
        prompt = f"""Analyze the following {language} code:

```{language}
{code}
```

Provide a comprehensive analysis including:
1. Structure and organization
2. Potential bugs or issues
3. Performance considerations
4. Best practices and improvements
"""
        
        # Generate analysis
        analysis = self.model_service.generate_text(prompt, {
            "temperature": 0.3,
            "max_tokens": 2000
        })
        
        return {
            "structure": structure,
            "analysis": analysis
        }
    
    def refactor_code(self, 
                     code: str, 
                     instructions: str,
                     language: Optional[str] = None,
                     params: Optional[Dict[str, Any]] = None) -> str:
        """
        Refactor code based on instructions.
        
        Args:
            code: Code to refactor
            instructions: Refactoring instructions
            language: Programming language (detected if not provided)
            params: Optional generation parameters
            
        Returns:
            Refactored code
        """
        if not self.model_service.is_loaded():
            return "Error: Model not loaded. Please load a model first."
        
        # Detect language if not provided
        if not language:
            language = detect_language("dummy." + language if language else "dummy.py")
        
        # Create prompt for refactoring
        prompt = f"""Refactor the following {language} code according to these instructions:
{instructions}

Original code:
```{language}
{code}
```

Refactored code:
```{language}
"""
        
        # Set parameters optimized for code refactoring
        generation_params = {
            "temperature": 0.2,
            "top_p": 0.95,
            "repetition_penalty": 1.1,
            "max_tokens": 4000
        }
        
        # Override with user params if provided
        if params:
            generation_params.update(params)
        
        # Generate refactored code
        logger.info(f"Refactoring {language} code")
        result = self.model_service.generate_text(prompt, generation_params)
        
        # Extract code from the response
        refactored_code = result.split("```")[0].strip()
        
        return refactored_code