"""
Model service for MLX models.
Handles core model operations including loading, unloading, and text generation.
"""

import logging
from typing import Dict, Any, Optional, Tuple, List, Union

from mlx_support_model.config import (
    DEFAULT_MODEL,
    ADDITIONAL_MODELS,
    DEFAULT_GENERATION_PARAMS,
    CONTEXT_SETTINGS
)

from mlx_support_model.services.cache_service import CacheService
from mlx_support_model.services.utils.token_utils import count_tokens, is_within_context_limit

logger = logging.getLogger(__name__)

# Try to import mlx_lm, handle gracefully if not available
try:
    import mlx.core as mx
    from mlx_lm import load, generate
    MLX_AVAILABLE = True
except ImportError:
    logger.warning("mlx_lm not installed. Please install with: pip install mlx-lm")
    MLX_AVAILABLE = False


class ModelService:
    """
    Handles core model operations including loading, unloading, and text generation.
    """
    
    def __init__(self, 
                cache_service: Optional[CacheService] = None,
                verbose: bool = False):
        """
        Initialize the model service.
        
        Args:
            cache_service: Optional cache service for model caching
            verbose: Whether to enable verbose logging
        """
        # Set up logging
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            
        # Initialize cache service if not provided
        self.cache_service = cache_service or CacheService()
        
        # Model state
        self.model = None
        self.tokenizer = None
        self.current_model_path = None
        
        logger.info("Model service initialized")
        
    def is_available(self) -> bool:
        """
        Check if MLX is available.
        
        Returns:
            Boolean indicating MLX availability
        """
        return MLX_AVAILABLE
    
    def is_loaded(self) -> bool:
        """
        Check if a model is currently loaded.
        
        Returns:
            Boolean indicating if model is loaded
        """
        return self.model is not None and self.tokenizer is not None
    
    def get_current_model(self) -> Optional[str]:
        """
        Get the current model path.
        
        Returns:
            Current model path or None if no model is loaded
        """
        return self.current_model_path
    
    def load_model(self, model_name_or_path: str = DEFAULT_MODEL) -> bool:
        """
        Load a model.
        
        Args:
            model_name_or_path: Name of default model or path to custom model
            
        Returns:
            Boolean indicating success
        """
        if not MLX_AVAILABLE:
            logger.error("MLX is not available. Cannot load model.")
            return False
            
        # First check if this is a shorthand for a default model
        if model_name_or_path in ADDITIONAL_MODELS:
            model_path = ADDITIONAL_MODELS[model_name_or_path]
            logger.info(f"Using default model: {model_name_or_path} ({model_path})")
        else:
            model_path = model_name_or_path
            logger.info(f"Using model path: {model_path}")
        
        # Check if already loaded
        if self.current_model_path == model_path and self.is_loaded():
            logger.info(f"Model already loaded: {model_path}")
            return True
            
        try:
            # Check cache first
            self.model, self.tokenizer = self.cache_service.get_model(model_path)
            
            if not self.is_loaded():
                # Not in cache, load from source
                logger.info(f"Loading model from: {model_path}")
                self.model, self.tokenizer = load(model_path)
                
                # Add to cache
                self.cache_service.add_model(model_path, self.model, self.tokenizer)
            
            # Update current model path
            self.current_model_path = model_path
            logger.info(f"Model loaded successfully: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model {model_path}: {e}")
            return False
    
    def unload_model(self) -> bool:
        """
        Unload the current model to free up memory.
        
        Returns:
            Boolean indicating success
        """
        if not self.is_loaded():
            logger.info("No model loaded to unload")
            return False
            
        try:
            # Set to None to allow garbage collection
            self.model = None
            self.tokenizer = None
            self.current_model_path = None
            logger.info("Model unloaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error unloading model: {e}")
            return False
    
    def get_available_models(self) -> Dict[str, str]:
        """
        Get available models.
        
        Returns:
            Dictionary of model names and paths
        """
        available_models = {"default": DEFAULT_MODEL}
        available_models.update(ADDITIONAL_MODELS)
        return available_models
    
    def get_best_context_length(self, input_length: int) -> int:
        """
        Determine the best context length based on input size.
        
        Args:
            input_length: Length of input in tokens
            
        Returns:
            Appropriate context length
        """
        if not CONTEXT_SETTINGS['dynamic_scaling']:
            # Use default if dynamic scaling is disabled
            return CONTEXT_SETTINGS['default_max_length']
        
        # For Qwen models, use their maximum context
        if self.current_model_path and "qwen" in self.current_model_path.lower():
            max_length = CONTEXT_SETTINGS['qwen_max_length']
        else:
            max_length = CONTEXT_SETTINGS['default_max_length']
        
        # Scale context based on input length
        if input_length < 1000:
            # For small inputs, use a modest context
            return min(4096, max_length)
        elif input_length < 4000:
            # For medium inputs, scale up
            return min(8192, max_length)
        elif input_length < 8000:
            # For large inputs, use larger context
            return min(16384, max_length)
        else:
            # For very large inputs, use maximum
            return max_length
    
    def generate_text(self, 
                    prompt: str, 
                    params: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate text using the model.
        
        Args:
            prompt: Input prompt
            params: Optional generation parameters
            
        Returns:
            Generated text
        """
        if not self.is_loaded():
            return "Error: No model loaded"
        
        try:
            # Set up generation parameters
            generation_params = DEFAULT_GENERATION_PARAMS.copy()
            if params:
                generation_params.update(params)
            
            # First check cache
            cached_result = self.cache_service.get_cached_result(prompt, generation_params)
            if cached_result:
                logger.info("Using cached result")
                return cached_result
            
            # Count tokens to optimize context window
            token_count = count_tokens(prompt, self.tokenizer)
            logger.info(f"Input token count: {token_count}")
            logger.info(f"Input prompt: {prompt}")
            
            # Adjust context length based on input size
            context_length = self.get_best_context_length(token_count)
            logger.debug(f"Using context length: {context_length}")
            
            # Extract max_tokens parameter
            max_tokens = generation_params.pop("max_tokens", 4000)
            
            # Generate text - MLX-LM API expects different params
            # The correct parameters for MLX-LM generate function may vary
            # with different versions
            try:
                # Try the standard approach first
                logger.info("Generating response...")
                response = generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    verbose=True
                )
            except TypeError as e:
                # If we get parameter errors, try a minimal approach
                logger.warning(f"Error with standard parameters: {e}. Using simplified parameters.")
                response = generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt
                )
            
            # Cache the result
            self.cache_service.add_result(prompt, generation_params, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return f"Error generating text: {str(e)}"