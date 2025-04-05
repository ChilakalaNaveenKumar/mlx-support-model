"""
Model service factory for creating appropriate model service instances.
Supports both Ollama and MLX backends.
"""

import logging
from typing import Dict, Any, Optional, Union

from mlx_support_model.services.cache_service import CacheService
from mlx_support_model.services.model_service import ModelService
from mlx_support_model.services.ollama_service import OllamaService
from mlx_support_model.config import LLM_PROVIDER

logger = logging.getLogger(__name__)

class ModelServiceFactory:
    """
    Factory for creating model service instances based on configuration.
    """
    
    @staticmethod
    def create_service(
        provider: Optional[str] = None,
        cache_service: Optional[CacheService] = None,
        verbose: bool = False
    ) -> Union[OllamaService, ModelService]:
        """
        Create a model service instance.
        
        Args:
            provider: Provider name ('ollama' or 'mlx')
            cache_service: Optional cache service
            verbose: Whether to enable verbose logging
            
        Returns:
            Model service instance (OllamaService or ModelService)
        """
        # Use configured provider if none specified
        provider = provider or LLM_PROVIDER
        
        # Create cache service if not provided
        if cache_service is None:
            cache_service = CacheService()
        
        # Create appropriate service based on provider
        if provider.lower() == 'ollama':
            logger.info("Creating Ollama service")
            service = OllamaService(cache_service=cache_service, verbose=verbose)
            
            # Check if Ollama is actually available
            if not service.is_available():
                logger.warning("Ollama is not available, falling back to MLX")
                return ModelService(cache_service=cache_service, verbose=verbose)
                
            return service
            
        elif provider.lower() == 'mlx':
            logger.info("Creating MLX service")
            return ModelService(cache_service=cache_service, verbose=verbose)
            
        else:
            logger.warning(f"Unknown provider: {provider}, using default (Ollama)")
            return OllamaService(cache_service=cache_service, verbose=verbose)