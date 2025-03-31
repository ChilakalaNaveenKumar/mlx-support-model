"""
Cache service for MLX models.
Handles caching of models, responses, and file content.
"""

import os
import json
import time
import hashlib
import logging
from typing import Dict, Any, Optional, Tuple, List, Union

from mlx_support_model.config import CACHE_SETTINGS

logger = logging.getLogger(__name__)


class CacheService:
    """
    Handles caching functionality for models, responses, and files.
    Optimizes performance by avoiding duplicate operations.
    """
    
    def __init__(self, cache_dir: str = CACHE_SETTINGS['cache_dir'], enable_cache: bool = CACHE_SETTINGS['enable_cache']):
        """
        Initialize the cache service.
        
        Args:
            cache_dir: Directory for cache storage
            enable_cache: Whether caching is enabled
        """
        self.cache_dir = cache_dir
        self.enable_cache = enable_cache
        self.models_cache = {}  # Maps model_path -> (model, tokenizer, last_used)
        self.results_cache = {}  # Maps hash(input+params) -> result
        self.file_cache = {}  # Maps file_path -> (content, hash, last_modified)
        
        if not self.enable_cache:
            logger.info("Cache is disabled")
            return
            
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"Cache directory: {self.cache_dir}")
        
        # Load persistent cache if available
        self._load_persistent_cache()
    
    def _load_persistent_cache(self):
        """Load cached results from disk if available."""
        if not self.enable_cache:
            return
            
        cache_file = os.path.join(self.cache_dir, "results_cache.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    self.results_cache = json.load(f)
                logger.info(f"Loaded {len(self.results_cache)} cached results from disk")
            except Exception as e:
                logger.warning(f"Failed to load cache file: {e}")
    
    def _save_persistent_cache(self):
        """Save cached results to disk."""
        if not self.enable_cache:
            return
            
        cache_file = os.path.join(self.cache_dir, "results_cache.json")
        try:
            # Save only a limited number of entries (most recent)
            if len(self.results_cache) > CACHE_SETTINGS['max_response_cache_entries']:
                # Sort by last_used timestamp
                cache_items = [(k, v) for k, v in self.results_cache.items()]
                cache_items.sort(key=lambda x: x[1].get('timestamp', 0), reverse=True)
                
                # Keep only the most recent entries
                self.results_cache = {k: v for k, v in cache_items[:CACHE_SETTINGS['max_response_cache_entries']]}
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.results_cache, f)
            logger.debug("Saved cache to disk")
        except Exception as e:
            logger.warning(f"Failed to save cache file: {e}")
    
    def add_model(self, model_path: str, model, tokenizer):
        """
        Add a model to the cache.
        
        Args:
            model_path: Path or name of the model
            model: The model object
            tokenizer: The tokenizer object
        """
        if not self.enable_cache:
            return
            
        # Check if we need to free up space
        if len(self.models_cache) >= CACHE_SETTINGS['max_models_cached']:
            # Remove least recently used model
            lru_model = min(self.models_cache.items(), key=lambda x: x[1][2])
            logger.info(f"Removing least recently used model from cache: {lru_model[0]}")
            del self.models_cache[lru_model[0]]
        
        # Add new model to cache
        self.models_cache[model_path] = (model, tokenizer, time.time())
        logger.info(f"Added model to cache: {model_path}")
    
    def get_model(self, model_path: str) -> Tuple[Optional[Any], Optional[Any]]:
        """
        Get a model from the cache.
        
        Args:
            model_path: Path or name of the model
            
        Returns:
            Tuple of (model, tokenizer) or (None, None) if not in cache
        """
        if not self.enable_cache:
            return None, None
            
        if model_path in self.models_cache:
            # Update last used time
            model, tokenizer, _ = self.models_cache[model_path]
            self.models_cache[model_path] = (model, tokenizer, time.time())
            logger.info(f"Using cached model: {model_path}")
            return model, tokenizer
        return None, None
    
    def clear_models(self):
        """Clear all models from the cache."""
        if not self.enable_cache:
            return
            
        self.models_cache.clear()
        logger.info("Cleared model cache")
    
    def _get_cache_key(self, input_text: str, params: Dict[str, Any]) -> str:
        """
        Generate a cache key for the given input and parameters.
        
        Args:
            input_text: The input text
            params: Generation parameters
            
        Returns:
            Cache key as string
        """
        # Create a stable representation for caching
        sorted_params = json.dumps(params, sort_keys=True)
        content_hash = hashlib.md5(f"{input_text}|{sorted_params}".encode('utf-8')).hexdigest()
        return content_hash
    
    def get_cached_result(self, input_text: str, params: Dict[str, Any]) -> Optional[str]:
        """
        Get cached generation result if available.
        
        Args:
            input_text: The input text
            params: Generation parameters
            
        Returns:
            Cached result or None if not in cache
        """
        if not self.enable_cache:
            return None
            
        cache_key = self._get_cache_key(input_text, params)
        result = self.results_cache.get(cache_key)
        
        if result:
            # If result has timestamp, return the actual result
            if isinstance(result, dict) and 'result' in result:
                return result.get('result')
            # For backward compatibility with old cache format
            return result
            
        return None
    
    def add_result(self, input_text: str, params: Dict[str, Any], result: str):
        """
        Add a generation result to the cache.
        
        Args:
            input_text: The input text
            params: Generation parameters
            result: The generated result
        """
        if not self.enable_cache:
            return
            
        cache_key = self._get_cache_key(input_text, params)
        self.results_cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }
        logger.debug(f"Added result to cache with key: {cache_key[:8]}...")
        
        # Save to disk periodically (every 10 additions)
        if len(self.results_cache) % 10 == 0:
            self._save_persistent_cache()
    
    def add_file(self, file_path: str, content: str):
        """
        Add a file to the cache.
        
        Args:
            file_path: Path to the file
            content: Content of the file
        """
        if not self.enable_cache:
            return
            
        file_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        last_modified = os.path.getmtime(file_path) if os.path.exists(file_path) else time.time()
        self.file_cache[file_path] = (content, file_hash, last_modified)
        
        # Prune cache if too large
        if len(self.file_cache) > CACHE_SETTINGS['max_file_cache_entries']:
            # Remove least recently accessed files
            sorted_files = sorted(self.file_cache.items(), key=lambda x: x[1][2])
            for path, _ in sorted_files[:len(sorted_files)//4]:  # Remove 25%
                del self.file_cache[path]
                
        logger.debug(f"Added file to cache: {file_path}")
    
    def get_file(self, file_path: str) -> Optional[Tuple[str, str]]:
        """
        Get a file from the cache if it's still valid.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Tuple of (content, hash) or None if not in cache or outdated
        """
        if not self.enable_cache:
            return None
            
        if file_path in self.file_cache:
            content, file_hash, cached_mtime = self.file_cache[file_path]
            # Check if file has been modified
            if os.path.exists(file_path):
                current_mtime = os.path.getmtime(file_path)
                if current_mtime > cached_mtime:
                    logger.debug(f"File changed since cached: {file_path}")
                    return None
                # Update access time
                self.file_cache[file_path] = (content, file_hash, time.time())
                logger.debug(f"Using cached file: {file_path}")
                return content, file_hash
                
        return None