"""
Ollama service for language models.
Handles core model operations including loading, unloading, and text generation via Ollama.
"""

import logging
import json
import requests
from typing import Dict, Any, Optional, Tuple, List, Union

# Import necessary services and utilities
from mlx_support_model.config import (
    DEFAULT_GENERATION_PARAMS,
    CONTEXT_SETTINGS
)

from mlx_support_model.services.cache_service import CacheService
from mlx_support_model.services.utils.token_utils import estimate_tokens

logger = logging.getLogger(__name__)

# Ollama API settings
OLLAMA_API_BASE = "http://localhost:11434/api"
OLLAMA_MODELS_ENDPOINT = f"{OLLAMA_API_BASE}/tags"
OLLAMA_GENERATE_ENDPOINT = f"{OLLAMA_API_BASE}/generate"
OLLAMA_CHAT_ENDPOINT = f"{OLLAMA_API_BASE}/chat"
OLLAMA_EMBEDDINGS_ENDPOINT = f"{OLLAMA_API_BASE}/embeddings"

class OllamaService:
    """
    Handles core Ollama operations including loading, unloading, and text generation.
    """
    
    def __init__(self, 
                cache_service: Optional[CacheService] = None,
                verbose: bool = False):
        """
        Initialize the Ollama service.
        
        Args:
            cache_service: Optional cache service for response caching
            verbose: Whether to enable verbose logging
        """
        # Set up logging
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            
        # Initialize cache service if not provided
        self.cache_service = cache_service or CacheService()
        
        # Model state
        self.current_model = None
        
        logger.info("Ollama service initialized")
    
    def is_available(self) -> bool:
        """
        Check if Ollama is available by making a simple API call.
        
        Returns:
            Boolean indicating Ollama availability
        """
        try:
            response = requests.get(OLLAMA_MODELS_ENDPOINT, timeout=5)
            if response.status_code == 200:
                logger.info("Ollama is available")
                return True
            logger.warning(f"Ollama responded with status code: {response.status_code}")
            return False
        except Exception as e:
            logger.error(f"Error checking Ollama availability: {e}")
            return False
    
    def is_loaded(self) -> bool:
        """
        Check if a model is currently loaded.
        
        Returns:
            Boolean indicating if model is loaded
        """
        return self.current_model is not None
    
    def get_current_model(self) -> Optional[str]:
        """
        Get the current model name.
        
        Returns:
            Current model name or None if no model is loaded
        """
        return self.current_model
    
    def get_available_models(self) -> List[str]:
        """
        Get available models from Ollama.
        
        Returns:
            List of available model names
        """
        try:
            response = requests.get(OLLAMA_MODELS_ENDPOINT, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'models' in data:
                    # Extract model names
                    model_names = [model['name'] for model in data['models']]
                    logger.info(f"Available Ollama models: {model_names}")
                    return model_names
                else:
                    logger.warning("Unexpected response structure from Ollama")
                    return []
            else:
                logger.error(f"Failed to get models: Status code {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return []
    
    def load_model(self, model_name: str) -> bool:
        """
        Load a model.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Boolean indicating success
        """
        try:
            # Check if the model exists
            available_models = self.get_available_models()
            if not available_models:
                logger.error("Failed to retrieve available models")
                return False
                
            if model_name not in available_models:
                logger.warning(f"Model '{model_name}' not found in available models")
                logger.info(f"Available models: {available_models}")
                return False
            
            # Set the current model name
            self.current_model = model_name
            logger.info(f"Model '{model_name}' loaded")
            return True
        except Exception as e:
            logger.error(f"Error loading model '{model_name}': {e}")
            return False
    
    def unload_model(self) -> bool:
        """
        Unload the current model.
        
        Returns:
            Boolean indicating success
        """
        if not self.current_model:
            logger.info("No model loaded to unload")
            return False
            
        # For Ollama, we don't need to explicitly unload models
        # We just reset our reference
        self.current_model = None
        logger.info("Model unloaded")
        return True
    
    def _prepare_ollama_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare parameters for Ollama API.
        
        Args:
            params: Generation parameters
            
        Returns:
            Ollama-compatible parameters
        """
        # Map our parameters to Ollama's format
        ollama_params = {
            "model": self.current_model,
            "options": {}
        }
        
        # Handle temperature
        if 'temperature' in params:
            ollama_params['options']['temperature'] = params['temperature']
        
        # Handle top_p
        if 'top_p' in params:
            ollama_params['options']['top_p'] = params['top_p']
        
        # Handle top_k 
        if 'top_k' in params:
            ollama_params['options']['top_k'] = params['top_k']
        
        # Handle max_tokens - Ollama uses 'num_predict'
        if 'max_tokens' in params:
            ollama_params['options']['num_predict'] = params['max_tokens']
        
        # Handle repetition penalty - Ollama uses 'repeat_penalty' 
        if 'repetition_penalty' in params:
            ollama_params['options']['repeat_penalty'] = params['repetition_penalty']
        
        return ollama_params
    
    def generate_text(self, 
                    prompt: str, 
                    params: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate text using the Ollama API.
        
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
            
            # Estimate token count
            token_count = estimate_tokens(prompt)
            logger.info(f"Estimated token count: {token_count}")
            
            # Prepare Ollama parameters
            ollama_params = self._prepare_ollama_params(generation_params)
            ollama_params['prompt'] = prompt
            
            # Make API call
            logger.info("Generating response from Ollama...")
            
            # Option 1: Stream response and accumulate
            response_text = ""
            
            try:
                with requests.post(
                    OLLAMA_GENERATE_ENDPOINT,
                    json=ollama_params,
                    stream=True,
                    timeout=120
                ) as response:
                    if response.status_code != 200:
                        logger.error(f"Ollama API error: {response.status_code}")
                        return f"Error: Ollama API returned status code {response.status_code}"
                    
                    # Process the streaming response
                    buffer = b""
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            buffer += chunk
                            try:
                                # Try to extract valid JSON objects from the buffer
                                while buffer:
                                    # Find JSON object
                                    try:
                                        # Try to find a complete JSON object
                                        obj_end = buffer.find(b"}\n")
                                        if obj_end == -1:
                                            break
                                        
                                        # Extract and parse the JSON object
                                        json_str = buffer[:obj_end+1].decode('utf-8')
                                        data = json.loads(json_str)
                                        
                                        # Update the buffer to remove the processed JSON object
                                        buffer = buffer[obj_end+2:]
                                        
                                        # Extract response and append to result
                                        if 'response' in data:
                                            response_text += data['response']
                                            
                                        # Check if this is the final response
                                        if 'done' in data and data['done']:
                                            break
                                    except json.JSONDecodeError:
                                        # If we can't decode JSON, try to find the next valid object
                                        next_obj = buffer.find(b"{\n", 1)
                                        if next_obj != -1:
                                            buffer = buffer[next_obj:]
                                        else:
                                            # If no valid object start found, wait for more data
                                            break
                            except Exception as e:
                                logger.error(f"Error processing chunk: {e}")
                
                # Cache the result
                if response_text:
                    self.cache_service.add_result(prompt, generation_params, response_text)
                    return response_text
                else:
                    logger.error("No response from Ollama")
                    return "Error: No response from Ollama API"
                    
            except Exception as e:
                logger.error(f"Error with streaming response: {e}")
                
                # Fall back to non-streaming approach
                logger.info("Falling back to non-streaming API call")
                response = requests.post(
                    OLLAMA_GENERATE_ENDPOINT,
                    json=ollama_params,
                    timeout=120
                )
                
                if response.status_code != 200:
                    logger.error(f"Ollama API error: {response.status_code}")
                    return f"Error: Ollama API returned status code {response.status_code}"
                
                # Process the response
                try:
                    data = response.json()
                    if 'response' in data:
                        result = data['response']
                        
                        # Cache the result
                        self.cache_service.add_result(prompt, generation_params, result)
                        
                        return result
                    else:
                        logger.error(f"Unexpected response structure: {data}")
                        return "Error: Unexpected response from Ollama API"
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e}")
                    logger.error(f"Response content: {response.text[:100]}...")
                    return "Error: Invalid JSON response from Ollama API"
                
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return f"Error generating text: {str(e)}"
    
    def chat_completion(self, 
                       messages: List[Dict[str, str]], 
                       params: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a chat completion using the Ollama API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            params: Optional generation parameters
            
        Returns:
            Generated assistant response
        """
        if not self.is_loaded():
            return "Error: No model loaded"
        
        try:
            # Set up generation parameters
            generation_params = DEFAULT_GENERATION_PARAMS.copy()
            if params:
                generation_params.update(params)
            
            # First check cache
            cache_key = json.dumps(messages) + json.dumps(generation_params, sort_keys=True)
            cached_result = self.cache_service.get_cached_result(cache_key, generation_params)
            if cached_result:
                logger.info("Using cached chat result")
                return cached_result
            
            # Prepare Ollama parameters
            ollama_params = self._prepare_ollama_params(generation_params)
            ollama_params['model'] = self.current_model
            ollama_params['messages'] = messages
            
            # Make API call
            logger.info("Generating chat response from Ollama...")
            
            try:
                # First try the chat endpoint (newer Ollama versions)
                response = requests.post(
                    OLLAMA_CHAT_ENDPOINT,
                    json=ollama_params,
                    timeout=120
                )
                
                if response.status_code != 200:
                    # If chat endpoint fails, fall back to generate endpoint
                    logger.warning(f"Chat endpoint failed, falling back to generate endpoint")
                    
                    # Convert messages to a prompt
                    prompt = self._messages_to_prompt(messages)
                    return self.generate_text(prompt, params)
                
                # Parse response
                data = response.json()
                if 'message' in data and 'content' in data['message']:
                    result = data['message']['content']
                    
                    # Cache the result
                    self.cache_service.add_result(cache_key, generation_params, result)
                    
                    return result
                else:
                    logger.error(f"Unexpected response structure: {data}")
                    return "Error: Unexpected response from Ollama API"
                    
            except Exception as e:
                logger.warning(f"Error with chat endpoint: {e}")
                
                # Fall back to generate endpoint
                logger.info("Falling back to generate endpoint")
                prompt = self._messages_to_prompt(messages)
                return self.generate_text(prompt, params)
                
        except Exception as e:
            logger.error(f"Error generating chat response: {e}")
            return f"Error generating chat response: {str(e)}"
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert chat messages to a prompt string.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Formatted prompt string
        """
        prompt = ""
        
        # Extract system message if present
        system_messages = [msg for msg in messages if msg["role"] == "system"]
        if system_messages:
            prompt += f"System: {system_messages[0]['content']}\n\n"
        
        # Add other messages
        for msg in messages:
            if msg["role"] != "system":  # Skip system messages here
                role = msg["role"].capitalize()
                content = msg["content"]
                prompt += f"{role}: {content}\n\n"
        
        # Add final assistant marker
        prompt += "Assistant: "
        
        return prompt
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text using Ollama API.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of embedding values or empty list on failure
        """
        if not self.is_loaded():
            logger.error("No model loaded")
            return []
        
        try:
            # Prepare request
            params = {
                "model": self.current_model,
                "prompt": text
            }
            
            # Make API call
            logger.info("Getting embedding from Ollama...")
            response = requests.post(
                OLLAMA_EMBEDDINGS_ENDPOINT,
                json=params,
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"Ollama API error: {response.status_code}")
                return []
            
            # Parse response
            data = response.json()
            if 'embedding' in data:
                return data['embedding']
            else:
                logger.error(f"Unexpected response structure: {data}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return []