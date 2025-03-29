"""
Chat service for MLX models.
Handles chat-specific functionality including history management.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Tuple

from config import CHAT_SETTINGS
from services.model_service import ModelService
from services.utils.prompt_utils import format_chat_prompt

logger = logging.getLogger(__name__)


class ChatService:
    """
    Handles chat interactions with the model.
    Maintains conversation history and manages chat-specific formatting.
    """
    
    def __init__(self, model_service: ModelService):
        """
        Initialize the chat service.
        
        Args:
            model_service: ModelService instance to use for generation
        """
        self.model_service = model_service
        self.chat_history = []
        self.system_prompt = CHAT_SETTINGS['system_prompt']
        
        # Load chat history if available and enabled
        if CHAT_SETTINGS['save_history']:
            self._load_history()
        
        logger.info("Chat service initialized")
    
    def _load_history(self):
        """Load chat history from file if available."""
        if not CHAT_SETTINGS['save_history']:
            return
            
        history_file = CHAT_SETTINGS['history_file']
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    self.chat_history = json.load(f)
                logger.info(f"Loaded {len(self.chat_history)} chat history entries")
            except Exception as e:
                logger.warning(f"Failed to load chat history: {e}")
    
    def _save_history(self):
        """Save chat history to file."""
        if not CHAT_SETTINGS['save_history']:
            return
            
        history_file = CHAT_SETTINGS['history_file']
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(history_file)), exist_ok=True)
            
            # Limit history size if needed
            if len(self.chat_history) > CHAT_SETTINGS['max_history_turns']:
                self.chat_history = self.chat_history[-CHAT_SETTINGS['max_history_turns']:]
            
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(self.chat_history, f)
            logger.debug("Chat history saved")
        except Exception as e:
            logger.warning(f"Failed to save chat history: {e}")
    
    def set_system_prompt(self, system_prompt: str):
        """
        Set the system prompt for the chat.
        
        Args:
            system_prompt: New system prompt
        """
        self.system_prompt = system_prompt
        logger.info("System prompt updated")
    
    def reset_chat(self):
        """
        Reset the chat history.
        """
        self.chat_history = []
        logger.info("Chat history reset")
    
    def get_history(self) -> List[Dict[str, str]]:
        """
        Get the current chat history.
        
        Returns:
            List of chat messages
        """
        return self.chat_history
    
    def add_message(self, role: str, content: str):
        """
        Add a message to the chat history.
        
        Args:
            role: Role of the message sender (user/assistant)
            content: Message content
        """
        self.chat_history.append({"role": role, "content": content})
        
        # Save history if enabled
        if CHAT_SETTINGS['save_history']:
            self._save_history()
            
        logger.debug(f"Added {role} message to history")
    
    def chat(self, 
           user_message: str, 
           system_prompt: Optional[str] = None,
           params: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a chat response.
        
        Args:
            user_message: User message
            system_prompt: Custom system prompt (uses default if None)
            params: Generation parameters
            
        Returns:
            Assistant response
        """
        if not self.model_service.is_loaded():
            return "Error: Model not loaded. Please load a model first."
        
        # Use provided system prompt or default
        system_prompt = system_prompt or self.system_prompt
        
        # Prepare messages including history
        messages = []
        
        # Add system message
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add chat history
        messages.extend(self.chat_history)
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        # Format as chat prompt
        formatted_prompt = format_chat_prompt(messages, self.model_service.tokenizer)
        
        # Add user message to history
        self.add_message("user", user_message)
        
        # Generate response
        assistant_response = self.model_service.generate_text(formatted_prompt, params)
        
        # Add assistant response to history
        self.add_message("assistant", assistant_response)
        
        return assistant_response