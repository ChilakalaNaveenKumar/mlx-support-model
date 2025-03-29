"""
Token utility functions.
Provides tools for token counting and management.
"""

import logging
from typing import Optional, Dict, Any, Union

from config import CONTEXT_SETTINGS

logger = logging.getLogger(__name__)


def count_tokens(text: str, tokenizer) -> int:
    """
    Count tokens in text using the provided tokenizer.
    
    Args:
        text: Text to tokenize
        tokenizer: Tokenizer to use
        
    Returns:
        Number of tokens
    """
    try:
        tokens = tokenizer.encode(text)
        return len(tokens)
    except Exception as e:
        logger.warning(f"Error counting tokens: {e}")
        # Fallback to simple approximation (avg. 4 chars per token)
        return estimate_tokens(text)


def estimate_tokens(text: str) -> int:
    """
    Estimate token count when tokenizer is not available.
    Uses a simple heuristic based on character count.
    
    Args:
        text: Text to estimate tokens for
        
    Returns:
        Estimated token count
    """
    # Simple approximation: ~4 chars per token for English text
    # This varies by language and content, but works as a fallback
    return len(text) // 4


def is_within_context_limit(text: str, tokenizer, max_length: Optional[int] = None) -> bool:
    """
    Check if text is within context limit.
    
    Args:
        text: Text to check
        tokenizer: Tokenizer to use
        max_length: Maximum context length (uses default if None)
        
    Returns:
        Boolean indicating if text is within limit
    """
    if max_length is None:
        max_length = CONTEXT_SETTINGS['default_max_length']
        
    token_count = count_tokens(text, tokenizer)
    return token_count <= max_length


def calculate_max_new_tokens(prompt: str, tokenizer, max_total_tokens: Optional[int] = None) -> int:
    """
    Calculate the maximum number of new tokens that can be generated.
    
    Args:
        prompt: Input prompt
        tokenizer: Tokenizer to use
        max_total_tokens: Maximum total tokens (uses default if None)
        
    Returns:
        Maximum number of new tokens
    """
    if max_total_tokens is None:
        max_total_tokens = CONTEXT_SETTINGS['default_max_length']
        
    prompt_tokens = count_tokens(prompt, tokenizer)
    max_new_tokens = max(max_total_tokens - prompt_tokens, 0)
    
    # Set a reasonable minimum for very long prompts
    if max_new_tokens < 100:
        max_new_tokens = 100
        
    return max_new_tokens


def truncate_to_token_limit(text: str, tokenizer, max_length: int) -> str:
    """
    Truncate text to fit within token limit.
    
    Args:
        text: Text to truncate
        tokenizer: Tokenizer to use
        max_length: Maximum token length
        
    Returns:
        Truncated text
    """
    try:
        tokens = tokenizer.encode(text)
        
        if len(tokens) <= max_length:
            return text
            
        # Truncate tokens and decode
        truncated_tokens = tokens[:max_length]
        return tokenizer.decode(truncated_tokens)
        
    except Exception as e:
        logger.warning(f"Error truncating text: {e}")
        
        # Fallback to character-based truncation
        # This is rough but better than nothing
        chars_per_token = 4  # Approximation
        max_chars = max_length * chars_per_token
        
        if len(text) <= max_chars:
            return text
            
        return text[:max_chars]


def split_text_by_tokens(text: str, tokenizer, chunk_size: int, overlap: int = 0) -> list:
    """
    Split text into chunks based on token count.
    
    Args:
        text: Text to split
        tokenizer: Tokenizer to use
        chunk_size: Token size for each chunk
        overlap: Number of overlapping tokens between chunks
        
    Returns:
        List of text chunks
    """
    try:
        tokens = tokenizer.encode(text)
        
        # If text is smaller than chunk size, return as is
        if len(tokens) <= chunk_size:
            return [text]
            
        # Create chunks with overlap
        chunks = []
        for i in range(0, len(tokens), chunk_size - overlap):
            # Take a chunk of tokens
            chunk_tokens = tokens[i:i + chunk_size]
            # Decode the chunk back to text
            chunk_text = tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            
        return chunks
        
    except Exception as e:
        logger.warning(f"Error splitting text: {e}")
        
        # Fallback to character-based chunking
        chars_per_token = 4  # Approximation
        char_chunk_size = chunk_size * chars_per_token
        char_overlap = overlap * chars_per_token
        
        # Create chunks with overlap
        chunks = []
        for i in range(0, len(text), char_chunk_size - char_overlap):
            chunk_text = text[i:i + char_chunk_size]
            chunks.append(chunk_text)
            
        return chunks