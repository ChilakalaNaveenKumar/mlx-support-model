"""
Prompt utility functions.
Provides tools for prompt formatting and optimization.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union

from mlx_support_model.config import FIM_SETTINGS, CHAT_SETTINGS

logger = logging.getLogger(__name__)


def optimize_prompt_for_completion(prefix: str, suffix: str) -> Tuple[str, str]:
    """
    Optimize prompt for code completion by focusing on relevant context.
    
    Args:
        prefix: Code before cursor
        suffix: Code after cursor
        
    Returns:
        Tuple of (optimized_prefix, optimized_suffix)
    """
    # For very long prefixes, keep the most relevant parts
    if len(prefix) > 5000:
        # Keep the beginning for context
        beginning = prefix[:1000]
        
        # Keep the end as it's most relevant for completion
        ending = prefix[-4000:]
        
        # Create summary marker
        marker = "\n# ... [middle code omitted for brevity] ...\n"
        
        # Combine to create optimized prefix
        optimized_prefix = beginning + marker + ending
    else:
        optimized_prefix = prefix
    
    # For very long suffixes, keep the most relevant parts
    if len(suffix) > 2000:
        # Keep the beginning of the suffix as it's most relevant
        beginning = suffix[:1500]
        
        # Keep some of the end for context
        ending = suffix[-500:]
        
        # Create summary marker
        marker = "\n# ... [remaining code omitted for brevity] ...\n"
        
        # Combine to create optimized suffix
        optimized_suffix = beginning + marker + ending
    else:
        optimized_suffix = suffix
    
    return optimized_prefix, optimized_suffix


def format_fim_prompt(prefix: str, suffix: str) -> str:
    """
    Format a prompt for Fill-in-Middle (FIM) mode.
    
    Args:
        prefix: Content before the cursor/gap
        suffix: Content after the cursor/gap
        
    Returns:
        Formatted FIM prompt
    """
    # Format with separator
    separator = FIM_SETTINGS['prefix_suffix_separator']
    fim_prompt = f"{prefix}{separator}{suffix}"
    return fim_prompt


def format_chat_prompt(messages: List[Dict[str, str]], tokenizer) -> str:
    """
    Format messages for chat mode.
    
    Args:
        messages: List of messages with role and content
        tokenizer: Tokenizer to use for formatting
        
    Returns:
        Formatted chat prompt
    """
    try:
        # Use the model's chat template if available
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return formatted_prompt
    except Exception as e:
        logger.warning(f"Error applying chat template: {e}. Using fallback format.")
        
        # Fallback to a simple format if chat template not available
        formatted_prompt = ""
        
        # Add system message if present
        system_messages = [msg for msg in messages if msg["role"] == "system"]
        if system_messages:
            formatted_prompt += f"System: {system_messages[0]['content']}\n\n"
        
        # Add conversation messages
        for msg in messages:
            if msg["role"] != "system":  # Skip system messages here
                role = msg["role"].capitalize()
                content = msg["content"]
                formatted_prompt += f"{role}: {content}\n\n"
        
        # Add final marker for assistant
        formatted_prompt += "Assistant: "
        
        return formatted_prompt


def format_code_prompt(code: str, language: str, instruction: str) -> str:
    """
    Format a prompt for code-related tasks.
    
    Args:
        code: Code content
        language: Programming language
        instruction: Instruction for the model
        
    Returns:
        Formatted code prompt
    """
    prompt = f"""{instruction}

```{language}
{code}
```

"""
    return prompt


def create_few_shot_prompt(examples: List[Dict[str, str]], query: str, task_description: str) -> str:
    """
    Create a few-shot prompt with examples.
    
    Args:
        examples: List of example dictionaries (input/output pairs)
        query: Current query to answer
        task_description: Description of the task
        
    Returns:
        Formatted few-shot prompt
    """
    prompt = f"{task_description}\n\n"
    
    # Add examples
    for i, example in enumerate(examples):
        prompt += f"Example {i+1}:\n"
        prompt += f"Input: {example['input']}\n"
        prompt += f"Output: {example['output']}\n\n"
    
    # Add current query
    prompt += f"Now, please answer this query:\n"
    prompt += f"Input: {query}\n"
    prompt += f"Output: "
    
    return prompt


def create_system_prompt(role: str, capabilities: List[str], constraints: List[str]) -> str:
    """
    Create a system prompt with role, capabilities, and constraints.
    
    Args:
        role: Role description
        capabilities: List of capabilities
        constraints: List of constraints
        
    Returns:
        Formatted system prompt
    """
    prompt = f"You are {role}.\n\n"
    
    if capabilities:
        prompt += "Capabilities:\n"
        for capability in capabilities:
            prompt += f"- {capability}\n"
        prompt += "\n"
    
    if constraints:
        prompt += "Constraints:\n"
        for constraint in constraints:
            prompt += f"- {constraint}\n"
        prompt += "\n"
    
    return prompt.strip()