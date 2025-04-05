"""
CLI service for language models.
Handles command-line argument parsing and setup.
"""

import argparse
import logging
from typing import Dict, Any, Optional

from mlx_support_model.config import (
    DEFAULT_MODEL, 
    ADDITIONAL_MODELS, 
    LLM_PROVIDER,
    OLLAMA_ADDITIONAL_MODELS,
    MLX_ADDITIONAL_MODELS
)

logger = logging.getLogger(__name__)


class CLIService:
    """
    Handles command-line interface for the language model application.
    Provides argument parsing and command-line setup.
    """
    
    @staticmethod
    def setup_argument_parser() -> argparse.ArgumentParser:
        """
        Set up command line argument parser.
        
        Returns:
            Configured ArgumentParser instance
        """
        parser = argparse.ArgumentParser(
            description="LLM File Processor",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        # Provider selection
        provider_group = parser.add_argument_group("Provider Selection")
        provider_group.add_argument(
            "--provider", "-p",
            choices=["ollama", "mlx"],
            default=LLM_PROVIDER,
            help=f"LLM provider to use (default: {LLM_PROVIDER})"
        )
        
        # Model selection arguments
        model_group = parser.add_argument_group("Model Selection")
        model_group.add_argument(
            "--model", "-m",
            help=f"The model to use (default: {DEFAULT_MODEL})"
        )
        model_group.add_argument(
            "--list-models", "-l",
            action="store_true",
            help="List available models and exit"
        )
        
        # File processing arguments
        file_group = parser.add_argument_group("File Processing")
        file_group.add_argument(
            "--file", "-f",
            help="Input file path to process"
        )
        file_group.add_argument(
            "--output", "-o",
            help="Output file path (default: auto-generated)"
        )
        file_group.add_argument(
            "--convert-to", "-c",
            help="Target format for conversion"
        )
        
        # Input arguments (non-file)
        input_group = parser.add_argument_group("Input")
        input_group.add_argument(
            "--prompt", "-pr",
            help="Text prompt to send to the model"
        )
        input_group.add_argument(
            "--complete-code", "-ccode",
            help="Code to complete using Fill-in-Middle"
        )
        input_group.add_argument(
            "--suffix", "-s",
            help="Code suffix for Fill-in-Middle completion"
        )
        
        # Mode selection
        mode_group = parser.add_argument_group("Operation Mode")
        mode_group.add_argument(
            "--chat", "-ch",
            action="store_true",
            help="Use chat mode"
        )
        mode_group.add_argument(
            "--system-prompt", "-sp",
            help="Custom system prompt for chat mode"
        )
        mode_group.add_argument(
            "--interactive", "-i",
            action="store_true",
            help="Start in interactive mode"
        )
        mode_group.add_argument(
            "--code-mode", "-cm",
            action="store_true",
            help="Use code completion mode"
        )
        
        # Generation parameters
        gen_group = parser.add_argument_group("Generation Parameters")
        gen_group.add_argument(
            "--max-tokens", "-mt",
            type=int,
            default=4000,
            help="Maximum number of tokens to generate"
        )
        gen_group.add_argument(
            "--temperature", "-t",
            type=float,
            default=0.7,
            help="Temperature for generation"
        )
        gen_group.add_argument(
            "--repetition-penalty", "-rp",
            type=float,
            default=1.1,
            help="Repetition penalty for generation"
        )
        
        # Debug arguments
        debug_group = parser.add_argument_group("Debug Options")
        debug_group.add_argument(
            "--verbose", "-v",
            action="store_true",
            help="Enable verbose logging"
        )
        debug_group.add_argument(
            "--no-cache", "-nc",
            action="store_true",
            help="Disable response caching"
        )
        debug_group.add_argument(
            "--clear-cache", "-clc",
            action="store_true",
            help="Clear cache before starting"
        )
        
        return parser
    
    @staticmethod
    def parse_args():
        """
        Parse command-line arguments.
        
        Returns:
            Parsed arguments
        """
        parser = CLIService.setup_argument_parser()
        return parser.parse_args()
    
    @staticmethod
    def get_generation_params(args) -> Dict[str, Any]:
        """
        Extract generation parameters from parsed arguments.
        
        Args:
            args: Parsed command-line arguments
            
        Returns:
            Dictionary of generation parameters
        """
        return {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "repetition_penalty": args.repetition_penalty
        }
    
    @staticmethod
    def print_available_models() -> None:
        """Print the list of available default models."""
        print("\nAvailable models:")
        print("-" * 50)
        print(f"Default: {DEFAULT_MODEL}")
        
        # Print Ollama models
        print("\nOllama models:")
        for key, path in OLLAMA_ADDITIONAL_MODELS.items():
            print(f"{key}: {path}")
            
        # Print MLX models
        print("\nMLX models:")
        for key, path in MLX_ADDITIONAL_MODELS.items():
            print(f"mlx_{key}: {path}")
        
        print()