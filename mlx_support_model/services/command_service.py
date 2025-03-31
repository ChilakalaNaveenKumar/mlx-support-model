"""
Command service for MLX models.
Implements the Command pattern for executing different operations.
"""

import os
import logging
from typing import Dict, Any, Optional, Callable, List
from abc import ABC, abstractmethod

from services.model_service import ModelService
from services.chat_service import ChatService
from services.code_service import CodeService
from services.file_service import FileService
from services.interactive_service import InteractiveService

logger = logging.getLogger(__name__)


class Command(ABC):
    """Abstract command interface."""
    
    @abstractmethod
    def execute(self) -> int:
        """
        Execute the command.
        
        Returns:
            Exit code (0 for success, non-zero for error)
        """
        pass


class ListModelsCommand(Command):
    """Command for listing available models."""
    
    def __init__(self, print_func: Callable):
        """
        Initialize the command.
        
        Args:
            print_func: Function to print models
        """
        self.print_func = print_func
    
    def execute(self) -> int:
        """Execute the command to list models."""
        self.print_func()
        return 0


class ChatCommand(Command):
    """Command for chat mode."""
    
    def __init__(self, 
                chat_service: ChatService, 
                prompt: str,
                params: Dict[str, Any],
                output_path: Optional[str] = None,
                file_service: Optional[FileService] = None):
        """
        Initialize the command.
        
        Args:
            chat_service: ChatService instance
            prompt: Chat prompt
            params: Generation parameters
            output_path: Optional path to save output
            file_service: Optional FileService for saving output
        """
        self.chat_service = chat_service
        self.prompt = prompt
        self.params = params
        self.output_path = output_path
        self.file_service = file_service
    
    def execute(self) -> int:
        """Execute the chat command."""
        try:
            # Generate response
            result = self.chat_service.chat(self.prompt, params=self.params)
            
            # Save or print result
            if self.output_path and self.file_service:
                success = self.file_service.save_file(result, self.output_path, overwrite=True)
                if not success:
                    logger.error(f"Failed to save output to: {self.output_path}")
                    return 1
            else:
                print("\n" + "=" * 80)
                print(result)
                print("=" * 80 + "\n")
                
            return 0
        except Exception as e:
            logger.error(f"Error in chat command: {e}")
            return 1


class GenerateTextCommand(Command):
    """Command for generating text."""
    
    def __init__(self, 
                model_service: ModelService, 
                prompt: str,
                params: Dict[str, Any],
                output_path: Optional[str] = None,
                file_service: Optional[FileService] = None):
        """
        Initialize the command.
        
        Args:
            model_service: ModelService instance
            prompt: Text prompt
            params: Generation parameters
            output_path: Optional path to save output
            file_service: Optional FileService for saving output
        """
        self.model_service = model_service
        self.prompt = prompt
        self.params = params
        self.output_path = output_path
        self.file_service = file_service
    
    def execute(self) -> int:
        """Execute the generate text command."""
        try:
            # Generate response
            result = self.model_service.generate_text(self.prompt, self.params)
            
            # Save or print result
            if self.output_path and self.file_service:
                success = self.file_service.save_file(result, self.output_path, overwrite=True)
                if not success:
                    logger.error(f"Failed to save output to: {self.output_path}")
                    return 1
            else:
                print("\n" + "=" * 80)
                print(result)
                print("=" * 80 + "\n")
                
            return 0
        except Exception as e:
            logger.error(f"Error in generate text command: {e}")
            return 1


class CompleteCodeCommand(Command):
    """Command for code completion."""
    
    def __init__(self, 
                code_service: CodeService, 
                code: str,
                suffix: Optional[str] = None,
                params: Optional[Dict[str, Any]] = None,
                output_path: Optional[str] = None,
                file_service: Optional[FileService] = None):
        """
        Initialize the command.
        
        Args:
            code_service: CodeService instance
            code: Code to complete
            suffix: Optional code suffix
            params: Generation parameters
            output_path: Optional path to save output
            file_service: Optional FileService for saving output
        """
        self.code_service = code_service
        self.code = code
        self.suffix = suffix
        self.params = params
        self.output_path = output_path
        self.file_service = file_service
    
    def execute(self) -> int:
        """Execute the code completion command."""
        try:
            # Complete code
            result = self.code_service.complete_code(
                self.code,
                suffix=self.suffix,
                params=self.params
            )
            
            # Save or print result
            if self.output_path and self.file_service:
                success = self.file_service.save_file(result, self.output_path, overwrite=True)
                if not success:
                    logger.error(f"Failed to save output to: {self.output_path}")
                    return 1
            else:
                print("\n" + "=" * 80)
                print(result)
                print("=" * 80 + "\n")
                
            return 0
        except Exception as e:
            logger.error(f"Error in code completion command: {e}")
            return 1


class ProcessFileCommand(Command):
    """Command for processing a file."""
    
    def __init__(self, 
                file_service: FileService, 
                file_path: str,
                params: Optional[Dict[str, Any]] = None,
                output_path: Optional[str] = None):
        """
        Initialize the command.
        
        Args:
            file_service: FileService instance
            file_path: Path to the file
            params: Generation parameters
            output_path: Optional path to save output
        """
        self.file_service = file_service
        self.file_path = file_path
        self.params = params
        self.output_path = output_path
    
    def execute(self) -> int:
        """Execute the file processing command."""
        try:
            # Check if file exists
            if not os.path.exists(self.file_path):
                logger.error(f"File not found: {self.file_path}")
                return 1
                
            # Process file
            result = self.file_service.process_file(self.file_path, self.params)
            
            # Save or print result
            if self.output_path:
                success = self.file_service.save_file(result, self.output_path, overwrite=True)
                if not success:
                    logger.error(f"Failed to save output to: {self.output_path}")
                    return 1
            else:
                print("\n" + "=" * 80)
                print(result)
                print("=" * 80 + "\n")
                
            return 0
        except Exception as e:
            logger.error(f"Error in file processing command: {e}")
            return 1


class ConvertFileCommand(Command):
    """Command for converting a file."""
    
    def __init__(self, 
                file_service: FileService, 
                file_path: str,
                target_format: str,
                params: Optional[Dict[str, Any]] = None,
                output_path: Optional[str] = None):
        """
        Initialize the command.
        
        Args:
            file_service: FileService instance
            file_path: Path to the file
            target_format: Target format for conversion
            params: Generation parameters
            output_path: Optional path to save output
        """
        self.file_service = file_service
        self.file_path = file_path
        self.target_format = target_format
        self.params = params
        self.output_path = output_path
    
    def execute(self) -> int:
        """Execute the file conversion command."""
        try:
            # Check if file exists
            if not os.path.exists(self.file_path):
                logger.error(f"File not found: {self.file_path}")
                return 1
                
            # Convert file
            success, result = self.file_service.convert_file(
                self.file_path,
                self.target_format,
                self.params
            )
            
            if not success:
                logger.error(f"Conversion failed: {result}")
                return 1
                
            # Save or print result
            if self.output_path:
                save_success = self.file_service.save_file(result, self.output_path, overwrite=True)
                if not save_success:
                    logger.error(f"Failed to save output to: {self.output_path}")
                    return 1
            else:
                print("\n" + "=" * 80)
                print(result)
                print("=" * 80 + "\n")
                
            return 0
        except Exception as e:
            logger.error(f"Error in file conversion command: {e}")
            return 1


class InteractiveChatCommand(Command):
    """Command for interactive chat mode."""
    
    def __init__(self, chat_service: ChatService):
        """
        Initialize the command.
        
        Args:
            chat_service: ChatService instance
        """
        self.chat_service = chat_service
    
    def execute(self) -> int:
        """Execute the interactive chat command."""
        try:
            InteractiveService.run_chat_mode(self.chat_service)
            return 0
        except Exception as e:
            logger.error(f"Error in interactive chat mode: {e}")
            return 1


class InteractiveCodeCommand(Command):
    """Command for interactive code completion mode."""
    
    def __init__(self, code_service: CodeService):
        """
        Initialize the command.
        
        Args:
            code_service: CodeService instance
        """
        self.code_service = code_service
    
    def execute(self) -> int:
        """Execute the interactive code completion command."""
        try:
            InteractiveService.run_code_mode(self.code_service)
            return 0
        except Exception as e:
            logger.error(f"Error in interactive code mode: {e}")
            return 1


class CommandFactory:
    """Factory for creating command objects based on arguments."""
    
    @staticmethod
    def create_command(args, services: Dict[str, Any]) -> Command:
        """
        Create appropriate command based on command-line arguments.
        
        Args:
            args: Parsed command-line arguments
            services: Dictionary of service instances
            
        Returns:
            Command instance
            
        Raises:
            ValueError: If no valid command can be created
        """
        model_service = services.get('model_service')
        chat_service = services.get('chat_service')
        code_service = services.get('code_service')
        file_service = services.get('file_service')
        cli_service = services.get('cli_service')
        
        generation_params = {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "repetition_penalty": args.repetition_penalty
        }
        
        # List models command
        if args.list_models:
            return ListModelsCommand(cli_service.print_available_models)
        
        # Interactive mode commands
        if args.interactive:
            if args.code_mode:
                return InteractiveCodeCommand(code_service)
            else:
                return InteractiveChatCommand(chat_service)
        
        # File processing commands
        if args.file:
            if args.convert_to:
                return ConvertFileCommand(
                    file_service,
                    args.file,
                    args.convert_to,
                    generation_params,
                    args.output
                )
            else:
                return ProcessFileCommand(
                    file_service,
                    args.file,
                    generation_params,
                    args.output
                )
        
        # Code completion command
        if args.complete_code is not None:
            return CompleteCodeCommand(
                code_service,
                args.complete_code,
                args.suffix,
                generation_params,
                args.output,
                file_service
            )
        
        # Chat or text generation commands
        if args.prompt is not None:
            if args.chat:
                return ChatCommand(
                    chat_service,
                    args.prompt,
                    generation_params,
                    args.output,
                    file_service
                )
            else:
                return GenerateTextCommand(
                    model_service,
                    args.prompt,
                    generation_params,
                    args.output,
                    file_service
                )
        
        # No valid command
        raise ValueError("No valid command could be determined from arguments")