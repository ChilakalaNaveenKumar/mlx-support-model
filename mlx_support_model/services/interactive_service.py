"""
Interactive service for MLX models.
Handles interactive chat and code completion modes.
"""

import logging
from typing import Dict, Any, Optional

from mlx_support_model.services.chat_service import ChatService
from mlx_support_model.services.code_service import CodeService

logger = logging.getLogger(__name__)


class InteractiveService:
    """
    Handles interactive modes for the MLX model application.
    Provides interactive chat and code completion interfaces.
    """
    
    @staticmethod
    def run_chat_mode(chat_service: ChatService) -> None:
        """
        Run interactive chat mode.
        
        Args:
            chat_service: Initialized ChatService
        """
        print("\nWelcome to Interactive Chat Mode")
        print("Type 'exit' to quit, 'reset' to clear chat history\n")
        
        while True:
            try:
                user_input = input("> ")
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("Exiting chat mode...")
                    break
                    
                if user_input.lower() in ['reset', 'clear']:
                    chat_service.reset_chat()
                    print("Chat history cleared")
                    continue
                    
                # Generate response
                print("Generating response...")
                response = chat_service.chat(user_input)
                
                print("\n" + "-" * 80)
                print(response)
                print("-" * 80 + "\n")
                
            except KeyboardInterrupt:
                print("\nChat interrupted. Type 'exit' to quit or continue chatting.")
            except Exception as e:
                print(f"Error: {e}")
    
    @staticmethod
    def run_code_mode(code_service: CodeService) -> None:
        """
        Run interactive code completion mode.
        
        Args:
            code_service: Initialized CodeService
        """
        print("\nWelcome to Interactive Code Completion Mode")
        print("Enter your code and write '###' on a new line when ready for completion")
        print("Add '+++' on a new line followed by suffix code to use FIM mode")
        print("Type 'exit' to quit\n")
        
        while True:
            try:
                print("Enter code prefix:")
                lines = []
                suffix = None
                
                while True:
                    line = input("")
                    
                    if line == "exit":
                        print("Exiting code completion mode...")
                        return
                        
                    if line == "###":
                        break
                        
                    if line == "+++":
                        print("Enter code suffix:")
                        suffix_lines = []
                        while True:
                            suffix_line = input("")
                            if suffix_line == "###":
                                break
                            suffix_lines.append(suffix_line)
                        suffix = "\n".join(suffix_lines)
                        break
                        
                    lines.append(line)
                
                prefix = "\n".join(lines)
                
                # Skip if empty
                if not prefix.strip():
                    print("No code entered. Try again.")
                    continue
                    
                print("\nGenerating completion...")
                
                # Generate completion
                completion = code_service.complete_code(
                    code=prefix,
                    suffix=suffix
                )
                    
                print("\n" + "-" * 80)
                print(completion)
                print("-" * 80 + "\n")
                
            except KeyboardInterrupt:
                print("\nCompletion interrupted. Type 'exit' to quit or continue.")
            except Exception as e:
                print(f"Error: {e}")