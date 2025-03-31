"""
Application class for MLX models.
Handles the main application flow and service coordination.
"""

import os
import logging
from typing import Dict, Any, Optional, List

from mlx_support_model.config import DEFAULT_MODEL, LOG_SETTINGS
from mlx_support_model.services.cache_service import CacheService
from mlx_support_model.services.model_service import ModelService
from mlx_support_model.services.chat_service import ChatService
from mlx_support_model.services.code_service import CodeService
from mlx_support_model.services.file_service import FileService
from mlx_support_model.services.cli_service import CLIService
from mlx_support_model.services.command_service import CommandFactory

logger = logging.getLogger(__name__)


class Application:
    """
    Main application class.
    Coordinates services and handles application flow.
    """
    
    def __init__(self, args=None):
        """
        Initialize the application.
        
        Args:
            args: Optional parsed command-line arguments
        """
        # Initialize logging
        self._setup_logging(args)
        
        # Parse arguments if not provided
        if args is None:
            cli_service = CLIService()
            self.args = cli_service.parse_args()
        else:
            self.args = args
            
        # Initialize services
        self.services = self._initialize_services()
        
        logger.info("Application initialized")
    
    def _setup_logging(self, args):
        """
        Set up logging configuration.
        
        Args:
            args: Command-line arguments
        """
        log_level = logging.DEBUG if args and args.verbose else getattr(logging, LOG_SETTINGS['level'])
        logging.basicConfig(
            level=log_level,
            format=LOG_SETTINGS['format'],
            filename=LOG_SETTINGS['file']
        )
    
    def _initialize_services(self) -> Dict[str, Any]:
        """
        Initialize all required services.
        
        Returns:
            Dictionary of service instances
        """
        services = {}
        
        # Create CLI service
        services['cli_service'] = CLIService()
        
        # Create cache service
        services['cache_service'] = CacheService(enable_cache=not self.args.no_cache)
        
        # Clear cache if requested
        if self.args.clear_cache:
            services['cache_service'].clear_models()
        
        # Create model service
        services['model_service'] = ModelService(
            cache_service=services['cache_service'],
            verbose=self.args.verbose
        )
        
        # Create specialized services
        services['chat_service'] = ChatService(services['model_service'])
        services['code_service'] = CodeService(services['model_service'])
        services['file_service'] = FileService(
            services['model_service'], 
            services['code_service'], 
            services['cache_service']
        )
        
        # Set custom system prompt if provided
        if self.args.system_prompt:
            services['chat_service'].set_system_prompt(self.args.system_prompt)
        
        return services
    
    def _load_model_if_needed(self) -> bool:
        """
        Load model if required for the operation.
        
        Returns:
            Boolean indicating success
        """
        model_service = self.services['model_service']
        
        # Check if operation requires model
        loading_required = (
            self.args.file is not None or
            self.args.prompt is not None or
            self.args.complete_code is not None or
            self.args.interactive
        )
        
        if loading_required and not model_service.is_loaded():
            model_path = self.args.model or DEFAULT_MODEL
            logger.info(f"Loading model: {model_path}")
            return model_service.load_model(model_path)
        
        return True
    
    def run(self) -> int:
        """
        Run the application.
        
        Returns:
            Exit code (0 for success, non-zero for error)
        """
        try:
            # Check if we need to load a model
            if not self._load_model_if_needed():
                logger.error("Failed to load required model")
                return 1
            
            # Create and execute appropriate command
            command = CommandFactory.create_command(self.args, self.services)
            return command.execute()
            
        except ValueError as e:
            # Handle case where no valid command could be determined
            logger.error(f"Command error: {e}")
            self.services['cli_service'].setup_argument_parser().print_help()
            return 1
            
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Application error: {e}")
            return 1