"""
Application class for language models.
Handles the main application flow and service coordination.
"""

import os
import logging
from typing import Dict, Any, Optional, List

from mlx_support_model.config import DEFAULT_MODEL, LOG_SETTINGS, LLM_PROVIDER
from mlx_support_model.services.cache_service import CacheService
from mlx_support_model.services.model_service_factory import ModelServiceFactory
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
        log_level = logging.DEBUG if args and getattr(args, 'verbose', False) else getattr(logging, LOG_SETTINGS['level'])
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
        services['cache_service'] = CacheService(enable_cache=not getattr(self.args, 'no_cache', False))
        
        # Clear cache if requested
        if getattr(self.args, 'clear_cache', False):
            services['cache_service'].clear_models()
        
        # Determine provider
        provider = getattr(self.args, 'provider', LLM_PROVIDER)
        
        # Create model service using factory
        services['model_service'] = ModelServiceFactory.create_service(
            provider=provider,
            cache_service=services['cache_service'],
            verbose=getattr(self.args, 'verbose', False)
        )
        
        # Create code service first (it doesn't depend on chat service)
        services['code_service'] = CodeService(services['model_service'])
        
        # Create chat service
        services['chat_service'] = ChatService(services['model_service'])
        
        # Create file service
        services['file_service'] = FileService(
            services['model_service'], 
            services['code_service'], 
            services['cache_service']
        )
        
        # Set custom system prompt if provided
        if getattr(self.args, 'system_prompt', None):
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
            getattr(self.args, 'file', None) is not None or
            getattr(self.args, 'prompt', None) is not None or
            getattr(self.args, 'complete_code', None) is not None or
            getattr(self.args, 'interactive', False)
        )
        
        if loading_required and not model_service.is_loaded():
            model_path = getattr(self.args, 'model', None) or DEFAULT_MODEL
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
            
            # Check if we just want to list models
            if getattr(self.args, 'list_models', False):
                self.services['cli_service'].print_available_models()
                return 0
            
            # Create and execute appropriate command
            command = CommandFactory.create_command(self.args, self.services)
            return command.execute() if command else 1
            
        except ValueError as e:
            # Handle case where no valid command could be determined
            logger.error(f"Command error: {e}")
            self.services['cli_service'].setup_argument_parser().print_help()
            return 1
            
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Application error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return 1