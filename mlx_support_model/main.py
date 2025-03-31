#!/usr/bin/env python3
"""
Main entry point for MLX model-based file processing.
This slim launcher delegates to the Application class.
"""

import sys
import logging
from mlx_support_model.services.cli_service import CLIService
from mlx_support_model.services.application import Application

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")


def main() -> int:
    """
    Main application entry point.
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        # Parse command-line arguments
        cli_service = CLIService()
        args = cli_service.parse_args()
        
        # Create and run application
        app = Application(args)
        return app.run()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())