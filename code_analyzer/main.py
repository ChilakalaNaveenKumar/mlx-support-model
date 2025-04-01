#!/usr/bin/env python3
"""
Code analyzer that builds a hierarchical mind map of project components.
"""

import os
import sys
import argparse
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("code_analyzer")

# Import MLX model service
from mlx_support_model.services.model_service import ModelService
from mlx_support_model.services.cache_service import CacheService

# Import our analyzer components
from code_analyzer.analyzer import CodeAnalyzer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze code and generate a mind map.")
    parser.add_argument("project_path", help="Path to the project directory")
    parser.add_argument("--output", "-o", help="Output directory for mind map")
    parser.add_argument("--model", "-m", help="Model to use for analysis")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    # Set log level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Initialize cache and model services
    cache_service = CacheService()
    model_service = ModelService(cache_service=cache_service)
    
    # Load model
    model_path = args.model or "mlx-community/Qwen2.5-Coder-32B-Instruct-6bit"
    print(f"Loading model: {model_path}")
    model_loaded = model_service.load_model(model_path)
    
    if not model_loaded:
        print("Failed to load model")
        return 1
    
    print("Model loaded successfully")
    
    # Initialize code analyzer
    analyzer = CodeAnalyzer(model_service)
    
    # Analyze project
    print(f"Analyzing project: {args.project_path}")
    output_path = analyzer.analyze_project(args.project_path, args.output)
    
    print(f"Analysis complete! Mind map saved to: {output_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())