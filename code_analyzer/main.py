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
    parser.add_argument("--structure-only", "-s", action="store_true", help="Only analyze directory structure, skip file content analysis")
    parser.add_argument("--file", "-f", help="Analyze a single file instead of the entire project")
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    # Set log level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # If structure-only mode, don't need to load model
    if not args.structure_only:
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
    else:
        # For structure-only, we don't need the model
        print("Running in structure-only mode - skipping model loading")
        # Use a placeholder model service
        class PlaceholderModelService:
            def generate_text(self, prompt, params=None):
                return "Structure-only mode - no text generation"
                
        analyzer = CodeAnalyzer(PlaceholderModelService())
    
    # Determine output directory
    output_dir = args.output or os.getcwd()
    
    # Analyze a single file or full project
    if args.file:
        if not os.path.isfile(args.file):
            print(f"Error: File not found: {args.file}")
            return 1
            
        print(f"Analyzing single file: {args.file}")
        
        # Read file content
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return 1
            
        # Create analysis file name
        file_name = os.path.basename(args.file)
        output_file = os.path.join(output_dir, f"{file_name}_analysis.txt")
        
        # Analyze file
        analysis = analyzer.analyze_file(args.file, content)
        
        # Format and save analysis
        analysis_text = analyzer._format_analysis_for_file(args.file, analysis, content)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(analysis_text)
            
        print(f"Analysis complete! Results saved to: {output_file}")
    else:
        # Analyze full project
        print(f"Analyzing project: {args.project_path}")
        output_path = analyzer.analyze_project(args.project_path, output_dir)
        
        print(f"Analysis complete! Results saved to: {output_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())