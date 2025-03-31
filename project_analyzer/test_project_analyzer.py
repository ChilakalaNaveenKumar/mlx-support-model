#!/usr/bin/env python3
"""
Simple test script for the Project Analyzer.
"""

import os
import sys
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("project_analyzer_test")

# Import from the new package structure
from mlx_support_model.services.model_service import ModelService
from mlx_support_model.services.cache_service import CacheService

# Create a simple cache service (if your CacheService has different parameters)
class SimpleCacheService:
    def __init__(self, enable_cache=True):
        self.enable_cache = enable_cache
        self.cache = {}
    
    def get_cached_result(self, *args):
        return None
    
    def add_result(self, *args):
        pass
    
    def get_model(self, *args):
        return None, None
    
    def add_model(self, *args):
        pass

# Import the project reader
from project_analyzer.project_reader import ProjectReaderService
from project_analyzer.llm_processor import LLMProcessingService

def main():
    """Main test function."""
    # Check command line args
    if len(sys.argv) < 2:
        print("Usage: python test_project_analyzer.py <path_to_test_project>")
        return 1
        
    test_project_path = sys.argv[1]
    
    # Initialize services
    print("Initializing services...")
    cache_service = SimpleCacheService()
    model_service = ModelService(cache_service=cache_service)
    
    # Load model
    print("Loading model...")
    model_loaded = model_service.load_model("mlx-community/Qwen2.5-Coder-32B-Instruct-6bit")
    if not model_loaded:
        print("Failed to load model")
        return 1
    
    print("Model loaded successfully")
    
    # Read project files
    print(f"Reading files from {test_project_path}")
    reader = ProjectReaderService(test_project_path)
    project_files = reader.read_project()
    
    # Get already analyzed files
    results_dir = os.path.join(os.getcwd(), "analysis_results")
    os.makedirs(results_dir, exist_ok=True)
    analyzed_files = []
    for filename in os.listdir(results_dir):
        if filename.endswith("_analysis.json") and not filename == "project_context.json":
            # Extract original filename (reverse the transformation used when saving)
            original_basename = filename.replace("_analysis.json", "").replace("_", ".")
            analyzed_files.append(original_basename)
    
    print(f"Found {len(analyzed_files)} already analyzed files")
    
    # Skip files that have already been analyzed
    new_project_files = []
    for file_path, content in project_files:
        file_basename = os.path.basename(file_path)
        if file_basename not in analyzed_files:
            new_project_files.append((file_path, content))
        else:
            print(f"Skipping already analyzed file: {file_basename}")
    
    print(f"Will analyze {len(new_project_files)} new files")
    
    # Only analyze a limited number of files at once to prevent memory issues
    MAX_FILES = 10  # Adjust as needed
    new_project_files = new_project_files[:MAX_FILES]
    
    # Convert to dictionary format
    file_contents = {path: content for path, content in new_project_files}
    
    # Load existing project context if available
    context_file = os.path.join(results_dir, "project_context.json")
    project_context = None
    if os.path.exists(context_file):
        with open(context_file, 'r') as f:
            project_context = json.load(f)
            print("Loaded existing project context")
    
    # Initialize LLM processor
    llm_processor = LLMProcessingService(model_service)
    
    # Restore context if available
    if project_context:
        llm_processor.project_context = project_context
    
    # Skip SVG files and other potentially problematic files
    filtered_file_contents = {}
    for file_path, content in file_contents.items():
        file_extension = os.path.splitext(file_path)[1].lower()
        file_size = len(content)
        
        # Skip SVG files with base64 data and very large files
        if file_extension == '.svg' and 'base64' in content:
            print(f"Skipping SVG file with base64 data: {os.path.basename(file_path)}")
        elif file_size > 50000:  # Skip files larger than ~50KB
            print(f"Skipping large file ({file_size} bytes): {os.path.basename(file_path)}")
        else:
            filtered_file_contents[file_path] = content
    
    file_contents = filtered_file_contents
    
    # Analyze files
    print("Analyzing files...")
    for i, (file_path, content) in enumerate(file_contents.items()):
        print(f"Analyzing file {i+1}/{len(file_contents)}: {os.path.basename(file_path)}")
        
        try:
            # Analyze file
            analysis = llm_processor.analyze_file(file_path, content)
            
            # Save results
            file_name = os.path.basename(file_path).replace(".", "_")
            output_file = os.path.join(results_dir, f"{file_name}_analysis.json")
            
            with open(output_file, 'w') as f:
                json.dump(analysis, f, indent=2)
                
            print(f"Analysis saved to {output_file}")
            
            # Save the updated project context after each file
            with open(context_file, 'w') as f:
                json.dump(llm_processor.project_context, f, indent=2)
        except Exception as e:
            print(f"Error analyzing file {file_path}: {e}")
    
    print(f"Analysis complete. Results saved to {results_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())