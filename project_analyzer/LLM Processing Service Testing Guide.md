# LLM Processing Service Testing Guide

This document describes how to test the `LLMProcessingService` implementation, which is responsible for analyzing code files using the existing MLX LLM service.

## Overview

The `LLMProcessingService` sends code files to your LLM service for analysis, maintains context between files, and parses the responses into structured data. This guide shows how to test it with your existing code.

## Setup

1. Save the `LLMProcessingService` implementation in a file called `llm_processor.py` inside your `project_analyzer` directory:

```
project_analyzer/
├── __init__.py
├── project_reader.py        # Your existing file for reading project files
├── analysis_storage.py      # Your existing file for storing analysis data
├── vector_embedding.py      # Your existing file for embedding analyses
└── llm_processor.py         # New file for the LLM processing service
```

2. Make sure your existing `model_service.py` is accessible and configured properly.

## Testing the LLM Processor

Create a simple test script in your project root:

```python
#!/usr/bin/env python3
"""
Test script for the LLM Processing Service.
Tests analyzing files with the LLM model.
"""

import os
import sys
import logging
import json
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_llm_processor")

# Import your services
from services.model_service import ModelService
from services.cache_service import CacheService
from project_analyzer.project_reader import ProjectReaderService
from project_analyzer.llm_processor import LLMProcessingService

def main():
    """Main test function."""
    # Check command line args
    if len(sys.argv) < 2:
        print("Usage: python test_llm_processor.py <path_to_test_project>")
        return 1
        
    test_project_path = sys.argv[1]
    
    if not os.path.isdir(test_project_path):
        print(f"Error: {test_project_path} is not a valid directory")
        return 1
    
    # Initialize cache service
    cache_service = CacheService()
    
    # Initialize model service
    model_service = ModelService(cache_service=cache_service)
    
    # Load the model (adjust model path as needed)
    # NOTE: Adjust this model path to your configuration
    model_loaded = model_service.load_model("mlx-community/Qwen2.5-Coder-32B-Instruct-6bit")
    
    if not model_loaded:
        print("Failed to load model")
        return 1
    
    print(f"Model loaded successfully")
    
    # Initialize project reader
    reader_service = ProjectReaderService(test_project_path)
    
    # Read a subset of files for testing (limit to 3 files to start)
    print(f"Reading files from {test_project_path}")
    project_files = reader_service.read_project()[:3]
    
    if not project_files:
        print("No files found in the project")
        return 1
        
    print(f"Read {len(project_files)} files")
    
    # Convert to dictionary
    file_contents = {path: content for path, content in project_files}
    
    # Initialize LLM processing service
    llm_processor = LLMProcessingService(model_service)
    
    # Test analyzing the files
    print("Starting file analysis...")
    
    # Create output directory for results
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each file
    for i, (file_path, content) in enumerate(file_contents.items()):
        print(f"Analyzing file {i+1}/{len(file_contents)}: {file_path}")
        
        try:
            # Analyze the file
            analysis = llm_processor.analyze_file(file_path, content)
            
            # Save the result
            output_file = os.path.join(output_dir, f"analysis_{i+1}.json")
            with open(output_file, 'w') as f:
                json.dump(analysis, f, indent=2)
                
            print(f"Analysis saved to {output_file}")
            
        except Exception as e:
            print(f"Error analyzing file: {e}")
    
    print("Testing completed")
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

## Running the Test

1. Save the test script as `test_llm_processor.py` in your project root.

2. Run the test with a small project or a subset of files:

```bash
python test_llm_processor.py /path/to/small/project
```

3. Check the generated analysis files in the `test_output` directory.

## Expected Output

After running the test, you should see analysis files in the `test_output` directory. Each analysis file should contain a structured JSON object with:

- `summary`: A brief description of the file's purpose
- `components`: Key functions, classes, or structures defined in the file
- `dependencies`: External libraries or internal imports the file relies on
- `functionality`: The main functionality this file provides to the project
- `relationships`: How this file might relate to other parts of the project
- `symbols`: Important symbols defined in this file

## Testing Incremental Analysis

To test incremental analysis (analyzing files while maintaining context between them), modify the test script to analyze all files in the project sequentially:

```python
# Process all files in sequence
all_analysis = {}
for i, (file_path, content) in enumerate(file_contents.items()):
    print(f"Analyzing file {i+1}/{len(file_contents)}: {file_path}")
    
    try:
        # Analyze the file (context builds up between files)
        analysis = llm_processor.analyze_file(file_path, content)
        all_analysis[file_path] = analysis
        
        # Save the individual result
        output_file = os.path.join(output_dir, f"analysis_{i+1}.json")
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
            
        # Save the current project context
        context_file = os.path.join(output_dir, f"context_{i+1}.json")
        with open(context_file, 'w') as f:
            json.dump(llm_processor.project_context, f, indent=2)
            
        print(f"Analysis saved to {output_file}")
        
    except Exception as e:
        print(f"Error analyzing file: {e}")
```

This modified test will also save the evolving project context after each file analysis, allowing you to see how the LLM builds up its understanding of the project over time.

## Troubleshooting

If you encounter issues during testing:

1. **Model loading problems**: Ensure the correct model path is specified and the model is available.

2. **Parsing errors**: Check the LLM responses and adjust the parsing logic in `parse_analysis_response()` if needed.

3. **Context issues**: Review how the project context evolves and adjust the context maintenance logic if necessary.

4. **Large files**: Verify that very large files are handled appropriately (skipped or chunked).

5. **Memory usage**: For large projects, monitor memory usage when analyzing many files sequentially.

## Next Steps

After successfully testing the LLM Processing Service, you can:

1. Integrate it with the rest of your project analyzer components
2. Test with larger projects to verify scalability
3. Fine-tune the prompts to improve analysis quality
4. Implement more sophisticated context management if needed

### Testing
- python test_project_analyzer.py /path/to/small/project