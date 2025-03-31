#!/usr/bin/env python3
"""
Test script for the ProjectReaderService.
Tests reading files from a project directory.
"""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_project_reader")

from project_analyzer.project_reader import ProjectReaderService

def main():
    """Main test function."""
    # Check command line args
    if len(sys.argv) < 2:
        print("Usage: python test_project_reader.py <path_to_test_project>")
        return 1
        
    test_project_path = sys.argv[1]
    
    if not os.path.isdir(test_project_path):
        print(f"Error: {test_project_path} is not a valid directory")
        return 1
    
    # Initialize project reader
    reader = ProjectReaderService(test_project_path)
    
    # Read all files
    print(f"Reading files from {test_project_path}")
    files = reader.read_project()
    
    print(f"Found {len(files)} files:")
    
    # Group files by directory
    dirs = {}
    for file_path, _ in files:
        dir_name = os.path.dirname(file_path)
        if dir_name not in dirs:
            dirs[dir_name] = []
        dirs[dir_name].append(os.path.basename(file_path))
    
    # Print files by directory
    for dir_name, file_list in dirs.items():
        print(f"\nDirectory: {dir_name}")
        for file_name in file_list:
            print(f"  - {file_name}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())