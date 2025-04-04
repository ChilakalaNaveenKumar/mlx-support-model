#!/usr/bin/env python3
"""
Main script for the File to Q&A System.
Processes files and generates comprehensive Q&A pairs for model fine-tuning.
"""

import os
import sys
import json
import argparse
import logging
from typing import List, Dict, Any, Optional

from file_to_qa_system.qa_generator import generate_qa_from_file, create_finetune_script
from file_to_qa_system.file_processor import read_file, save_qa_pairs, get_file_extension, find_files, is_text_file

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("file_to_qa")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive Q&A pairs from files for model fine-tuning"
    )
    
    # Input sources
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--file", "-f",
        help="Path to a single file to process"
    )
    input_group.add_argument(
        "--directory", "-d",
        help="Path to a directory of files to process"
    )
    
    # Output options
    parser.add_argument(
        "--output", "-o",
        default="qa_output",
        help="Output directory for Q&A pairs (default: qa_output)"
    )
    parser.add_argument(
        "--output-format", "-of",
        choices=["jsonl", "json", "csv"],
        default="jsonl",
        help="Output format for Q&A pairs (default: jsonl)"
    )
    
    # Generation parameters
    parser.add_argument(
        "--min-pairs", "-m",
        type=int,
        default=50,
        help="Minimum number of Q&A pairs to generate per file (default: 50)"
    )
    parser.add_argument(
        "--priority-keywords", "-pk",
        nargs="+",
        help="Keywords to prioritize in Q&A generation (e.g., file names, key terms)"
    )
    parser.add_argument(
        "--llm-model", "-lm",
        default="default",
        help="Local LLM model to use (default: use the system's default model)"
    )
    
    # Filtering options
    parser.add_argument(
        "--extensions", "-e",
        nargs="+",
        default=[".py", ".js", ".html", ".css", ".md", ".txt", ".json"],
        help="File extensions to process (default: common code and text files)"
    )
    parser.add_argument(
        "--max-files", "-mf",
        type=int,
        default=0,
        help="Maximum number of files to process (0 = no limit, default: 0)"
    )
    
    # Fine-tuning options
    parser.add_argument(
        "--create-finetune", "-cf",
        action="store_true",
        help="Create a fine-tuning script after generating Q&A pairs"
    )
    parser.add_argument(
        "--train-split", "-ts",
        type=float,
        default=0.8,
        help="Train/test split ratio (default: 0.8)"
    )
    
    # Other options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def process_single_file(file_path: str, args):
    """
    Process a single file to generate Q&A pairs.
    
    Args:
        file_path: Path to the file
        args: Command line arguments
        
    Returns:
        List of Q&A pairs generated from the file
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return []
        
    if not is_text_file(file_path):
        logger.error(f"Not a text file or unsupported format: {file_path}")
        return []
    
    # Read file content
    file_content = read_file(file_path)
    if not file_content:
        logger.error(f"Failed to read file or file is empty: {file_path}")
        return []
    
    # Get file extension for format detection
    file_extension = get_file_extension(file_path)
    
    # Extract filename and directory for priority keywords
    file_name = os.path.basename(file_path)
    base_name = os.path.splitext(file_name)[0]
    
    # Combine user-provided keywords with filename-based keywords
    priority_keywords = args.priority_keywords or []
    if file_name not in priority_keywords:
        priority_keywords.append(file_name)
    if base_name not in priority_keywords:
        priority_keywords.append(base_name)
    
    # Generate Q&A pairs
    logger.info(f"Generating Q&A pairs from file: {file_path}")
    qa_pairs = generate_qa_from_file(
        file_content=file_content,
        file_path=file_path,
        file_extension=file_extension,
        min_pairs=args.min_pairs,
        priority_keywords=priority_keywords,
        model_name=args.llm_model if args.llm_model != "default" else None
    )
    
    # Save Q&A pairs
    if qa_pairs:
        output_dir = args.output
        os.makedirs(output_dir, exist_ok=True)
        
        # Create file-specific output path
        output_filename = f"{base_name}_qa.{args.output_format}"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save to file
        save_qa_pairs(qa_pairs, output_path, args.output_format)
        logger.info(f"Saved {len(qa_pairs)} Q&A pairs to {output_path}")
    
    return qa_pairs


def process_directory(directory_path: str, args):
    """
    Process all files in a directory to generate Q&A pairs.
    
    Args:
        directory_path: Path to the directory
        args: Command line arguments
        
    Returns:
        Dictionary mapping file paths to their Q&A pairs
    """
    if not os.path.isdir(directory_path):
        logger.error(f"Directory not found: {directory_path}")
        return {}
    
    # Find all matching files in the directory
    extensions = args.extensions
    file_paths = find_files(directory_path, extensions)
    
    if not file_paths:
        logger.error(f"No files with extensions {extensions} found in {directory_path}")
        return {}
    
    # Limit the number of files if specified
    if args.max_files > 0 and len(file_paths) > args.max_files:
        logger.info(f"Limiting to {args.max_files} files (from {len(file_paths)} total)")
        file_paths = file_paths[:args.max_files]
    
    # Process each file
    all_qa_pairs = {}
    for file_path in file_paths:
        qa_pairs = process_single_file(file_path, args)
        if qa_pairs:
            all_qa_pairs[file_path] = qa_pairs
    
    return all_qa_pairs


def create_combined_dataset(all_qa_pairs: Dict[str, List[Dict[str, str]]], args):
    """
    Create combined datasets for fine-tuning.
    
    Args:
        all_qa_pairs: Dictionary mapping file paths to Q&A pairs
        args: Command line arguments
    """
    if not all_qa_pairs:
        logger.error("No Q&A pairs to create combined dataset")
        return
        
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine all Q&A pairs
    combined_pairs = []
    for file_path, qa_pairs in all_qa_pairs.items():
        combined_pairs.extend(qa_pairs)
    
    if not combined_pairs:
        logger.error("No valid Q&A pairs found")
        return
    
    # Shuffle and split into train/test sets
    import random
    random.shuffle(combined_pairs)
    
    split_idx = int(len(combined_pairs) * args.train_split)
    train_pairs = combined_pairs[:split_idx]
    test_pairs = combined_pairs[split_idx:]
    
    # Save combined datasets
    combined_path = os.path.join(output_dir, f"combined.{args.output_format}")
    train_path = os.path.join(output_dir, f"train.{args.output_format}")
    test_path = os.path.join(output_dir, f"test.{args.output_format}")
    
    save_qa_pairs(combined_pairs, combined_path, args.output_format)
    save_qa_pairs(train_pairs, train_path, args.output_format)
    save_qa_pairs(test_pairs, test_path, args.output_format)
    
    logger.info(f"Created combined dataset with {len(combined_pairs)} Q&A pairs")
    logger.info(f"Train set: {len(train_pairs)} pairs")
    logger.info(f"Test set: {len(test_pairs)} pairs")
    
    # Create fine-tuning script if requested
    if args.create_finetune:
        script_path = os.path.join(output_dir, "finetune.py")
        create_finetune_script(
            train_dataset_path=train_path,
            test_dataset_path=test_path,
            output_script_path=script_path,
            model_name=args.llm_model if args.llm_model != "default" else None
        )
        logger.info(f"Created fine-tuning script: {script_path}")


def main():
    """Main entry point for the script."""
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        all_qa_pairs = {}
        
        # Process input (file or directory)
        if args.file:
            qa_pairs = process_single_file(args.file, args)
            if qa_pairs:
                all_qa_pairs[args.file] = qa_pairs
        
        elif args.directory:
            all_qa_pairs = process_directory(args.directory, args)
        
        # Create combined dataset (even for a single file if create_finetune is requested)
        if len(all_qa_pairs) > 0 and (len(all_qa_pairs) > 1 or args.create_finetune):
            create_combined_dataset(all_qa_pairs, args)
        
        # Print summary
        total_files = len(all_qa_pairs)
        total_pairs = sum(len(pairs) for pairs in all_qa_pairs.values())
        
        print("\n" + "=" * 60)
        print(f"Processing complete!")
        print(f"Files processed: {total_files}")
        print(f"Total Q&A pairs generated: {total_pairs}")
        print(f"Output directory: {os.path.abspath(args.output)}")
        print("=" * 60 + "\n")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())