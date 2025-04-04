"""
File to Q&A System - A tool for generating comprehensive Q&A pairs from files.
"""

from file_to_qa_system.qa_generator import generate_qa_from_file, create_finetune_script
from file_to_qa_system.file_processor import (
    read_file, save_qa_pairs, get_file_extension, 
    find_files, get_file_stat, is_text_file
)

__version__ = "0.1.0"