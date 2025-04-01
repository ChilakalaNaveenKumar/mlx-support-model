import os
import logging
from typing import Dict, Any

from code_analyzer.scanner import ProjectScanner
from code_analyzer.processor import LLMProcessor
from code_analyzer.generator import MindMapGenerator

logger = logging.getLogger("code_analyzer")

class CodeAnalyzer:
    """
    Main orchestrator for code analysis and mind map generation.
    """
    
    def __init__(self, model_service):
        """Initialize with an MLX model service."""
        self.model_service = model_service
        logger.info("Initialized code analyzer")
    
    def analyze_project(self, project_path: str, output_dir: str = None) -> str:
        """
        Analyze a project and generate a mind map.
        
        Args:
            project_path: Path to the project directory
            output_dir: Directory to save output files (default: current directory)
            
        Returns:
            Path to the generated mind map file
        """
        # Create output directory if it doesn't exist
        if output_dir is None:
            output_dir = os.getcwd()
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        scanner = ProjectScanner(project_path)
        processor = LLMProcessor(self.model_service)
        generator = MindMapGenerator()
        
        # Scan the project
        logger.info(f"Scanning project: {project_path}")
        project_files = scanner.scan_project()
        
        # Process each file
        logger.info("Analyzing files...")
        for i, (file_path, content) in enumerate(project_files):
            logger.info(f"Analyzing file {i+1}/{len(project_files)}: {os.path.basename(file_path)}")
            processor.analyze_file(file_path, content)
        
        # Generate mind map
        logger.info("Generating mind map...")
        mind_map_data = processor.generate_mind_map()
        mind_map_text = generator.generate_text_mind_map(mind_map_data)
        
        # Save text mind map
        project_name = os.path.basename(os.path.normpath(project_path))
        output_path = os.path.join(output_dir, f"{project_name}_mind_map.txt")
        generator.save_mind_map(mind_map_text, output_path)
        
        # Generate and save HTML mind map
        html_output_path = os.path.join(output_dir, f"{project_name}_mind_map.html")
        generator.visualize_html_mind_map(mind_map_data, html_output_path)
        
        return output_path
    
    def analyze_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """
        Analyze a single file.
        
        Args:
            file_path: Path to the file
            content: File content
            
        Returns:
            Analysis results
        """
        # Initialize processor
        processor = LLMProcessor(self.model_service)
        
        # Process file
        logger.info(f"Analyzing file: {os.path.basename(file_path)}")
        analysis = processor.analyze_file(file_path, content)
        
        return analysis
    
    def generate_incremental_mind_map(self, analysis_results: Dict[str, Dict[str, Any]]):
        """
        Generate a mind map from incremental analysis results.
        
        Args:
            analysis_results: Dictionary of file paths to analysis results
            
        Returns:
            Mind map text
        """
        # Initialize processor and generator
        processor = LLMProcessor(self.model_service)
        generator = MindMapGenerator()
        
        # Add analysis results to processor context
        for file_path, analysis in analysis_results.items():
            processor.project_context["components"][file_path] = analysis
            processor._update_project_context(file_path, analysis)
        
        # Generate mind map
        mind_map_data = processor.generate_mind_map()
        mind_map_text = generator.generate_text_mind_map(mind_map_data)
        
        return mind_map_text