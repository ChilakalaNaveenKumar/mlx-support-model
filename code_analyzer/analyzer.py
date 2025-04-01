import os
import logging
from typing import Dict, List, Any, Tuple

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
        
        # Create analyses directory for individual file analyses
        analyses_dir = os.path.join(output_dir, "analyses")
        os.makedirs(analyses_dir, exist_ok=True)
        
        # Import these locally to avoid circular imports
        from code_analyzer.scanner import ProjectScanner
        from code_analyzer.processor import LLMProcessor
        from code_analyzer.generator import MindMapGenerator
        
        # Initialize components
        scanner = ProjectScanner(project_path)
        processor = LLMProcessor(self.model_service)
        generator = MindMapGenerator()
        
        # Scan the project
        logger.info(f"Scanning project: {project_path}")
        project_files = scanner.scan_project()
        
        # Process each file
        logger.info("Analyzing files...")
        all_analyses = []
        for i, (file_path, content) in enumerate(project_files):
            logger.info(f"Analyzing file {i+1}/{len(project_files)}: {os.path.basename(file_path)}")
            
            # Analyze file
            analysis = processor.analyze_file(file_path, content)
            
            # Create a formatted analysis text
            analysis_text = self._format_analysis_for_file(file_path, analysis, content)
            
            # Save individual analysis
            rel_path = os.path.relpath(file_path, project_path)
            safe_filename = rel_path.replace(os.sep, '_').replace('.', '_')
            file_analysis_path = os.path.join(analyses_dir, f"{safe_filename}_analysis.txt")
            
            try:
                with open(file_analysis_path, 'w', encoding='utf-8') as f:
                    f.write(analysis_text)
                    
                all_analyses.append(analysis_text)
                
                logger.info(f"Saved analysis to {file_analysis_path}")
            except Exception as e:
                logger.error(f"Error saving analysis to {file_analysis_path}: {e}")
        
        # Create consolidated analysis file
        consolidated_path = os.path.join(output_dir, "consolidated_analysis.txt")
        try:
            with open(consolidated_path, 'w', encoding='utf-8') as f:
                f.write(f"# Consolidated Analysis for {project_path}\n\n")
                f.write(f"Total files analyzed: {len(project_files)}\n\n")
                
                # Add separator between file analyses
                separator = "\n" + "="*80 + "\n\n"
                f.write(separator.join(all_analyses))
                
            logger.info(f"Saved consolidated analysis to {consolidated_path}")
        except Exception as e:
            logger.error(f"Error saving consolidated analysis: {e}")
            consolidated_path = ""
        
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
        
        return consolidated_path
        
    def _format_analysis_for_file(self, file_path: str, analysis: Dict[str, Any], content: str) -> str:
        """Format the analysis for a single file into a readable text format."""
        # Count lines in the file
        line_count = content.count('\n') + 1
        
        # Create a formatted header
        header = f"File: {file_path}\n"
        header += f"{'=' * len(file_path)}\n\n"
        
        # Add component name and description
        body = f"Component: {analysis.get('name', os.path.basename(file_path))}\n"
        if analysis.get('description'):
            body += f"Description: {analysis['description']}\n\n"
        else:
            body += "Description: No description available\n\n"
        
        # Add imports with line numbers if available
        body += "Imports:\n"
        if analysis.get('imports') and analysis['imports']:
            for imp in analysis['imports']:
                # Add line numbers if available
                line_info = self._get_line_info(imp, analysis.get('line_ranges', {}))
                body += f"  - {imp}{line_info}\n"
        else:
            body += "  None detected\n"
        body += "\n"
        
        # Add key functionality with line numbers if available
        body += "Key Functionality:\n"
        if isinstance(analysis.get('key_functionality'), list) and analysis['key_functionality']:
            for func in analysis['key_functionality']:
                line_info = self._get_line_info(func, analysis.get('line_ranges', {}))
                body += f"  - {func}{line_info}\n"
        elif analysis.get('key_functionality'):
            body += f"  {analysis['key_functionality']}\n"
        elif 'raw_response' in analysis:
            # Try to extract key functionality from raw response
            import re
            func_match = re.search(r'Key Functionality:(.*?)(?:Components/Functions|$)', 
                                  analysis['raw_response'], re.DOTALL)
            if func_match:
                func_text = func_match.group(1).strip()
                lines = func_text.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and (line.startswith('-') or line.startswith('*')):
                        body += f"  {line}\n"
                    elif line and not line.startswith('#'):
                        body += f"  - {line}\n"
                if not any(line.strip() for line in lines):
                    body += "  No specific functionality detected\n"
            else:
                body += "  No specific functionality detected\n"
        else:
            body += "  No specific functionality detected\n"
        body += "\n"
        
        # Add functions/classes with line numbers if available
        body += "Components/Functions Defined:\n"
        components_added = False
        
        # Add functions
        if analysis.get('functions'):
            for func in analysis['functions']:
                # Extract function name for line range lookup
                func_name = func.split(':', 1)[0].strip() if ':' in func else func
                line_info = self._get_line_info(func_name, analysis.get('line_ranges', {}))
                body += f"  - {func}{line_info}\n"
                components_added = True
                
        # Add classes
        if analysis.get('classes'):
            for cls in analysis['classes']:
                # Extract class name for line range lookup
                cls_name = cls.split(':', 1)[0].strip() if ':' in cls else cls
                line_info = self._get_line_info(cls_name, analysis.get('line_ranges', {}))
                body += f"  - {cls}{line_info}\n"
                components_added = True
                
        # If no components were added, check raw response
        if not components_added and 'raw_response' in analysis:
            # Try to extract from raw response
            import re
            comp_match = re.search(r'Components/Functions Defined:(.*?)(?:$|```)', 
                                analysis['raw_response'], re.DOTALL)
            if comp_match:
                comp_text = comp_match.group(1).strip()
                if "None" not in comp_text and not comp_text.lower().startswith('none'):
                    lines = comp_text.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and (line.startswith('-') or line.startswith('*')):
                            body += f"  {line}\n"
                            components_added = True
                        elif line and not line.startswith('#'):
                            body += f"  - {line}\n"
                            components_added = True
            
        if not components_added:
            body += "  No specific components defined\n"
        body += "\n"
        
        # Add file metadata
        footer = f"File size: {len(content)} bytes\n"
        footer += f"Line count: {line_count} lines\n"
        
        return header + body + footer
    
    def _get_line_info(self, item: str, line_ranges: Dict[str, Tuple[int, int]]) -> str:
        """Generate line number information string if available."""
        if item in line_ranges:
            start_line, end_line = line_ranges[item]
            if start_line == end_line:
                return f" (line {start_line})"
            else:
                return f" (lines {start_line}-{end_line})"
        return ""
    
    def analyze_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """
        Analyze a single file.
        
        Args:
            file_path: Path to the file
            content: File content
            
        Returns:
            Analysis results
        """
        # Import LLMProcessor locally to avoid circular imports
        from code_analyzer.processor import LLMProcessor
        
        # Initialize processor
        processor = LLMProcessor(self.model_service)
        
        # Process file
        logger.info(f"Analyzing file: {os.path.basename(file_path)}")
        analysis = processor.analyze_file(file_path, content)
        
        return analysis
    
    def generate_incremental_mind_map(self, analysis_results: Dict[str, Dict[str, Any]]) -> str:
        """
        Generate a mind map from incremental analysis results.
        
        Args:
            analysis_results: Dictionary of file paths to analysis results
            
        Returns:
            Mind map text
        """
        # Import services locally to avoid circular imports
        from code_analyzer.processor import LLMProcessor
        from code_analyzer.generator import MindMapGenerator
        
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