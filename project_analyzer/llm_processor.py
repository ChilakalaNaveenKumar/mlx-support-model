import os
import json
import logging
from typing import Dict, List, Any, Tuple

# Import your existing model service
from mlx_support_model.services.model_service import ModelService

logger = logging.getLogger(__name__)

class LLMProcessingService:
    """
    Service for processing files with the LLM model.
    Handles prompt formatting, context management, and response parsing.
    """
    
    def __init__(self, model_service: ModelService):
        """
        Initialize the LLM processing service.
        
        Args:
            model_service: ModelService instance to use for analysis
        """
        self.model_service = model_service
        self.project_context = {
            "analyzed_files": 0,
            "summary": "Project analysis has just started.",
            "key_components": [],
            "technologies": []
        }
        
        logger.info("Initialized LLM processing service")
    
    def initialize_project_context(self, project_path: str):
        """
        Initialize the project context with basic information.
        
        Args:
            project_path: Path to the project directory
        """
        project_name = os.path.basename(project_path)
        
        # Reset project context
        self.project_context = {
            "project_name": project_name,
            "project_path": project_path,
            "analyzed_files": 0,
            "summary": f"Starting analysis of {project_name} project.",
            "key_components": [],
            "technologies": [],
            "file_summaries": {}
        }
        
        logger.info(f"Initialized context for project: {project_name}")
    
    def get_initial_project_prompt(self, project_path: str) -> str:
        """
        Generate the initial project understanding prompt.
        
        Args:
            project_path: Path to the project directory
            
        Returns:
            Formatted prompt
        """
        project_name = os.path.basename(project_path)
        
        prompt = f"""You are analyzing a software project located at {project_path}.
            The project name appears to be '{project_name}'.
            Your task is to analyze each file to understand its purpose, components, and relationships.
            As you analyze files, build a comprehensive understanding of the project structure.
            Maintain this context throughout the analysis process.

            Your analysis will be used for code navigation, documentation, and understanding.
            Provide structured, accurate information about each file.
        """
        
        return prompt
    
    def get_file_analysis_prompt(self, file_path: str, content: str, file_type: str = None) -> str:
        """
        Generate a prompt for file analysis.
        
        Args:
            file_path: Path to the file
            content: File content
            file_type: Optional file type
            
        Returns:
            Formatted prompt
        """
        # Determine file type from extension if not provided
        if file_type is None:
            _, ext = os.path.splitext(file_path)
            file_type = ext.lstrip('.') or "text"
        
        file_name = os.path.basename(file_path)
        context_info = self._get_context_summary()
        
        prompt = f"""Analyze the following file:

            File Path: {file_path}
            File Name: {file_name}
            File Type: {file_type}

            Project Context:
            {context_info}

            File Content:
            ```{file_type}
            {content}
            Provide a structured analysis with the following information:

            Summary: A brief description of the file's purpose
            Components: Key functions, classes, or structures defined in this file
            Dependencies: External libraries or internal imports this file relies on
            Functionality: The main functionality this file provides to the project
            Relationships: How this file might relate to other parts of the project

            Format your response as JSON with the following structure:
            {{
            "summary": "Brief description of the file's purpose",
            "components": ["component1", "component2", ...],
            "dependencies": ["dependency1", "dependency2", ...],
            "functionality": "Description of main functionality",
            "relationships": ["description of relationship to other files", ...],
            "symbols": ["important symbols defined in this file", ...]
            }}
        """
        
        return prompt

    def get_relationship_prompt(self, file_path: str, analysis: Dict[str, Any]) -> str:
        """
        Generate a prompt for identifying relationships.
        
        Args:
            file_path: Path to the file
            analysis: Analysis of the file
            
        Returns:
            Formatted prompt
        """
        file_name = os.path.basename(file_path)
        summary = analysis.get("summary", "No summary available")
        components = analysis.get("components", [])
        components_str = ", ".join(components) if components else "None identified"
        
        # Get a list of previously analyzed files
        analyzed_files = list(self.project_context.get("file_summaries", {}).keys())
        analyzed_files_str = "\n".join([f"- {f}" for f in analyzed_files[:10]])
        if len(analyzed_files) > 10:
            analyzed_files_str += f"\n- ... and {len(analyzed_files) - 10} more files"
        
        prompt = f"""Based on your analysis of file "{file_name}", identify relationships with other project files.
            File Summary: {summary}
            Components: {components_str}
            Some previously analyzed files in this project (up to 10 shown):
            {analyzed_files_str}
            For each relationship you identify, specify:

            Source File: {file_path}
            Target File: Path to the related file
            Relationship Type: (imports, inherits, references, etc.)
            Description: Brief description of the relationship

            Format your response as a JSON list of relationship objects:
            {{
            "relationships": [
            {{
            "source": "{file_path}",
            "target": "path/to/related/file",
            "type": "relationship type",
            "description": "description of relationship"
            }},
            ...
            ]
            }}
        """
        
        return prompt

    def _get_context_summary(self) -> str:
        """
        Get a summary of the current project context.
        
        Returns:
            Context summary string
        """
        num_files = self.project_context.get("analyzed_files", 0)
        summary = self.project_context.get("summary", "No summary available")
        
        key_components = self.project_context.get("key_components", [])
        key_components_str = ", ".join(key_components[:5])
        if len(key_components) > 5:
            key_components_str += f", and {len(key_components) - 5} more"
            
        technologies = self.project_context.get("technologies", [])
        technologies_str = ", ".join(technologies)
        
        context = f"""So far, {num_files} files have been analyzed in this project.
            Project Summary: {summary}
            Key Components: {key_components_str if key_components else "None identified yet"}
            Technologies: {technologies_str if technologies else "None identified yet"}
        """
        
        return context

    def update_project_context(self, file_path: str, analysis: Dict[str, Any]):
        """
        Update the project context with new file analysis.
        
        Args:
            file_path: Path to the analyzed file
            analysis: Analysis results
        """
        # Increment analyzed files count
        self.project_context["analyzed_files"] = self.project_context.get("analyzed_files", 0) + 1
        
        # Add file summary to context
        file_summaries = self.project_context.get("file_summaries", {})
        file_summaries[file_path] = analysis.get("summary", "No summary available")
        self.project_context["file_summaries"] = file_summaries
        
        # Extract technologies from dependencies
        dependencies = analysis.get("dependencies", [])
        current_technologies = set(self.project_context.get("technologies", []))
        
        # Common technology keywords to look for
        tech_keywords = [
            "python", "javascript", "typescript", "react", "node", "django", "flask",
            "angular", "vue", "express", "spring", "hibernate", "maven", "gradle",
            "docker", "kubernetes", "aws", "azure", "gcp", "tensorflow", "pytorch",
            "pandas", "numpy", "scikit", "fastapi", "sqlalchemy", "mongodb", "postgresql"
        ]
        
        for dep in dependencies:
            for keyword in tech_keywords:
                if keyword.lower() in dep.lower():
                    current_technologies.add(keyword)
        
        self.project_context["technologies"] = list(current_technologies)
        
        # Extract key components
        components = analysis.get("components", [])
        if components:
            current_components = set(self.project_context.get("key_components", []))
            # Add up to 3 new components from this file
            for comp in components[:3]:
                current_components.add(comp)
            self.project_context["key_components"] = list(current_components)
        
        # Update project summary periodically
        if self.project_context["analyzed_files"] % 10 == 0:
            self._update_project_summary()
            
        logger.debug(f"Updated project context with analysis of {file_path}")

    def _update_project_summary(self):
        """
        Update the overall project summary based on accumulated analysis.
        Called periodically to refine the project understanding.
        """
        # For now, we'll use a simple approach
        # In a full implementation, you might want to use the LLM to generate a summary
        technologies = self.project_context.get("technologies", [])
        num_files = self.project_context.get("analyzed_files", 0)
        
        if technologies:
            tech_str = ", ".join(technologies)
            summary = f"This appears to be a project using {tech_str} with {num_files} files analyzed so far."
        else:
            summary = f"Project analysis in progress with {num_files} files analyzed so far."
            
        self.project_context["summary"] = summary

    def parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM's response into a structured analysis.
        
        Args:
            response: LLM response text
            
        Returns:
            Structured analysis as dictionary
        """
        try:
            # Try to find and parse JSON in the response
            import re
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
            
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            
            # If no JSON block with markers, try to parse the whole response
            return json.loads(response)
            
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"Failed to parse JSON from response: {e}")
            
            # Fallback: Create a structured response from text
            fallback_analysis = {
                "summary": "Failed to parse structured analysis from LLM response.",
                "components": [],
                "dependencies": [],
                "functionality": "Unknown",
                "relationships": [],
                "symbols": []
            }
            
            # Try to extract sections using patterns
            sections = {
                "summary": r"(?:Summary|1\.)[^\n]*(.+?)(?=Components|2\.|\Z)",
                "components": r"(?:Components|2\.)[^\n]*(.+?)(?=Dependencies|3\.|\Z)",
                "dependencies": r"(?:Dependencies|3\.)[^\n]*(.+?)(?=Functionality|4\.|\Z)",
                "functionality": r"(?:Functionality|4\.)[^\n]*(.+?)(?=Relationships|5\.|\Z)",
                "relationships": r"(?:Relationships|5\.)[^\n]*(.+?)(?=\Z)"
            }
            
            for key, pattern in sections.items():
                match = re.search(pattern, response, re.DOTALL)
                if match:
                    content = match.group(1).strip()
                    if key in ["components", "dependencies", "relationships", "symbols"]:
                        # Convert list-like text to actual list
                        items = re.findall(r'[-*•]?\s*(.+?)(?=[-*•]|\Z)', content, re.DOTALL)
                        fallback_analysis[key] = [item.strip() for item in items if item.strip()]
                    else:
                        fallback_analysis[key] = content
            
            return fallback_analysis

    def parse_relationships_response(self, response: str) -> List[Dict[str, str]]:
        """
        Parse the LLM's response about relationships.
        
        Args:
            response: LLM response text
            
        Returns:
            List of relationship dictionaries
        """
        try:
            # Try to parse JSON
            import re
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
            
            if json_match:
                json_str = json_match.group(1)
                data = json.loads(json_str)
                return data.get("relationships", [])
            
            # If no JSON block, try to parse the whole response
            data = json.loads(response)
            return data.get("relationships", [])
            
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"Failed to parse relationships from response: {e}")
            return []

    def analyze_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """
        Analyze a file using the LLM.
        
        Args:
            file_path: Path to the file
            content: File content
            
        Returns:
            Structured analysis as dictionary
        """
        
        # Skip very large files or binary content
        file_extension = os.path.splitext(file_path)[1].lower()
        
        # Specific file type handling
        if file_extension == '.svg' and 'base64' in content:
            logger.warning(f"Skipping SVG with embedded base64 content: {file_path}")
            return {
                "summary": "SVG file with embedded base64 data",
                "components": ["SVG elements", "Embedded base64 image"],
                "dependencies": ["SVG standard"],
                "functionality": "Displays a vector graphic with embedded image data",
                "relationships": ["Likely used in UI components or assets"],
                "symbols": []
            }
        
        # Skip very large files or binary content
        if len(content) > 100000 or content.startswith("[BINARY CONTENT:"):
            logger.warning(f"Skipping analysis for large or binary file: {file_path}")
            return {
                "summary": f"Large or binary file ({len(content)} bytes)",
                "components": [],
                "dependencies": [],
                "functionality": "Cannot analyze (file too large or binary)",
                "relationships": [],
                "symbols": []
            }
            
        try:
            # Generate analysis prompt
            prompt = self.get_file_analysis_prompt(file_path, content)
            
            # Get response from LLM
            response = self.model_service.generate_text(prompt, {
                "temperature": 0.3,
                "max_tokens": 2000
            })
            
            # Parse response
            analysis = self.parse_analysis_response(response)
            
            # Add file metadata
            analysis["file_path"] = file_path
            analysis["file_name"] = os.path.basename(file_path)
            
            # Update project context with this analysis
            self.update_project_context(file_path, analysis)
            
            # If we have some files analyzed already, try to identify relationships
            if self.project_context.get("analyzed_files", 0) > 5:
                relationships = self.identify_relationships(file_path, analysis)
                if relationships:
                    analysis["identified_relationships"] = relationships
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {e}")
            return {
                "summary": f"Error during analysis: {str(e)}",
                "components": [],
                "dependencies": [],
                "functionality": "Analysis failed",
                "relationships": [],
                "symbols": []
            }

    def identify_relationships(self, file_path: str, analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Identify relationships between a file and other project files.
        
        Args:
            file_path: Path to the file
            analysis: Analysis of the file
            
        Returns:
            List of relationship dictionaries
        """
        # Only try to identify relationships if we've analyzed some files
        if self.project_context.get("analyzed_files", 0) < 5:
            return []
        
        try:
            # Generate relationship prompt
            prompt = self.get_relationship_prompt(file_path, analysis)
            
            # Get response from LLM
            response = self.model_service.generate_text(prompt, {
                "temperature": 0.3,
                "max_tokens": 1000
            })
            
            # Parse relationship response
            return self.parse_relationships_response(response)
            
        except Exception as e:
            logger.error(f"Error identifying relationships for {file_path}: {e}")
            return []

    def analyze_files(self, file_contents: Dict[str, str]) -> Dict[str, Any]:
        """
        Analyze multiple files in a project.
        
        Args:
            file_contents: Dictionary of file paths to contents
            
        Returns:
            Dictionary of file paths to analysis results
        """
        # Initialize project context
        project_path = os.path.dirname(next(iter(file_contents))) if file_contents else ""
        self.initialize_project_context(project_path)
        
        # Initialize with project understanding prompt
        initial_prompt = self.get_initial_project_prompt(project_path)
        self.model_service.generate_text(initial_prompt, {
            "temperature": 0.3,
            "max_tokens": 100
        })
        
        # Process each file
        analysis_results = {}
        total_files = len(file_contents)
        
        for i, (file_path, content) in enumerate(file_contents.items()):
            logger.info(f"Analyzing file {i+1}/{total_files}: {file_path}")
            analysis = self.analyze_file(file_path, content)
            analysis_results[file_path] = analysis
        
        logger.info(f"Completed analysis of {len(analysis_results)} files")
        return analysis_results