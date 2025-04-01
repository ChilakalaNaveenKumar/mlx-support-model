import os
import logging
from typing import Dict, List, Any, Tuple

logger = logging.getLogger("code_analyzer")

class LLMProcessor:
    """
    Processes code files using a local MLX model to extract relationships.
    """
    
    def __init__(self, model_service):
        """Initialize with an MLX model service."""
        self.model_service = model_service
        self.project_context = {
            "analyzed_files": 0,
            "summary": "Project analysis has just started.",
            "components": {},  # Will store component info by path
            "relationships": []  # Will store relationships between components
        }
        logger.info("Initialized LLM processor")
    
    def analyze_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """
        Analyze a file using the LLM.
        
        Args:
            file_path: Path to the file
            content: File content
            
        Returns:
            Dictionary with analysis results
        """
        # Skip very large or binary files
        if len(content) > 100000 or content.startswith("[BINARY CONTENT:"):
            logger.warning(f"Skipping analysis for large or binary file: {file_path}")
            return self._create_placeholder_analysis(file_path)
        
        try:
            # Generate a prompt for the LLM
            prompt = self._create_analysis_prompt(file_path, content)
            
            # Get analysis from the model
            response = self.model_service.generate_text(prompt, {
                "temperature": 0.3,
                "max_tokens": 2000
            })
            
            # Parse the response
            analysis = self._parse_llm_response(response, file_path)
            
            # Update project context
            self._update_project_context(file_path, analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {e}")
            return self._create_placeholder_analysis(file_path)
    
    def _create_analysis_prompt(self, file_path: str, content: str) -> str:
        """Create a prompt for the LLM to analyze the file."""
        # Get file extension and name
        file_name = os.path.basename(file_path)
        _, file_ext = os.path.splitext(file_name)
        
        # Add project context summary
        context_summary = self._get_context_summary()
        
        prompt = f"""Analyze this code file and identify its components, imports, and relationships:

File path: {file_path}
File name: {file_name}

Project context so far:
{context_summary}

File content:
```{file_ext}
{content}
```

Provide the following information in your analysis:
1. Component name (class, module, or file name)
2. Brief description of the component's purpose
3. Imports or dependencies on other components (by name and path if possible)
4. Key functionality provided by this component
5. Components or functions defined within this file

Format your response as a simple text tree showing the component and its relationships.
"""
        return prompt
    
    def _parse_llm_response(self, response: str, file_path: str) -> Dict[str, Any]:
        """Parse the LLM's response into a structured analysis."""
        # This is a simplified parser - a real implementation would be more robust
        lines = response.strip().split('\n')
        
        analysis = {
            "file_path": file_path,
            "name": os.path.basename(file_path),
            "description": "",
            "imports": [],
            "functions": [],
            "classes": [],
            "key_functionality": ""
        }
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for section headers
            if "Component name:" in line or "# Component:" in line:
                analysis["name"] = line.split(":", 1)[1].strip()
            elif "Description:" in line or "# Description:" in line:
                analysis["description"] = line.split(":", 1)[1].strip()
                current_section = "description"
            elif "Imports:" in line or "# Imports:" in line:
                current_section = "imports"
            elif "Key functionality:" in line or "# Functionality:" in line:
                current_section = "key_functionality"
            elif "Functions:" in line or "# Functions:" in line:
                current_section = "functions"
            elif "Classes:" in line or "# Classes:" in line:
                current_section = "classes"
            elif line.startswith('-') and current_section in ["imports", "functions", "classes"]:
                item = line.lstrip('- ').strip()
                if item and item not in analysis[current_section]:
                    analysis[current_section].append(item)
            elif current_section in ["description", "key_functionality"]:
                if current_section == "description" and not analysis["description"]:
                    analysis["description"] = line
                elif current_section == "key_functionality":
                    if not analysis["key_functionality"]:
                        analysis["key_functionality"] = line
                    else:
                        analysis["key_functionality"] += " " + line
        
        return analysis
    
    def _update_project_context(self, file_path: str, analysis: Dict[str, Any]):
        """Update the project context with new analysis results."""
        self.project_context["analyzed_files"] += 1
        self.project_context["components"][file_path] = analysis
        
        # Extract relationships
        for imported_item in analysis.get("imports", []):
            # Try to identify the import path
            import_path = self._resolve_import_path(file_path, imported_item)
            if import_path:
                relationship = {
                    "source": file_path,
                    "target": import_path,
                    "type": "imports"
                }
                if relationship not in self.project_context["relationships"]:
                    self.project_context["relationships"].append(relationship)
        
        # Update project summary periodically
        if self.project_context["analyzed_files"] % 10 == 0:
            self._update_project_summary()
    
    def _resolve_import_path(self, source_file: str, import_statement: str) -> str:
        """Attempt to resolve an import statement to a file path."""
        # This is a simplified implementation - real one would be more robust
        # For now, just look for components we've already analyzed
        import_name = import_statement.split()[0] if " " in import_statement else import_statement
        
        # Check if this import name matches a component we already know
        for path, comp in self.project_context["components"].items():
            if comp["name"] == import_name:
                return path
                
        # We couldn't resolve it yet - might be analyzed later
        return ""
    
    def _get_context_summary(self) -> str:
        """Get a summary of the current project context."""
        num_files = self.project_context["analyzed_files"]
        summary = self.project_context["summary"]
        
        # List some key components we've found
        components_list = list(self.project_context["components"].values())
        component_names = [c["name"] for c in components_list[:5]]
        comp_str = ", ".join(component_names)
        if len(components_list) > 5:
            comp_str += f", and {len(components_list) - 5} more"
        
        return f"""Analysis progress: {num_files} files analyzed.
Summary: {summary}
Key components identified: {comp_str if component_names else "None yet"}"""
    
    def _update_project_summary(self):
        """Update the overall project summary based on accumulated analysis."""
        num_files = self.project_context["analyzed_files"]
        num_components = len(self.project_context["components"])
        self.project_context["summary"] = f"Project analysis in progress. Found {num_components} components in {num_files} files."
    
    def _create_placeholder_analysis(self, file_path: str) -> Dict[str, Any]:
        """Create a placeholder analysis for files that couldn't be analyzed."""
        return {
            "file_path": file_path,
            "name": os.path.basename(file_path),
            "description": "Could not analyze this file (too large or binary)",
            "imports": [],
            "functions": [],
            "classes": [],
            "key_functionality": "Unknown (file not analyzed)"
        }
    
    def generate_mind_map(self) -> Dict[str, Any]:
        """Generate a mind map structure from the analysis results."""
        # Create a hierarchical tree based on file paths and relationships
        root = {
            "name": os.path.basename(os.path.dirname(list(self.project_context["components"].keys())[0]) 
                    if self.project_context["components"] else "Project"),
            "children": []
        }
        
        # Build a directory tree structure
        dir_structure = {}
        
        for file_path in self.project_context["components"]:
            parts = file_path.split(os.sep)
            current = dir_structure
            
            # Build the tree structure
            for i, part in enumerate(parts):
                if i == len(parts) - 1:  # This is a file
                    if "files" not in current:
                        current["files"] = []
                    current["files"].append(file_path)
                else:  # This is a directory
                    if part not in current:
                        current[part] = {}
                    current = current[part]
        
        # Now convert the directory structure to a mind map
        root["children"] = self._convert_dir_structure_to_mind_map(dir_structure, "")
        
        return root
    
    def _convert_dir_structure_to_mind_map(self, dir_structure, path_so_far):
        """Convert a directory structure to a mind map structure."""
        result = []
        
        # Add directories
        for dirname, contents in dir_structure.items():
            if dirname == "files":
                continue
                
            new_path = os.path.join(path_so_far, dirname)
            dir_node = {
                "name": dirname,
                "path": new_path,
                "children": self._convert_dir_structure_to_mind_map(contents, new_path)
            }
            result.append(dir_node)
        
        # Add files
        if "files" in dir_structure:
            for file_path in dir_structure["files"]:
                component = self.project_context["components"].get(file_path, {})
                
                # Get relationships for this file
                children = []
                for rel in self.project_context["relationships"]:
                    if rel["source"] == file_path and rel["target"]:
                        target_comp = self.project_context["components"].get(rel["target"], {})
                        children.append({
                            "name": target_comp.get("name", os.path.basename(rel["target"])),
                            "path": rel["target"],
                            "type": rel["type"],
                            "description": target_comp.get("description", "")
                        })
                
                file_node = {
                    "name": component.get("name", os.path.basename(file_path)),
                    "path": file_path,
                    "description": component.get("description", ""),
                    "functionality": component.get("key_functionality", "")
                }
                
                if children:
                    file_node["children"] = children
                    
                result.append(file_node)
                
        return result