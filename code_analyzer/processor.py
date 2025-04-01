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
            
            # Print the raw response for debugging
            logger.debug(f"Raw LLM response for {file_path}:\n{response}")
            
            # Parse the response
            analysis = self._parse_llm_response(response, file_path)
            
            # Update project context
            self._update_project_context(file_path, analysis)
            
            # Also store the raw response for reference
            analysis['raw_response'] = response
            
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
        
        # Add line numbers to content for reference
        numbered_content = ""
        lines = content.split('\n')
        for i, line in enumerate(lines):
            numbered_content += f"{i+1:4d} | {line}\n"
        
        prompt = f"""Analyze this code file and identify its components, imports, and relationships:

File path: {file_path}
File name: {file_name}

Project context so far:
{context_summary}

File content (with line numbers):
```{file_ext}
{numbered_content}
```

Provide the following information in your analysis:
1. Component name (class, module, or file name)
2. Brief description of the component's purpose (1-2 sentences)
3. Imports or dependencies:
   - List all imports with their source modules/packages
   - Include line numbers where imports appear
   - List all the knowlegde gaps it may be words, sentences or concepts that need more information to understand
   - Include line numbers where the words or terms are used
4. Key functionality:
   - Describe the main functionality this component provides
   - Reference specific line numbers when relevant (e.g., "Lines 10-15: Implements user authentication")
5. Components/functions defined:
   - List classes, functions, or other components defined in this file
   - Include line number ranges for each (e.g., "Lines 20-35: UserProfile class")

Format your response using the following structure:
```
Component: [name]
Description: [brief description]
Imports:
  - [import1] from [source] (line X)
  - [import2] from [source] (lines X-Y)
Key Functionality:
  - [description of functionality] (lines X-Y)
  - [additional functionality] (lines X-Y)
Components/Functions Defined:
  - [component1] (lines X-Y): [brief description]
  - [component2] (lines X-Y): [brief description]
```
"""
        return prompt
    
    def _parse_llm_response(self, response: str, file_path: str) -> Dict[str, Any]:
        """Parse the LLM's response into a structured analysis."""
        analysis = {
            "file_path": file_path,
            "name": os.path.basename(file_path),
            "description": "",
            "imports": [],
            "functions": [],
            "classes": [],
            "key_functionality": [],
            "line_ranges": {}  # Store line number references
        }
        
        # Extract structured information directly from the response
        import re
        
        # Try to find component name
        component_match = re.search(r'Component:\s*(.*?)(?:\n|$)', response)
        if component_match:
            analysis["name"] = component_match.group(1).strip()
            
        # Try to find description
        description_match = re.search(r'Description:\s*(.*?)(?:\n\n|\nImports:)', response, re.DOTALL)
        if description_match:
            analysis["description"] = description_match.group(1).strip()
            
        # Try to find imports with line numbers
        imports_section_match = re.search(r'Imports:(.*?)(?:Key Functionality:|$)', response, re.DOTALL)
        if imports_section_match:
            imports_section = imports_section_match.group(1).strip()
            
            # If imports section has "None" or no content, keep the default empty list
            if imports_section and not imports_section.lower().startswith('none') and "None detected" not in imports_section:
                # Extract all import lines
                import_lines = imports_section.split('\n')
                for line in import_lines:
                    line = line.strip()
                    if line and (line.startswith('-') or line.startswith('*')):
                        # Extract the import description
                        import_text = line[1:].strip()
                        
                        # Extract line numbers if available
                        line_match = re.search(r'\((lines?\s+(\d+)(?:-(\d+))?)\)', import_text)
                        if line_match:
                            line_ref = line_match.group(1)  # Full line reference text
                            start_line = int(line_match.group(2))
                            end_line = int(line_match.group(3)) if line_match.group(3) else start_line
                            
                            # Remove line number reference from import text
                            import_text = re.sub(r'\s*\(lines?\s+\d+(?:-\d+)?\)', '', import_text)
                            
                            # Store the import and its line range
                            analysis["imports"].append(import_text)
                            analysis["line_ranges"][import_text] = (start_line, end_line)
                        else:
                            analysis["imports"].append(import_text)
        
        # Try to find key functionality with line numbers
        key_func_section_match = re.search(r'Key Functionality:(.*?)(?:Components/Functions|$)', response, re.DOTALL)
        if key_func_section_match:
            key_func_section = key_func_section_match.group(1).strip()
            
            # If key functionality section has "None" or no content, keep the default empty list
            if key_func_section and not key_func_section.lower().startswith('none') and "No specific functionality" not in key_func_section:
                # Extract all functionality lines
                func_lines = key_func_section.split('\n')
                for line in func_lines:
                    line = line.strip()
                    if line and (line.startswith('-') or line.startswith('*')):
                        # Extract the functionality description
                        func_text = line[1:].strip()
                        
                        # Extract line numbers if available
                        line_match = re.search(r'\((lines?\s+(\d+)(?:-(\d+))?)\)', func_text)
                        if line_match:
                            line_ref = line_match.group(1)  # Full line reference text
                            start_line = int(line_match.group(2))
                            end_line = int(line_match.group(3)) if line_match.group(3) else start_line
                            
                            # Remove line number reference from functionality text
                            func_text = re.sub(r'\s*\(lines?\s+\d+(?:-\d+)?\)', '', func_text)
                            
                            # Store the functionality and its line range
                            analysis["key_functionality"].append(func_text)
                            analysis["line_ranges"][func_text] = (start_line, end_line)
                        else:
                            analysis["key_functionality"].append(func_text)
        
        # Try to find components/functions defined with line numbers
        components_section_match = re.search(r'Components/Functions Defined:(.*?)(?:$|```)', response, re.DOTALL)
        if components_section_match:
            components_section = components_section_match.group(1).strip()
            
            # If components section has "None" or no content, keep the default empty lists
            if components_section and not components_section.lower().startswith('none') and "No specific components" not in components_section:
                # Extract all component lines
                component_lines = components_section.split('\n')
                for line in component_lines:
                    line = line.strip()
                    if line and (line.startswith('-') or line.startswith('*')):
                        # Extract the component description
                        comp_text = line[1:].strip()
                        
                        # Extract line numbers if available
                        line_match = re.search(r'\((lines?\s+(\d+)(?:-(\d+))?)\)', comp_text)
                        if line_match:
                            line_ref = line_match.group(1)  # Full line reference text
                            start_line = int(line_match.group(2))
                            end_line = int(line_match.group(3)) if line_match.group(3) else start_line
                            
                            # Remove line number reference from component text
                            comp_text = re.sub(r'\s*\(lines?\s+\d+(?:-\d+)?\)', '', comp_text)
                            
                            # Determine if this is a class or function
                            comp_name = comp_text.split(':', 1)[0].strip() if ':' in comp_text else comp_text
                            
                            # Add to appropriate list
                            if "class" in comp_text.lower():
                                analysis["classes"].append(comp_text)
                                analysis["line_ranges"][comp_name] = (start_line, end_line)
                            else:
                                analysis["functions"].append(comp_text)
                                analysis["line_ranges"][comp_name] = (start_line, end_line)
                        else:
                            # No line numbers, add to appropriate list
                            if "class" in comp_text.lower():
                                analysis["classes"].append(comp_text)
                            else:
                                analysis["functions"].append(comp_text)
                                
        # Convert key_functionality from list to string if requested
        if len(analysis["key_functionality"]) == 1:
            analysis["key_functionality"] = analysis["key_functionality"][0]
        elif len(analysis["key_functionality"]) == 0:
            analysis["key_functionality"] = ""
            
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