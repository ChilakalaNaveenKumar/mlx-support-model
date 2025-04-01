import os
import logging
from typing import Dict, List, Any

logger = logging.getLogger("code_analyzer")

class MindMapGenerator:
    """
    Generates a text-based mind map from analysis results.
    """
    
    def __init__(self):
        """Initialize the mind map generator."""
        logger.info("Initialized mind map generator")
    
    def generate_text_mind_map(self, mind_map_data: Dict[str, Any]) -> str:
        """
        Generate a text-based mind map.
        
        Args:
            mind_map_data: Mind map data structure from LLMProcessor
            
        Returns:
            Text representation of the mind map
        """
        result = []
        self._generate_tree_node(mind_map_data, "", True, result)
        return "\n".join(result)
    
    def _generate_tree_node(self, node: Dict[str, Any], prefix: str, is_root: bool, result: List[str]):
        """Recursively generate a tree node representation."""
        if is_root:
            result.append(f"{node['name']}/")
            new_prefix = ""
        else:
            # Add description as comment if available
            description = node.get('description', '')
            description_text = f" # {description}" if description else ""
            
            # Add functionality as part of comment if available
            functionality = node.get('functionality', '')
            if functionality and description:
                description_text += f" - {functionality}"
            elif functionality:
                description_text = f" # {functionality}"
            
            result.append(f"{prefix}├── {node['name']}{description_text}")
            new_prefix = prefix + "│   "
        
        # Process children
        if "children" in node and node["children"]:
            for i, child in enumerate(node["children"]):
                is_last = (i == len(node["children"]) - 1)
                
                # Use different prefix for last item
                if is_last:
                    child_prefix = prefix + "    "
                    result[-1] = result[-1].replace("├──", "└──")
                else:
                    child_prefix = new_prefix
                
                self._generate_tree_node(child, child_prefix, False, result)
    
    def save_mind_map(self, mind_map: str, output_path: str):
        """
        Save the mind map to a file.
        
        Args:
            mind_map: Text representation of the mind map
            output_path: Path to save the file
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(mind_map)
            logger.info(f"Mind map saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving mind map: {e}")
            
    def visualize_html_mind_map(self, mind_map_data: Dict[str, Any], output_path: str):
        """
        Generate an HTML visualization of the mind map.
        
        Args:
            mind_map_data: Mind map data structure
            output_path: Path to save the HTML file
        """
        # Basic HTML template with simple CSS for indentation
        html = """<!DOCTYPE html>
<html>
<head>
    <title>Code Structure Mind Map</title>
    <style>
        body { font-family: monospace; }
        .container { margin: 20px; }
        .tree ul { list-style-type: none; }
        .tree li { margin: 10px 0; position: relative; }
        .tree li::before {
            content: "├── ";
            color: #999;
        }
        .tree li:last-child::before {
            content: "└── ";
        }
        .tree ul li ul { margin-left: 20px; }
        .description { color: #666; font-style: italic; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Code Structure Mind Map</h1>
        <div class="tree">
"""
        
        # Add mind map content
        html += self._generate_html_tree(mind_map_data)
        
        # Close tags
        html += """
        </div>
    </div>
</body>
</html>
"""
        
        # Save HTML file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)
            logger.info(f"HTML mind map saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving HTML mind map: {e}")
    
    def _generate_html_tree(self, node: Dict[str, Any]) -> str:
        """Generate HTML representation of a tree node."""
        html = ""
        
        # Node name and description
        description = node.get('description', '')
        functionality = node.get('functionality', '')
        
        # Combine description and functionality
        comment = ""
        if description:
            comment += description
        if functionality:
            if comment:
                comment += " - "
            comment += functionality
            
        if comment:
            comment_html = f' <span class="description"># {comment}</span>'
        else:
            comment_html = ''
        
        if "children" in node and node["children"]:
            # This node has children
            html += f"<ul><li><strong>{node['name']}</strong>{comment_html}<ul>"
            
            for child in node["children"]:
                html += f"<li>{self._generate_html_tree(child)}</li>"
                
            html += "</ul></li></ul>"
        else:
            # Leaf node
            html += f"<strong>{node['name']}</strong>{comment_html}"
            
        return html