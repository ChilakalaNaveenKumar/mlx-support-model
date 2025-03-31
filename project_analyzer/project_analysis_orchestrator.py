import os
import logging
from typing import Dict, List, Any, Tuple

from project_analyzer.project_reader import ProjectReaderService
from project_analyzer.analysis_storage import AnalysisStorageService
from project_analyzer.vector_embedding import VectorEmbeddingService

# We'll implement LLM Processing Service later
# from project_analyzer.llm_processor import LLMProcessingService

logger = logging.getLogger(__name__)

class ProjectAnalysisOrchestrator:
    """
    Orchestrates the project analysis workflow.
    Coordinates between all services to analyze a project.
    """
    
    def __init__(self, 
                storage_path: str = None, 
                model_service = None):
        """
        Initialize the project analysis orchestrator.
        
        Args:
            storage_path: Path to store analysis data (default: ~/.project_analyzer)
            model_service: LLM model service (if None, will be implemented later)
        """
        # Set default storage path if not provided
        if storage_path is None:
            storage_path = os.path.join(os.path.expanduser("~"), ".project_analyzer")
            
        self.storage_path = os.path.abspath(storage_path)
        
        # Create storage directories
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Initialize storage paths
        db_path = os.path.join(self.storage_path, "db")
        embedding_path = os.path.join(self.storage_path, "embeddings")
        
        # Initialize services
        self.storage_service = AnalysisStorageService(db_path)
        self.embedding_service = VectorEmbeddingService(embedding_path)
        
        # Initialize model service (to be implemented)
        self.model_service = model_service
        
        logger.info(f"Initialized project analysis orchestrator with storage at {self.storage_path}")
    
    def analyze_project(self, project_path: str, incremental: bool = True) -> int:
        """
        Analyze a project.
        
        Args:
            project_path: Path to the project
            incremental: Whether to do incremental analysis
            
        Returns:
            Project ID in the database
        """
        # Initialize project reader
        reader_service = ProjectReaderService(project_path)
        
        # Read all project files
        logger.info(f"Reading project files from {project_path}")
        project_files = reader_service.read_project()
        
        file_contents = {file_path: content for file_path, content in project_files}
        logger.info(f"Read {len(file_contents)} files from project")
        
        # TODO: Implement LLM processing - for now, create placeholder analysis
        if self.model_service is None:
            logger.warning("LLM Processing Service not yet implemented, using placeholder analysis")
            analysis_results = self._create_placeholder_analysis(file_contents)
        else:
            # This will be implemented later
            # analysis_results = self.model_service.analyze_files(file_contents)
            analysis_results = self._create_placeholder_analysis(file_contents)
        
        # Store analysis results
        project_id = self.storage_service.store_project_analysis(project_path, analysis_results)
        
        # Generate and store embeddings
        embeddings = self.embedding_service.generate_embeddings(file_contents, analysis_results)
        self.embedding_service.store_embeddings(project_id, embeddings)
        
        logger.info(f"Completed analysis for project at {project_path}")
        return project_id
    
    def _create_placeholder_analysis(self, file_contents: Dict[str, str]) -> Dict[str, Any]:
        """
        Create placeholder analysis results for demonstration.
        
        Args:
            file_contents: Dictionary of file paths to contents
            
        Returns:
            Dictionary of file paths to analysis results
        """
        analysis_results = {}
        
        for file_path, content in file_contents.items():
            # Create a simple analysis structure
            file_type = os.path.splitext(file_path)[1]
            line_count = content.count('\n') + 1
            
            analysis = {
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "file_type": file_type,
                "line_count": line_count,
                "summary": f"This is a {file_type} file with {line_count} lines.",
                "components": [],
                "dependencies": [],
                "purpose": "Placeholder analysis - LLM processing not yet implemented",
                "symbols": []
            }
            
            analysis_results[file_path] = analysis
            
        return analysis_results
    
    def query_project(self, project_path: str, query: str, top_k: int = 5) -> List[Tuple[str, float, Any]]:
        """
        Query a project for relevant files.
        
        Args:
            project_path: Path to the project
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (file_path, score, analysis) tuples
        """
        # Get project ID
        conn = self.storage_service._db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT id FROM projects WHERE path = ?", (project_path,))
        result = cursor.fetchone()
        
        if not result:
            conn.close()
            logger.warning(f"No analysis found for project at {project_path}")
            return []
        
        project_id = result[0]
        conn.close()
        
        # Search for relevant files
        search_results = self.embedding_service.search_by_query(project_id, query, top_k)
        
        # Get analysis for each file
        project_analysis = self.storage_service.get_project_analysis(project_path)
        
        # Combine search results with analysis
        results = []
        for file_path, score in search_results:
            analysis = project_analysis.get(file_path, {})
            results.append((file_path, score, analysis))
            
        return results