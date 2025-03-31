import os
import json
import sqlite3
import logging
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)

class AnalysisStorageService:
    """
    Service for storing and retrieving project analysis results.
    Uses SQLite for basic metadata and JSON files for detailed analysis.
    """
    
    def __init__(self, storage_path: str):
        """
        Initialize the storage service.
        
        Args:
            storage_path: Path to store analysis data
        """
        self.storage_path = os.path.abspath(storage_path)
        self.db_path = os.path.join(self.storage_path, "project_analysis.db")
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        logger.info(f"Initialized analysis storage at {self.storage_path}")
    
    def _init_database(self):
        """Initialize SQLite database with required schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create projects table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            path TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(path)
        )
        ''')
        
        # Create files table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            path TEXT NOT NULL,
            relative_path TEXT NOT NULL,
            file_type TEXT,
            last_modified TIMESTAMP,
            analysis_path TEXT,
            FOREIGN KEY (project_id) REFERENCES projects(id),
            UNIQUE(project_id, path)
        )
        ''')
        
        # Create relationships table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS relationships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file_id INTEGER NOT NULL,
            target_file_id INTEGER NOT NULL,
            relationship_type TEXT NOT NULL,
            FOREIGN KEY (source_file_id) REFERENCES files(id),
            FOREIGN KEY (target_file_id) REFERENCES files(id),
            UNIQUE(source_file_id, target_file_id, relationship_type)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_project_analysis(self, project_path: str, analysis_results: Dict[str, Any]) -> int:
        """
        Store analysis results for a project.
        
        Args:
            project_path: Path to the project
            analysis_results: Dictionary of analysis results
            
        Returns:
            Project ID in the database
        """
        project_name = os.path.basename(project_path)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert or update project
        cursor.execute(
            "INSERT OR REPLACE INTO projects (name, path) VALUES (?, ?)",
            (project_name, project_path)
        )
        
        # Get project ID
        project_id = cursor.lastrowid
        
        # Create analysis directory for this project
        project_dir = os.path.join(self.storage_path, f"project_{project_id}")
        os.makedirs(project_dir, exist_ok=True)
        
        # Store each file's analysis
        for file_path, file_analysis in analysis_results.items():
            relative_path = os.path.relpath(file_path, project_path)
            file_type = os.path.splitext(file_path)[1]
            
            # Create a safe path for storing the analysis
            safe_path = relative_path.replace(os.path.sep, "_")
            analysis_path = os.path.join(project_dir, f"{safe_path}.json")
            
            # Save detailed analysis as JSON
            with open(analysis_path, 'w') as f:
                json.dump(file_analysis, f, indent=2)
            
            # Save file metadata in database
            cursor.execute(
                """
                INSERT OR REPLACE INTO files 
                (project_id, path, relative_path, file_type, last_modified, analysis_path) 
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id, 
                    file_path, 
                    relative_path, 
                    file_type, 
                    os.path.getmtime(file_path) if os.path.exists(file_path) else None,
                    analysis_path
                )
            )
        
        conn.commit()
        conn.close()
        
        logger.info(f"Stored analysis for project {project_name} with ID {project_id}")
        return project_id
    
    def get_project_analysis(self, project_path: str) -> Dict[str, Any]:
        """
        Retrieve analysis results for a project.
        
        Args:
            project_path: Path to the project
            
        Returns:
            Dictionary of analysis results
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get project ID
        cursor.execute("SELECT id FROM projects WHERE path = ?", (project_path,))
        result = cursor.fetchone()
        
        if not result:
            conn.close()
            logger.warning(f"No analysis found for project at {project_path}")
            return {}
        
        project_id = result[0]
        
        # Get all files for this project
        cursor.execute("SELECT path, analysis_path FROM files WHERE project_id = ?", (project_id,))
        files = cursor.fetchall()
        
        conn.close()
        
        # Load analysis for each file
        analysis_results = {}
        for file_path, analysis_path in files:
            try:
                with open(analysis_path, 'r') as f:
                    file_analysis = json.load(f)
                analysis_results[file_path] = file_analysis
            except Exception as e:
                logger.warning(f"Failed to load analysis for {file_path}: {e}")
        
        logger.info(f"Loaded analysis for project at {project_path} with {len(analysis_results)} files")
        return analysis_results
    
    def store_relationships(self, project_id: int, relationships: List[Tuple[str, str, str]]):
        """
        Store relationships between files.
        
        Args:
            project_id: Project ID in the database
            relationships: List of (source_path, target_path, relationship_type) tuples
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get file IDs for the project
        cursor.execute("SELECT id, path FROM files WHERE project_id = ?", (project_id,))
        file_id_map = {path: file_id for file_id, path in cursor.fetchall()}
        
        # Store relationships
        for source_path, target_path, rel_type in relationships:
            if source_path in file_id_map and target_path in file_id_map:
                source_id = file_id_map[source_path]
                target_id = file_id_map[target_path]
                
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO relationships 
                    (source_file_id, target_file_id, relationship_type) 
                    VALUES (?, ?, ?)
                    """,
                    (source_id, target_id, rel_type)
                )
        
        conn.commit()
        conn.close()
        
        logger.info(f"Stored {len(relationships)} relationships for project {project_id}")