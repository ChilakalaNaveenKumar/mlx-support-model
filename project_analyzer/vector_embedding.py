import os
import pickle
import logging
import numpy as np
from typing import Dict, List, Any, Tuple
import time

logger = logging.getLogger(__name__)

class VectorEmbeddingService:
    """
    Service for creating and managing vector embeddings of files and analyses.
    Enables semantic search across the codebase.
    """
    
    def __init__(self, embedding_path: str):
        """
        Initialize the vector embedding service.
        
        Args:
            embedding_path: Path to store embeddings
        """
        self.embedding_path = os.path.abspath(embedding_path)
        
        # Create embedding directory if it doesn't exist
        os.makedirs(self.embedding_path, exist_ok=True)
        
        logger.info(f"Initialized vector embedding service at {self.embedding_path}")
        
        # Import sentence transformers for embeddings
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_available = True
            logger.info("Loaded sentence transformer model for embeddings")
        except ImportError:
            logger.warning("SentenceTransformer not available. Using fallback embedding method.")
            self.model = None
            self.embedding_available = False
    
    def generate_embeddings(self, file_contents: Dict[str, str], analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate embeddings for files and their analyses.
        
        Args:
            file_contents: Dictionary of file paths to file contents
            analysis_results: Dictionary of file paths to analysis results
            
        Returns:
            Dictionary with embeddings and metadata
        """
        texts_to_embed = []
        file_paths = []
        
        # Prepare texts for embedding
        for file_path, content in file_contents.items():
            # Skip very large files or binary content
            if len(content) > 100000 or content.startswith("[BINARY CONTENT:"):
                continue
                
            # Get analysis for this file
            analysis = analysis_results.get(file_path, {})
            
            # For code files, embed the content + summary (if available)
            if analysis and "summary" in analysis:
                combined_text = f"{content}\n\nSummary: {analysis['summary']}"
            else:
                combined_text = content
                
            texts_to_embed.append(combined_text)
            file_paths.append(file_path)
        
        # Generate embeddings
        embeddings = self._create_embeddings(texts_to_embed)
        
        # Create mapping of file paths to embeddings
        embedding_map = {}
        for i, file_path in enumerate(file_paths):
            if i < len(embeddings):
                embedding_map[file_path] = embeddings[i]
        
        # Package everything together
        result = {
            "embeddings": embedding_map,
            "timestamp": time.time()
        }
        
        logger.info(f"Generated embeddings for {len(embedding_map)} files")
        return result
    
    def _create_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Create embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if self.embedding_available and self.model:
            # Use sentence transformers for embedding
            try:
                return self.model.encode(texts)
            except Exception as e:
                logger.error(f"Error generating embeddings with model: {e}")
                return self._fallback_embeddings(texts)
        else:
            # Use fallback method
            return self._fallback_embeddings(texts)
    
    def _fallback_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Create simple fallback embeddings when transformer models aren't available.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of simple embedding vectors
        """
        import hashlib
        from collections import Counter
        
        embeddings = []
        
        for text in texts:
            # Create a simple bag-of-words embedding (very basic)
            words = text.lower().split()
            word_counts = Counter(words)
            
            # Create a deterministic hash-based embedding (100 dimensions)
            embedding = np.zeros(100)
            
            for word, count in word_counts.items():
                # Use word hash to determine index
                hash_obj = hashlib.md5(word.encode())
                index = int(hash_obj.hexdigest(), 16) % 100
                
                # Use count as value
                embedding[index] += count
            
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                
            embeddings.append(embedding)
            
        return embeddings
    
    def store_embeddings(self, project_id: int, embeddings: Dict[str, Any]):
        """
        Store embeddings for a project.
        
        Args:
            project_id: Project ID
            embeddings: Embedding data to store
        """
        embedding_file = os.path.join(self.embedding_path, f"project_{project_id}_embeddings.pkl")
        
        with open(embedding_file, 'wb') as f:
            pickle.dump(embeddings, f)
            
        logger.info(f"Stored embeddings for project {project_id}")
    
    def load_embeddings(self, project_id: int) -> Dict[str, Any]:
        """
        Load embeddings for a project.
        
        Args:
            project_id: Project ID
            
        Returns:
            Dictionary with embeddings and metadata
        """
        embedding_file = os.path.join(self.embedding_path, f"project_{project_id}_embeddings.pkl")
        
        if not os.path.exists(embedding_file):
            logger.warning(f"No embeddings found for project {project_id}")
            return {}
            
        try:
            with open(embedding_file, 'rb') as f:
                embeddings = pickle.load(f)
                
            logger.info(f"Loaded embeddings for project {project_id}")
            return embeddings
        except Exception as e:
            logger.error(f"Error loading embeddings for project {project_id}: {e}")
            return {}
    
    def search_by_query(self, project_id: int, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for files related to a query.
        
        Args:
            project_id: Project ID
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (file_path, score) tuples
        """
        # Load embeddings
        project_embeddings = self.load_embeddings(project_id)
        
        if not project_embeddings or "embeddings" not in project_embeddings:
            logger.warning(f"No embeddings available for project {project_id}")
            return []
            
        embedding_map = project_embeddings["embeddings"]
        
        # Create query embedding
        query_embedding = self._create_embeddings([query])[0]
        
        # Calculate similarity scores
        results = []
        for file_path, file_embedding in embedding_map.items():
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, file_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(file_embedding)
            )
            results.append((file_path, float(similarity)))
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k results
        return results[:top_k]