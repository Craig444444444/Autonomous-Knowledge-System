import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
import hashlib
import pickle
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

LOGGER = logging.getLogger("aks")

class VectorDB:
    """
    Vector database implementation for the Autonomous Knowledge System (AKS)
    with support for semantic search, clustering, and efficient similarity operations.
    """
    def __init__(self, storage_path: Path, dimension: int = 768):
        """
        Initialize the vector database.
        
        Args:
            storage_path: Directory for storing vector data
            dimension: Dimensionality of the vectors
        """
        self.storage_path = storage_path.resolve()
        self.dimension = dimension
        self.index = {}  # {vector_hash: (vector, metadata)}
        self.metadata_index = {}  # {doc_id: [vector_hashes]}
        
        # Create directories if they don't exist
        self.storage_path.mkdir(parents=True, exist_ok=True)
        (self.storage_path / "vectors").mkdir(exist_ok=True)
        (self.storage_path / "metadata").mkdir(exist_ok=True)
        
        # Performance optimization
        self._normalized_cache = {}
        self._batch_size = 1000
        
        LOGGER.info(f"Initialized VectorDB at {self.storage_path} (dim={dimension})")

    def _vector_hash(self, vector: np.ndarray) -> str:
        """Generate a unique hash for a vector."""
        return hashlib.sha256(vector.tobytes()).hexdigest()

    def add_vector(self, vector: np.ndarray, metadata: Dict) -> str:
        """
        Add a vector to the database with associated metadata.
        
        Args:
            vector: The vector to add (numpy array)
            metadata: Dictionary containing metadata (must include 'doc_id')
            
        Returns:
            The vector hash if successful, None otherwise
        """
        try:
            # Validate input
            if vector.shape != (self.dimension,):
                raise ValueError(f"Vector must have shape ({self.dimension},)")
            if 'doc_id' not in metadata:
                raise ValueError("Metadata must contain 'doc_id'")
                
            # Generate unique hash
            vector_hash = self._vector_hash(vector)
            
            # Store in memory
            self.index[vector_hash] = (vector, metadata)
            
            # Update metadata index
            doc_id = metadata['doc_id']
            if doc_id not in self.metadata_index:
                self.metadata_index[doc_id] = []
            self.metadata_index[doc_id].append(vector_hash)
            
            # Persist to disk
            self._save_vector(vector_hash, vector, metadata)
            
            LOGGER.debug(f"Added vector {vector_hash[:8]} for doc {doc_id}")
            return vector_hash
            
        except Exception as e:
            LOGGER.error(f"Failed to add vector: {e}")
            return None

    def _save_vector(self, vector_hash: str, vector: np.ndarray, metadata: Dict) -> None:
        """Persist vector and metadata to disk."""
        try:
            # Save vector
            vector_path = self.storage_path / "vectors" / f"{vector_hash}.npy"
            np.save(vector_path, vector)
            
            # Save metadata
            metadata_path = self.storage_path / "metadata" / f"{vector_hash}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
                
        except Exception as e:
            LOGGER.error(f"Failed to persist vector {vector_hash[:8]}: {e}")

    def batch_add_vectors(self, vectors: List[np.ndarray], metadatas: List[Dict]) -> List[str]:
        """
        Add multiple vectors to the database efficiently.
        
        Args:
            vectors: List of vectors to add
            metadatas: List of corresponding metadata dictionaries
            
        Returns:
            List of vector hashes
        """
        if len(vectors) != len(metadatas):
            raise ValueError("Vectors and metadatas must have same length")
            
        hashes = []
        with ThreadPoolExecutor() as executor:
            futures = []
            for vector, metadata in zip(vectors, metadatas):
                futures.append(executor.submit(self.add_vector, vector, metadata))
            
            for future in futures:
                try:
                    if (vector_hash := future.result()):
                        hashes.append(vector_hash)
                except Exception as e:
                    LOGGER.error(f"Batch add failed: {e}")
                    
        return hashes

    def get_vector(self, vector_hash: str) -> Optional[Tuple[np.ndarray, Dict]]:
        """
        Retrieve a vector and its metadata by hash.
        
        Args:
            vector_hash: The hash of the vector to retrieve
            
        Returns:
            Tuple of (vector, metadata) if found, None otherwise
        """
        # First try memory
        if vector_hash in self.index:
            return self.index[vector_hash]
            
        # Fall back to disk
        try:
            vector_path = self.storage_path / "vectors" / f"{vector_hash}.npy"
            metadata_path = self.storage_path / "metadata" / f"{vector_hash}.json"
            
            if vector_path.exists() and metadata_path.exists():
                vector = np.load(vector_path)
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Cache in memory
                self.index[vector_hash] = (vector, metadata)
                return vector, metadata
                
        except Exception as e:
            LOGGER.error(f"Failed to load vector {vector_hash[:8]}: {e}")
            
        return None

    def get_vectors_by_doc(self, doc_id: str) -> List[Tuple[np.ndarray, Dict]]:
        """
        Retrieve all vectors associated with a document ID.
        
        Args:
            doc_id: The document ID to retrieve vectors for
            
        Returns:
            List of (vector, metadata) tuples
        """
        vectors = []
        if doc_id in self.metadata_index:
            for vector_hash in self.metadata_index[doc_id]:
                if result := self.get_vector(vector_hash):
                    vectors.append(result)
        return vectors

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """Normalize a vector to unit length."""
        if vector_hash := self._vector_hash(vector):
            if vector_hash in self._normalized_cache:
                return self._normalized_cache[vector_hash]
                
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
            
        normalized = vector / norm
        if vector_hash:
            self._normalized_cache[vector_hash] = normalized
        return normalized

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        norm1 = self._normalize(vec1)
        norm2 = self._normalize(vec2)
        return np.dot(norm1, norm2)

    def nearest_neighbors(self, query_vector: np.ndarray, k: int = 5) -> List[Dict]:
        """
        Find the k nearest neighbors to the query vector.
        
        Args:
            query_vector: The vector to compare against
            k: Number of neighbors to return
            
        Returns:
            List of dictionaries containing:
                - vector: The neighbor vector
                - metadata: Associated metadata
                - similarity: Cosine similarity score
        """
        if not self.index:
            return []
            
        # Normalize query vector once
        query_norm = self._normalize(query_vector)
        
        # Calculate similarities in batches
        results = []
        vector_hashes = list(self.index.keys())
        
        for i in range(0, len(vector_hashes), self._batch_size):
            batch_hashes = vector_hashes[i:i + self._batch_size]
            batch_vectors = [self.index[h][0] for h in batch_hashes]
            batch_metadatas = [self.index[h][1] for h in batch_hashes]
            
            # Batch normalize and compute dot products
            norms = np.array([self._normalize(v) for v in batch_vectors])
            similarities = np.dot(norms, query_norm)
            
            # Add to results
            for vec_hash, vec, meta, sim in zip(batch_hashes, batch_vectors, batch_metadatas, similarities):
                results.append({
                    'vector_hash': vec_hash,
                    'vector': vec,
                    'metadata': meta,
                    'similarity': float(sim)
                })
        
        # Sort and return top k
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:k]

    def semantic_search(self, query_vector: np.ndarray, threshold: float = 0.7) -> List[Dict]:
        """
        Perform semantic search with similarity threshold.
        
        Args:
            query_vector: The vector to search with
            threshold: Minimum similarity score (0-1)
            
        Returns:
            List of matching items with similarity scores
        """
        neighbors = self.nearest_neighbors(query_vector, k=len(self.index))
        return [item for item in neighbors if item['similarity'] >= threshold]

    def cluster_vectors(self, n_clusters: int = 5) -> Dict[int, List[str]]:
        """
        Cluster vectors using k-means.
        
        Args:
            n_clusters: Number of clusters to create
            
        Returns:
            Dictionary mapping cluster IDs to lists of vector hashes
        """
        try:
            from sklearn.cluster import KMeans
            
            if not self.index:
                return {}
                
            vectors = np.array([v[0] for v in self.index.values()])
            hashes = list(self.index.keys())
            
            # Normalize vectors for clustering
            norms = np.array([self._normalize(v) for v in vectors])
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(norms)
            
            # Group results
            clustered = {}
            for cluster_id, vec_hash in zip(clusters, hashes):
                if cluster_id not in clustered:
                    clustered[cluster_id] = []
                clustered[cluster_id].append(vec_hash)
                
            LOGGER.info(f"Created {n_clusters} clusters with {len(hashes)} vectors")
            return clustered
            
        except ImportError:
            LOGGER.error("scikit-learn not available for clustering")
            return {}
        except Exception as e:
            LOGGER.error(f"Clustering failed: {e}")
            return {}

    def save_index(self) -> bool:
        """Persist the current index state to disk."""
        try:
            index_path = self.storage_path / "index.pkl"
            with open(index_path, 'wb') as f:
                pickle.dump({
                    'index': self.index,
                    'metadata_index': self.metadata_index,
                    'dimension': self.dimension
                }, f)
                
            LOGGER.info(f"Saved vector index to {index_path}")
            return True
            
        except Exception as e:
            LOGGER.error(f"Failed to save index: {e}")
            return False

    def load_index(self) -> bool:
        """Load the index from disk."""
        try:
            index_path = self.storage_path / "index.pkl"
            if not index_path.exists():
                LOGGER.warning("No existing index found")
                return False
                
            with open(index_path, 'rb') as f:
                data = pickle.load(f)
                self.index = data['index']
                self.metadata_index = data['metadata_index']
                self.dimension = data['dimension']
                
            LOGGER.info(f"Loaded vector index with {len(self.index)} vectors")
            return True
            
        except Exception as e:
            LOGGER.error(f"Failed to load index: {e}")
            return False

    def backup(self, backup_path: Path) -> bool:
        """
        Create a backup of the vector database.
        
        Args:
            backup_path: Directory to store the backup
            
        Returns:
            True if backup succeeded, False otherwise
        """
        try:
            backup_path = backup_path.resolve()
            backup_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = backup_path / f"vectordb_backup_{timestamp}.zip"
            
            with zipfile.ZipFile(backup_file, 'w') as zipf:
                # Add index
                if (self.storage_path / "index.pkl").exists():
                    zipf.write(self.storage_path / "index.pkl", "index.pkl")
                
                # Add vectors
                for vec_file in (self.storage_path / "vectors").glob("*.npy"):
                    zipf.write(vec_file, f"vectors/{vec_file.name}")
                
                # Add metadata
                for meta_file in (self.storage_path / "metadata").glob("*.json"):
                    zipf.write(meta_file, f"metadata/{meta_file.name}")
            
            LOGGER.info(f"Created backup at {backup_file}")
            return True
            
        except Exception as e:
            LOGGER.error(f"Backup failed: {e}")
            return False

    def stats(self) -> Dict:
        """Get database statistics."""
        return {
            'total_vectors': len(self.index),
            'total_documents': len(self.metadata_index),
            'dimension': self.dimension,
            'storage_size': sum(f.stat().st_size for f in (self.storage_path / "vectors").glob('*') 
                              if f.is_file()) / (1024 * 1024)  # in MB
        }
