from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

class DatabaseConnector(ABC):
    """
    Abstract base class for vector database connectors.
    """
    
    def __init__(self, 
                 collection_name: str,
                 vector_dim: int,
                 metric_type: str):
        """
        Initialize the database connector.
        
        Args:
            collection_name: Name of the collection/index
            vector_dim: Dimensionality of vectors
            metric_type: Distance metric type
        """
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.metric_type = metric_type
        self._connected = False
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the database.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def create_collection(self) -> bool:
        """
        Create a collection/index in the database.
        
        Returns:
            True if creation successful, False otherwise
        """
        pass
    
    @abstractmethod
    def insert_vectors(self, vectors: List[Tuple[int, np.ndarray]]) -> bool:
        """
        Insert vectors into the database.
        
        Args:
            vectors: List of (id, vector) tuples
            
        Returns:
            True if insertion successful, False otherwise
        """
        pass
    
    @abstractmethod
    def search_vectors(self, query_vectors: List[np.ndarray], 
                      top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the database.
        
        Args:
            query_vectors: List of query vectors
            top_k: Number of results to return per query
            
        Returns:
            List of search results
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """
        Disconnect from the database.
        
        Returns:
            True if disconnection successful, False otherwise
        """
        pass
    
    def is_connected(self) -> bool:
        """
        Check if connected to the database.
        
        Returns:
            True if connected, False otherwise
        """
        return self._connected
    
    def load_vectors_from_file(self, file_path: str) -> List[Tuple[int, np.ndarray]]:
        """Load vectors from a pickle file."""
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    def load_and_insert_from_files(self, file_paths: List[str]) -> bool:
        """Load vectors from files and insert them into the database."""
        for file_path in file_paths:
            vectors = self.load_vectors_from_file(file_path)
            success = self.insert_vectors(vectors)
            if not success:
                return False
        return True
    
    def _insert_batch(self, args: Tuple[str, int]) -> Dict[str, Any]:
        """Helper method for parallel insertion."""
        file_path, batch_id = args
        start_time = time.time()
        vectors = self.load_vectors_from_file(file_path)
        success = self.insert_vectors(vectors)
        end_time = time.time()
        
        return {
            'batch_id': batch_id,
            'file_path': file_path,
            'success': success,
            'count': len(vectors),
            'time': end_time - start_time
        }
    
    def parallel_insert_from_files(self, file_paths: List[str], 
                                  processes: int = None) -> List[Dict[str, Any]]:
        """
        Insert vectors from files using multiple processes.
        
        Args:
            file_paths: List of file paths containing vectors
            processes: Number of processes to use
            
        Returns:
            List of batch results
        """
        processes = processes or mp.cpu_count()
        
        # Prepare batch arguments
        batch_args = [(path, i) for i, path in enumerate(file_paths)]
        
        # Use process pool for insertion
        with mp.Pool(processes) as pool:
            results = pool.map(self._insert_batch, batch_args)
        
        return results