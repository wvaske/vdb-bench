import numpy as np
from typing import List, Tuple, Union, Optional
import multiprocessing as mp
from pathlib import Path
import pickle
import os

class VectorGenerator:
    """
    Generates random vectors for benchmarking vector databases.
    """
    
    def __init__(self, 
                 dimensions: int, 
                 data_type: str = 'float32',
                 seed: Optional[int] = None):
        """
        Initialize the vector generator.
        
        Args:
            dimensions: The dimensionality of vectors to generate
            data_type: The data type of vector values ('float32', 'float64', 'int32', 'int64')
            seed: Random seed for reproducibility
        """
        self.dimensions = dimensions
        self.data_type = data_type
        self.seed = seed
        
        # Validate data type
        valid_types = ['float32', 'float64', 'int32', 'int64']
        if data_type not in valid_types:
            raise ValueError(f"Data type must be one of {valid_types}")
            
        # Set numpy data type
        self.np_dtype = getattr(np, data_type)
        
        # Initialize random generator
        self.rng = np.random.default_rng(seed)
    
    def generate_vector(self) -> np.ndarray:
        """Generate a single random vector."""
        if self.data_type.startswith('float'):
            # Generate floats between -1.0 and 1.0
            vector = self.rng.uniform(-1.0, 1.0, self.dimensions).astype(self.np_dtype)
        else:
            # Generate integers between -100 and 100
            vector = self.rng.integers(-100, 100, self.dimensions).astype(self.np_dtype)
        
        return vector
    
    def generate_vectors(self, count: int) -> List[np.ndarray]:
        """Generate multiple random vectors."""
        return [self.generate_vector() for _ in range(count)]
    
    def generate_vectors_with_ids(self, 
                                 start_id: int, 
                                 count: int) -> List[Tuple[int, np.ndarray]]:
        """Generate vectors with sequential IDs starting from start_id."""
        return [(start_id + i, self.generate_vector()) for i in range(count)]
    
    def _generate_batch(self, args: Tuple[int, int]) -> List[Tuple[int, np.ndarray]]:
        """Helper method for parallel generation."""
        start_id, count = args
        return self.generate_vectors_with_ids(start_id, count)
    
    def generate_vectors_parallel(self, 
                                 total_count: int, 
                                 processes: int = None,
                                 save_path: Optional[str] = None) -> Union[List[Tuple[int, np.ndarray]], List[str]]:
        """
        Generate vectors using multiple processes.
        
        Args:
            total_count: Total number of vectors to generate
            processes: Number of processes to use (defaults to CPU count)
            save_path: If provided, save batches to files and return file paths
            
        Returns:
            Either a list of (id, vector) tuples or a list of file paths if save_path is provided
        """
        processes = processes or mp.cpu_count()
        batch_size = total_count // processes
        remainder = total_count % processes
        
        # Prepare batches
        batches = []
        start_id = 0
        for i in range(processes):
            # Add remainder to the first batch
            count = batch_size + (remainder if i == 0 else 0)
            batches.append((start_id, count))
            start_id += count
        
        # Generate vectors in parallel
        with mp.Pool(processes) as pool:
            results = pool.map(self._generate_batch, batches)
        
        # Flatten results if not saving to files
        if not save_path:
            return [item for sublist in results for item in sublist]
        
        # Save to files and return paths
        file_paths = []
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for i, batch in enumerate(results):
            file_path = save_dir / f"vectors_batch_{i}.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(batch, f)
            file_paths.append(str(file_path))
        
        return file_paths