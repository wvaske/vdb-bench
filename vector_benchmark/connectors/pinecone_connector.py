import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import time
from ..database_connector import DatabaseConnector

class PineconeConnector(DatabaseConnector):
    """
    Connector for Pinecone vector database.
    """
    
    def __init__(self, 
                 api_key: str,
                 environment: str,
                 index_name: str = "benchmark-index",
                 vector_dim: int = 128,
                 metric_type: str = "cosine"):
        """
        Initialize Pinecone connector.
        
        Args:
            api_key: Pinecone API key
            environment: Pinecone environment
            index_name: Name of the index to use
            vector_dim: Dimensionality of vectors
            metric_type: Distance metric ("cosine", "euclidean", "dotproduct")
        """
        # Map metric types to Pinecone format
        metric_map = {
            "L2": "euclidean",
            "IP": "dotproduct",
            "COSINE": "cosine"
        }
        
        pinecone_metric = metric_type.lower()
        if metric_type.upper() in metric_map:
            pinecone_metric = metric_map[metric_type.upper()]
            
        super().__init__(index_name, vector_dim, pinecone_metric)
        self.api_key = api_key
        self.environment = environment
        self.client = None
        self.index = None
    
    def connect(self) -> bool:
        """Connect to Pinecone."""
        try:
            import pinecone
            
            # Initialize Pinecone
            pinecone.init(
                api_key=self.api_key,
                environment=self.environment
            )
            
            self.client = pinecone
            
            # Check if index exists
            if self.collection_name in pinecone.list_indexes():
                self.index = pinecone.Index(self.collection_name)
            
            self._connected = True
            return True
        except Exception as e:
            print(f"Failed to connect to Pinecone: {e}")
            return False
    
    def create_collection(self) -> bool:
        """Create an index if it doesn't exist."""
        try:
            if self.client is None:
                if not self.connect():
                    return False
            
            # Check if index exists
            if self.collection_name not in self.client.list_indexes():
                # Create index
                self.client.create_index(
                    name=self.collection_name,
                    dimension=self.vector_dim,
                    metric=self.metric_type
                )
                
                # Wait for index to be ready
                while not self.collection_name in self.client.list_indexes():
                    time.sleep(1)
            
            self.index = self.client.Index(self.collection_name)
            
            return True
        except Exception as e:
            print(f"Failed to create Pinecone index: {e}")
            return False
    
    def insert_vectors(self, vectors: List[Tuple[int, np.ndarray]]) -> bool:
        """Insert vectors into Pinecone."""
        try:
            if self.index is None:
                if not self.create_collection():
                    return False
            
            # Prepare vectors for insertion
            items = [
                (str(v[0]), v[1].tolist(), {"id": str(v[0])})
                for v in vectors
            ]
            
            # Insert in batches of 100 (Pinecone's recommended batch size)
            batch_size = 100
            for i in range(0, len(items), batch_size):
                batch = items[i:i+batch_size]
                
                # Format batch for Pinecone
                vectors_batch = [
                    {
                        "id": item[0],
                        "values": item[1],
                        "metadata": item[2]
                    }
                    for item in batch
                ]
                
                # Upsert batch
                self.index.upsert(vectors=vectors_batch)
            
            return True
        except Exception as e:
            print(f"Failed to insert vectors into Pinecone: {e}")
            return False
    
    def search_vectors(self, query_vectors: List[np.ndarray], 
                      top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar vectors in Pinecone."""
        try:
            if self.index is None:
                if not self.create_collection():
                    return []
            
            # Execute search for each query vector
            results = []
            for i, query_vector in enumerate(query_vectors):
                # Convert numpy array to list
                query_list = query_vector.tolist()
                
                # Execute search
                search_result = self.index.query(
                    vector=query_list,
                    top_k=top_k,
                    include_metadata=True
                )
                
                # Format results
                hits = []
                for match in search_result.matches:
                    hits.append({
                        "id": match.id,
                        "score": match.score,
                        "metadata": match.metadata
                    })
                
                results.append({
                    "query_index": i,
                    "hits": hits
                })
            
            return results
        except Exception as e:
            print(f"Failed to search vectors in Pinecone: {e}")
            return []
    
    def disconnect(self) -> bool:
        """Disconnect from Pinecone."""
        try:
            # Pinecone doesn't have an explicit disconnect method
            self.index = None
            self.client = None
            self._connected = False
            
            return True
        except Exception as e:
            print(f"Failed to disconnect from Pinecone: {e}")
            return False