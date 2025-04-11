import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import time
from ..database_connector import DatabaseConnector

class MilvusConnector(DatabaseConnector):
    """
    Connector for Milvus vector database.
    """
    
    def __init__(self, 
                 host: str = "localhost", 
                 port: str = "19530",
                 collection_name: str = "benchmark_collection",
                 vector_dim: int = 128,
                 metric_type: str = "L2"):
        """
        Initialize Milvus connector.
        
        Args:
            host: Milvus server host
            port: Milvus server port
            collection_name: Name of the collection to use
            vector_dim: Dimensionality of vectors
            metric_type: Distance metric ("L2", "IP", "COSINE")
        """
        super().__init__(collection_name, vector_dim, metric_type)
        self.host = host
        self.port = port
        self.client = None
        self.collection = None
    
    def connect(self) -> bool:
        """Connect to Milvus server."""
        try:
            from pymilvus import connections, Collection, utility
            
            # Connect to Milvus
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )
            
            self._connected = True
            
            # Check if collection exists
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                self.collection.load()
            
            return True
        except Exception as e:
            print(f"Failed to connect to Milvus: {e}")
            return False
    
    def create_collection(self) -> bool:
        """Create a collection if it doesn't exist."""
        try:
            from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, utility
            
            # Check if collection exists
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                self.collection.load()
                return True
            
            # Define fields
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim)
            ]
            
            # Define schema
            schema = CollectionSchema(fields=fields, description=f"Benchmark collection with {self.vector_dim}d vectors")
            
            # Create collection
            self.collection = Collection(name=self.collection_name, schema=schema)
            
            # Create index
            index_params = {
                "index_type": "HNSW",
                "metric_type": self.metric_type,
                "params": {"M": 16, "efConstruction": 200}
            }
            self.collection.create_index(field_name="vector", index_params=index_params)
            
            # Load collection
            self.collection.load()
            
            return True
        except Exception as e:
            print(f"Failed to create Milvus collection: {e}")
            return False
    
    def insert_vectors(self, vectors: List[Tuple[int, np.ndarray]]) -> bool:
        """Insert vectors into Milvus."""
        try:
            if self.collection is None:
                if not self.create_collection():
                    return False
            
            # Prepare data for insertion
            ids = [v[0] for v in vectors]
            embeddings = [v[1].tolist() for v in vectors]
            
            # Insert data
            insert_data = [
                ids,           # id field
                embeddings     # vector field
            ]
            
            self.collection.insert(insert_data)
            
            # Flush to ensure data is persisted
            self.collection.flush()
            
            return True
        except Exception as e:
            print(f"Failed to insert vectors into Milvus: {e}")
            return False
    
    def search_vectors(self, query_vectors: List[np.ndarray], 
                      top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar vectors in Milvus."""
        try:
            if self.collection is None:
                raise ValueError("Collection not initialized")
            
            # Convert numpy arrays to lists
            query_list = [vec.tolist() for vec in query_vectors]
            
            # Define search parameters
            search_params = {
                "metric_type": self.metric_type,
                "params": {"ef": 100}
            }
            
            # Execute search
            results = self.collection.search(
                data=query_list,
                anns_field="vector",
                param=search_params,
                limit=top_k,
                output_fields=["id"]
            )
            
            # Format results
            formatted_results = []
            for i, result in enumerate(results):
                hits = []
                for hit in result:
                    hits.append({
                        "id": hit.id,
                        "score": hit.score,
                        "distance": hit.distance
                    })
                
                formatted_results.append({
                    "query_index": i,
                    "hits": hits
                })
            
            return formatted_results
        except Exception as e:
            print(f"Failed to search vectors in Milvus: {e}")
            return []
    
    def disconnect(self) -> bool:
        """Disconnect from Milvus."""
        try:
            from pymilvus import connections
            
            if self.collection:
                self.collection.release()
            
            connections.disconnect("default")
            self._connected = False
            
            return True
        except Exception as e:
            print(f"Failed to disconnect from Milvus: {e}")
            return False