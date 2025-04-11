import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import time
from ..database_connector import DatabaseConnector

class QdrantConnector(DatabaseConnector):
    """
    Connector for Qdrant vector database.
    """
    
    def __init__(self, 
                 host: str = "localhost", 
                 port: int = 6333,
                 collection_name: str = "benchmark_collection",
                 vector_dim: int = 128,
                 metric_type: str = "Cosine"):
        """
        Initialize Qdrant connector.
        
        Args:
            host: Qdrant server host
            port: Qdrant server port
            collection_name: Name of the collection to use
            vector_dim: Dimensionality of vectors
            metric_type: Distance metric ("Cosine", "Euclid", "Dot")
        """
        # Map metric types to Qdrant format
        metric_map = {
            "L2": "Euclid",
            "IP": "Dot",
            "COSINE": "Cosine"
        }
        
        qdrant_metric = metric_type
        if metric_type.upper() in metric_map:
            qdrant_metric = metric_map[metric_type.upper()]
            
        super().__init__(collection_name, vector_dim, qdrant_metric)
        self.host = host
        self.port = port
        self.client = None
    
    def connect(self) -> bool:
        """Connect to Qdrant server."""
        try:
            from qdrant_client import QdrantClient
            
            self.client = QdrantClient(host=self.host, port=self.port)
            self._connected = True
            
            return True
        except Exception as e:
            print(f"Failed to connect to Qdrant: {e}")
            return False
    
    def create_collection(self) -> bool:
        """Create a collection if it doesn't exist."""
        try:
            from qdrant_client.models import VectorParams, Distance
            
            if self.client is None:
                if not self.connect():
                    return False
            
            # Map metric type to Qdrant Distance enum
            distance_map = {
                "Cosine": Distance.COSINE,
                "Euclid": Distance.EUCLID,
                "Dot": Distance.DOT
            }
            
            distance = distance_map.get(self.metric_type, Distance.COSINE)
            
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                # Create collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_dim, distance=distance)
                )
            
            return True
        except Exception as e:
            print(f"Failed to create Qdrant collection: {e}")
            return False
    
    def insert_vectors(self, vectors: List[Tuple[int, np.ndarray]]) -> bool:
        """Insert vectors into Qdrant."""
        try:
            from qdrant_client.models import PointStruct
            
            if self.client is None:
                if not self.connect():
                    return False
            
            # Prepare points for insertion
            points = [
                PointStruct(
                    id=int(v[0]),
                    vector=v[1].tolist(),
                    payload={"id": int(v[0])}
                )
                for v in vectors
            ]
            
            # Insert points
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            return True
        except Exception as e:
            print(f"Failed to insert vectors into Qdrant: {e}")
            return False
    
    def search_vectors(self, query_vectors: List[np.ndarray], 
                      top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar vectors in Qdrant."""
        try:
            if self.client is None:
                if not self.connect():
                    return []
            
            # Execute search for each query vector
            results = []
            for i, query_vector in enumerate(query_vectors):
                search_result = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector.tolist(),
                    limit=top_k
                )
                
                # Format results
                hits = []
                for hit in search_result:
                    hits.append({
                        "id": hit.id,
                        "score": hit.score,
                        "payload": hit.payload
                    })
                
                results.append({
                    "query_index": i,
                    "hits": hits
                })
            
            return results
        except Exception as e:
            print(f"Failed to search vectors in Qdrant: {e}")
            return []
    
    def disconnect(self) -> bool:
        """Disconnect from Qdrant."""
        try:
            if self.client:
                # Qdrant client doesn't have an explicit disconnect method
                self.client = None
            
            self._connected = False
            return True
        except Exception as e:
            print(f"Failed to disconnect from Qdrant: {e}")
            return False