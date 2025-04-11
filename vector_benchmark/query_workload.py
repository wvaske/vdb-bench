import numpy as np
from typing import List, Dict, Any, Optional, Callable
import time
import multiprocessing as mp
import random
from .database_connector import DatabaseConnector
from .metrics_collector import MetricsCollector
from .vector_generator import VectorGenerator

class QueryWorkload:
    """
    Defines and executes query patterns against vector databases.
    """
    
    def __init__(self, 
                 db_connector: DatabaseConnector,
                 vector_generator: VectorGenerator,
                 metrics_collector: MetricsCollector):
        """
        Initialize the query workload.
        
        Args:
            db_connector: Database connector to use
            vector_generator: Vector generator for query vectors
            metrics_collector: Metrics collector to record results
        """
        self.db_connector = db_connector
        self.vector_generator = vector_generator
        self.metrics_collector = metrics_collector
    
    def execute_single_query(self, 
                           query_id: int, 
                           query_vector: np.ndarray,
                           top_k: int = 10) -> Dict[str, Any]:
        """
        Execute a single query and record metrics.
        
        Args:
            query_id: ID of the query
            query_vector: Query vector
            top_k: Number of results to return
            
        Returns:
            Query results and metrics
        """
        start_time = time.time()
        results = self.db_connector.search_vectors([query_vector], top_k=top_k)
        end_time = time.time()
        
        latency = end_time - start_time
        self.metrics_collector.record_query_time(query_id, latency)
        
        return {
            "query_id": query_id,
            "latency": latency,
            "results": results
        }
    
    def execute_queries(self, 
                       num_queries: int, 
                       top_k: int = 10,
                       delay: float = 0.0) -> List[Dict[str, Any]]:
        """
        Execute multiple queries sequentially.
        
        Args:
            num_queries: Number of queries to execute
            top_k: Number of results to return per query
            delay: Delay between queries in seconds
            
        Returns:
            List of query results
        """
        results = []
        
        for i in range(num_queries):
            query_vector = self.vector_generator.generate_vector()
            result = self.execute_single_query(i, query_vector, top_k)
            results.append(result)
            
            if delay > 0 and i < num_queries - 1:
                time.sleep(delay)
        
        return results
    
    def _worker_process(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Worker process function for parallel queries."""
        process_id = args.get("process_id", 0)
        num_queries = args.get("num_queries", 10)
        top_k = args.get("top_k", 10)
        delay = args.get("delay", 0.0)
        seed = args.get("seed", None)
        
        # Set seed for this process
        if seed is not None:
            random.seed(seed + process_id)
            np.random.seed(seed + process_id)
        
        # Connect to database
        self.db_connector.connect()
        
        results = []
        for i in range(num_queries):
            query_id = process_id * num_queries + i
            query_vector = self.vector_generator.generate_vector()
            result = self.execute_single_query(query_id, query_vector, top_k)
            results.append(result)
            
            if delay > 0 and i < num_queries - 1:
                time.sleep(delay)
        
        return results
    
    def execute_parallel_queries(self, 
                               total_queries: int,
                               processes: int,
                               top_k: int = 10,
                               delay: float = 0.0,
                               seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Execute queries using multiple processes.
        
        Args:
            total_queries: Total number of queries to execute
            processes: Number of processes to use
            top_k: Number of results to return per query
            delay: Delay between queries in seconds within each process
            seed: Random seed for reproducibility
            
        Returns:
            List of query results
        """
        queries_per_process = total_queries // processes
        remainder = total_queries % processes
        
        # Prepare arguments for each process
        process_args = []
        for i in range(processes):
            # Add remainder to the first process
            num_queries = queries_per_process + (remainder if i == 0 else 0)
            
            args = {
                "process_id": i,
                "num_queries": num_queries,
                "top_k": top_k,
                "delay": delay,
                "seed": seed
            }
            process_args.append(args)
        
        # Execute queries in parallel
        with mp.Pool(processes) as pool:
            results_list = pool.map(self._worker_process, process_args)
        
        # Flatten results
        all_results = [result for sublist in results_list for result in sublist]
        
        return all_results