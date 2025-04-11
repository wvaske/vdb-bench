from .query_workload import QueryWorkload
from .database_connector import DatabaseConnector
from .vector_generator import VectorGenerator
from .metrics_collector import MetricsCollector
from typing import Dict, Any, List, Optional, Type, Union
import time
import os
import logging

class BenchmarkRunner:
    """
    Main class for running vector database benchmarks.
    """
    
    def __init__(self, 
                 db_connector: DatabaseConnector,
                 config: Dict[str, Any]):
        """
        Initialize the benchmark runner.
        
        Args:
            db_connector: Database connector to use
            config: Benchmark configuration
        """
        self.db_connector = db_connector
        self.config = config
        
        # Extract configuration
        self.vector_dim = config.get("vector_dim", 128)
        self.data_type = config.get("data_type", "float32")
        self.num_vectors = config.get("num_vectors", 10000)
        self.num_queries = config.get("num_queries", 1000)
        self.top_k = config.get("top_k", 10)
        self.query_delay = config.get("query_delay", 0.0)
        self.query_processes = config.get("query_processes", 1)
        self.insertion_processes = config.get("insertion_processes", 1)
        self.seed = config.get("seed", 42)
        self.output_dir = config.get("output_dir", "benchmark_results")
        self.benchmark_name = config.get("benchmark_name", "vector_benchmark")
        self.use_intermediate_files = config.get("use_intermediate_files", False)
        self.intermediate_dir = config.get("intermediate_dir", "intermediate_data")
        
        # Initialize components
        self.vector_generator = VectorGenerator(
            dimensions=self.vector_dim,
            data_type=self.data_type,
            seed=self.seed
        )
        
        self.metrics_collector = MetricsCollector(
            output_dir=self.output_dir
        )
        
        self.query_workload = QueryWorkload(
            db_connector=self.db_connector,
            vector_generator=self.vector_generator,
            metrics_collector=self.metrics_collector
        )
    
    def run_benchmark(self) -> Dict[str, Any]:
        """Run the complete benchmark."""
        # Create collection/index if it doesn't exist
        self.db_connector.create_collection()
        
        # Run insertion benchmark
        insertion_results = self.run_insertion_benchmark()
        
        # Run query benchmark
        query_results = self.run_query_benchmark()
        
        # Combine results
        results = {
            "insertion_stats": insertion_results,
            "query_stats": query_results,
            "config": self.config
        }
        
        # Save results
        self.metrics_collector.save_benchmark_results(
            results, f"{self.benchmark_name}_results.json"
        )
        
        return results
    
    def run_insertion_benchmark(self) -> Dict[str, Any]:
        """Run the insertion benchmark."""
        print(f"Running insertion benchmark with {self.num_vectors} vectors...")
        
        start_time = time.time()
        
        if self.use_intermediate_files:
            # Generate vectors to intermediate files
            os.makedirs(self.intermediate_dir, exist_ok=True)
            file_paths = self.vector_generator.generate_vector_files(
                num_vectors=self.num_vectors,
                output_dir=self.intermediate_dir,
                batch_size=10000
            )
            
            print(f"Inserting vectors from intermediate files...")
            self.db_connector.load_and_insert_from_files(file_paths)
        else:
            # Generate and insert vectors directly
            vectors = self.vector_generator.generate_vectors(self.num_vectors)
            self.db_connector.insert_vectors(vectors)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Calculate metrics
        results = {
            "total_duration": duration,
            "num_vectors": self.num_vectors,
            "insertion_rate_vectors_per_sec": self.num_vectors / duration if duration > 0 else 0,
            "timestamp": time.time()
        }
        
        print(f"Insertion completed in {duration:.2f} seconds")
        print(f"Insertion rate: {results['insertion_rate_vectors_per_sec']:.2f} vectors/sec")
        
        # Save metrics
        self.metrics_collector.save_insertion_metrics(results)
        
        return results
    
    def run_query_benchmark(self) -> Dict[str, Any]:
        """Run the query benchmark."""
        print(f"Running query benchmark with {self.num_queries} queries...")
        
        # Generate query vectors
        query_vectors = self.vector_generator.generate_vectors(self.num_queries)
        
        # Run queries
        results = self.query_workload.run_queries(
            query_vectors=query_vectors,
            top_k=self.top_k,
            processes=self.query_processes,
            delay=self.query_delay
        )
        
        print(f"Query benchmark completed")
        print(f"Mean latency: {results.get('mean_latency_ms', 0):.2f} ms")
        print(f"P95 latency: {results.get('p95_latency_ms', 0):.2f} ms")
        print(f"Throughput: {results.get('throughput_qps', 0):.2f} queries/sec")
        
        # Save metrics
        self.metrics_collector.save_query_metrics(results)
        
        return results