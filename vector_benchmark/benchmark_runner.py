from typing import Dict, Any, List, Optional, Type, Union
import time
import json
from pathlib import Path
import multiprocessing as mp

from .database_connector import DatabaseConnector
from .vector_generator import VectorGenerator
from .metrics_collector import MetricsCollector
from .query_workload import QueryWorkload

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

    def run_insertion_benchmark(self) -> Dict[str, Any]:
        """
        Run the insertion phase of the benchmark.

        Returns:
            Insertion metrics
        """
        print(f"Starting insertion benchmark: {self.num_vectors} vectors")
        start_time = time.time()

        # Connect to database
        self.db_connector.connect()

        # Create collection
        self.db_connector.create_collection()

        # Generate and insert vectors
        if self.use_intermediate_files:
            # Generate vectors and save to files
            print(f"Generating vectors and saving to intermediate files...")
            file_paths = self.vector_generator.generate_vectors_parallel(
                total_count=self.num_vectors,
                processes=self.insertion_processes,
                            save_dir=self.intermediate_dir,
                            batch_size=1000
                        )

            # Insert vectors from files
            print(f"Inserting vectors from intermediate files...")
            for i, file_path in enumerate(file_paths):
                print(f"Inserting batch {i+1}/{len(file_paths)}")
                vectors = self.vector_generator.load_vectors(file_path)

                batch_start = time.time()
                success = self.db_connector.insert_vectors(vectors)
                batch_end = time.time()

                batch_metrics = {
                    "batch_id": i,
                    "count": len(vectors),
                    "duration": batch_end - batch_start,
                    "success": success
                }
                self.metrics_collector.record_insertion_batch(batch_metrics)
        else:
            # Generate and insert vectors directly
            batch_size = 1000
            for i in range(0, self.num_vectors, batch_size):
                actual_batch_size = min(batch_size, self.num_vectors - i)
                print(f"Generating and inserting batch {i//batch_size + 1}/{(self.num_vectors + batch_size - 1)//batch_size}")

                # Generate batch
                vectors = [(i + j, self.vector_generator.generate_vector())
                          for j in range(actual_batch_size)]

                # Insert batch
                batch_start = time.time()
                success = self.db_connector.insert_vectors(vectors)
                batch_end = time.time()

                batch_metrics = {
                    "batch_id": i // batch_size,
                    "count": actual_batch_size,
                    "duration": batch_end - batch_start,
                    "success": success
                }
                self.metrics_collector.record_insertion_batch(batch_metrics)

        end_time = time.time()
        total_duration = end_time - start_time

        # Calculate insertion metrics
        insertion_stats = self.metrics_collector.calculate_insertion_statistics()
        insertion_stats["total_duration"] = total_duration

        print(f"Insertion benchmark completed in {total_duration:.2f} seconds")
        print(f"Insertion rate: {insertion_stats.get('insertion_rate_vectors_per_sec', 0):.2f} vectors/sec")

        return insertion_stats

    def run_query_benchmark(self) -> Dict[str, Any]:
        """
        Run the query phase of the benchmark.

        Returns:
            Query metrics
        """
        print(f"Starting query benchmark: {self.num_queries} queries")
        start_time = time.time()

        # Connect to database if not already connected
        if not self.db_connector.is_connected():
            self.db_connector.connect()

        # Execute queries
        if self.query_processes > 1:
            print(f"Running queries with {self.query_processes} parallel processes")
            self.query_workload.execute_parallel_queries(
                total_queries=self.num_queries,
                processes=self.query_processes,
                top_k=self.top_k,
                delay=self.query_delay,
                seed=self.seed
            )
        else:
            print("Running queries sequentially")
            self.query_workload.execute_queries(
                num_queries=self.num_queries,
                top_k=self.top_k,
                delay=self.query_delay
            )

        end_time = time.time()
        total_duration = end_time - start_time

        # Calculate query metrics
        query_stats = self.metrics_collector.calculate_query_statistics()
        query_stats["total_duration"] = total_duration

        print(f"Query benchmark completed in {total_duration:.2f} seconds")
        print(f"Average latency: {query_stats.get('mean_latency_ms', 0):.2f} ms")
        print(f"Throughput: {query_stats.get('throughput_qps', 0):.2f} queries/sec")

        return query_stats

    def run_benchmark(self) -> Dict[str, Any]:
        """
        Run the complete benchmark (insertion and query).

        Returns:
            Complete benchmark results
        """
        print(f"Starting benchmark: {self.benchmark_name}")
        print(f"Database: {self.db_connector.__class__.__name__}")
        print(f"Vector dimension: {self.vector_dim}")
        print(f"Number of vectors: {self.num_vectors}")
        print(f"Number of queries: {self.num_queries}")

        # Run insertion benchmark
        insertion_stats = self.run_insertion_benchmark()

        # Run query benchmark
        query_stats = self.run_query_benchmark()

        # Generate visualizations
        latency_histogram = self.metrics_collector.generate_latency_histogram()
        throughput_timeline = self.metrics_collector.generate_throughput_timeline()

        # Compile results
        results = {
            "benchmark_name": self.benchmark_name,
            "database": self.db_connector.__class__.__name__,
            "config": self.config,
            "insertion_stats": insertion_stats,
            "query_stats": query_stats,
            "visualizations": {
                "latency_histogram": latency_histogram,
                "throughput_timeline": throughput_timeline
            },
            "timestamp": time.time()
        }

        # Save results
        output_path = Path(self.output_dir) / f"{self.benchmark_name}_{int(time.time())}.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Benchmark results saved to {output_path}")

        # Disconnect from database
        self.db_connector.disconnect()

        return results