import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

from .benchmark_runner import BenchmarkRunner
from .connectors.milvus_connector import MilvusConnector
from .connectors.qdrant_connector import QdrantConnector
from .connectors.pinecone_connector import PineconeConnector
from .database_connector import DatabaseConnector

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Vector Database Benchmark Tool")
    
    # General benchmark configuration
    parser.add_argument("--config", type=str, help="Path to benchmark configuration file")
    parser.add_argument("--output-dir", type=str, default="benchmark_results", 
                        help="Directory to store benchmark results")
    parser.add_argument("--benchmark-name", type=str, default="vector_benchmark",
                        help="Name of the benchmark run")
    
    # Vector configuration
    parser.add_argument("--vector-dim", type=int, default=128,
                        help="Dimensionality of vectors")
    parser.add_argument("--data-type", type=str, default="float32",
                        choices=["float32", "float64"],
                        help="Data type of vectors")
    
    # Benchmark parameters
    parser.add_argument("--num-vectors", type=int, default=10000,
                        help="Number of vectors to insert")
    parser.add_argument("--num-queries", type=int, default=1000,
                        help="Number of queries to execute")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of results to return per query")
    parser.add_argument("--query-delay", type=float, default=0.0,
                        help="Delay between queries in seconds")
    parser.add_argument("--query-processes", type=int, default=1,
                        help="Number of processes for parallel queries")
    parser.add_argument("--insertion-processes", type=int, default=1,
                        help="Number of processes for parallel insertion")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # Database selection
    parser.add_argument("--database", type=str, required=True,
                        choices=["milvus", "qdrant", "pinecone"],
                        help="Vector database to benchmark")
    
    # Milvus configuration
    parser.add_argument("--milvus-host", type=str, default="localhost",
                        help="Milvus server host")
    parser.add_argument("--milvus-port", type=str, default="19530",
                        help="Milvus server port")
    
    # Qdrant configuration
    parser.add_argument("--qdrant-host", type=str, default="localhost",
                        help="Qdrant server host")
    parser.add_argument("--qdrant-port", type=int, default=6333,
                        help="Qdrant server port")
    
    # Pinecone configuration
    parser.add_argument("--pinecone-api-key", type=str,
                        help="Pinecone API key")
    parser.add_argument("--pinecone-environment", type=str,
                        help="Pinecone environment")
    
    # Collection/index name
    parser.add_argument("--collection-name", type=str, default="benchmark_collection",
                        help="Name of the collection/index to use")
    
    # Metric type
    parser.add_argument("--metric-type", type=str, default="L2",
                        choices=["L2", "IP", "COSINE"],
                        help="Distance metric type")
    
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def create_connector(args) -> Optional[DatabaseConnector]:
    """Create a database connector based on command line arguments."""
    if args.database == "milvus":
        return MilvusConnector(
            host=args.milvus_host,
            port=args.milvus_port,
            collection_name=args.collection_name,
            vector_dim=args.vector_dim,
            metric_type=args.metric_type
        )
    elif args.database == "qdrant":
        return QdrantConnector(
            host=args.qdrant_host,
            port=args.qdrant_port,
            collection_name=args.collection_name,
            vector_dim=args.vector_dim,
            metric_type=args.metric_type
        )
    elif args.database == "pinecone":
        if not args.pinecone_api_key or not args.pinecone_environment:
            print("Error: Pinecone API key and environment are required")
            return None
        
        return PineconeConnector(
            api_key=args.pinecone_api_key,
            environment=args.pinecone_environment,
            index_name=args.collection_name,
            vector_dim=args.vector_dim,
            metric_type=args.metric_type
        )
    else:
        print(f"Error: Unsupported database '{args.database}'")
        return None

def main():
    """Main entry point for the benchmark CLI."""
    args = parse_args()
    
    # Load configuration from file if provided
    config = {}
    if args.config:
        config = load_config(args.config)
        
        # Override config with command line arguments
        for key, value in vars(args).items():
            if value is not None and key != "config":
                config[key] = value
    else:
        config = vars(args)
    
    # Create output directory
    output_dir = Path(config.get("output_dir", "benchmark_results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create database connector
    db_connector = create_connector(args)
    if not db_connector:
        return
    
    # Create and run benchmark
    benchmark = BenchmarkRunner(
        db_connector=db_connector,
        benchmark_name=config.get("benchmark_name", "vector_benchmark"),
        vector_dim=config.get("vector_dim", 128),
        num_vectors=config.get("num_vectors", 10000),
        num_queries=config.get("num_queries", 1000),
        top_k=config.get("top_k", 10),
        query_delay=config.get("query_delay", 0.0),
        query_processes=config.get("query_processes", 1),
        insertion_processes=config.get("insertion_processes", 1),
        seed=config.get("seed", 42),
        output_dir=str(output_dir),
        config=config
    )
    
    # Run benchmark
    results = benchmark.run_benchmark()
    
    # Print summary
    print("\nBenchmark Summary:")
    print(f"Database: {config.get('database')}")
    print(f"Vectors: {config.get('num_vectors')}, Dimension: {config.get('vector_dim')}")
    print(f"Queries: {config.get('num_queries')}, Top-K: {config.get('top_k')}")
    
    print("\nInsertion Performance:")
    print(f"Total time: {results['insertion_stats']['total_duration']:.2f} seconds")
    print(f"Insertion rate: {results['insertion_stats'].get('insertion_rate_vectors_per_sec', 0):.2f} vectors/sec")
    
    print("\nQuery Performance:")
    print(f"Total time: {results['query_stats']['total_duration']:.2f} seconds")
    print(f"Mean latency: {results['query_stats'].get('mean_latency_ms', 0):.2f} ms")
    print(f"P95 latency: {results['query_stats'].get('p95_latency_ms', 0):.2f} ms")
    print(f"P99 latency: {results['query_stats'].get('p99_latency_ms', 0):.2f} ms")
    print(f"Throughput: {results['query_stats'].get('throughput_qps', 0):.2f} queries/sec")
    
    print(f"\nDetailed results saved to: {output_dir}")

if __name__ == "__main__":
    main()