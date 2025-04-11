import numpy as np
from typing import List, Dict, Any, Optional
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

class MetricsCollector:
    """
    Collects and analyzes performance metrics for vector database benchmarks.
    """
    
    def __init__(self, output_dir: str = "benchmark_results"):
        """
        Initialize the metrics collector.
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.query_times = []
        self.insertion_metrics = {}
        
    def record_query_time(self, query_id: int, latency: float) -> None:
        """
        Record the latency of a single query.
        
        Args:
            query_id: ID of the query
            latency: Query latency in seconds
        """
        self.query_times.append({
            "query_id": query_id,
            "latency": latency,
            "timestamp": time.time()
        })
    
    def record_insertion_batch(self, batch_metrics: Dict[str, Any]) -> None:
        """
        Record metrics for an insertion batch.
        
        Args:
            batch_metrics: Dictionary containing batch metrics
        """
        batch_id = batch_metrics.get("batch_id", len(self.insertion_metrics))
        self.insertion_metrics[batch_id] = batch_metrics
    
    def calculate_query_statistics(self) -> Dict[str, float]:
        """
        Calculate statistics for query latencies.
        
        Returns:
            Dictionary of statistics
        """
        if not self.query_times:
            return {}
        
        latencies = [q["latency"] * 1000 for q in self.query_times]  # Convert to ms
        
        # Calculate statistics
        stats = {
            "count": len(latencies),
            "min_latency_ms": np.min(latencies),
            "max_latency_ms": np.max(latencies),
            "mean_latency_ms": np.mean(latencies),
            "median_latency_ms": np.median(latencies),
            "p95_latency_ms": np.percentile(latencies, 95),
            "p99_latency_ms": np.percentile(latencies, 99),
            "p999_latency_ms": np.percentile(latencies, 99.9),
            "throughput_qps": len(latencies) / (sum(latencies) / 1000)
        }
        
        return stats
    
    def calculate_insertion_statistics(self) -> Dict[str, float]:
        """
        Calculate statistics for insertion operations.
        
        Returns:
            Dictionary of statistics
        """
        if not self.insertion_metrics:
            return {}
        
        total_vectors = sum(batch.get("count", 0) for batch in self.insertion_metrics.values())
        total_time = sum(batch.get("time", 0) for batch in self.insertion_metrics.values())
        
        stats = {
            "total_vectors": total_vectors,
            "total_batches": len(self.insertion_metrics),
            "total_time_seconds": total_time,
            "vectors_per_second": total_vectors / total_time if total_time > 0 else 0
        }
        
        return stats
    
    def save_results(self, benchmark_name: str, db_type: str, 
                    config: Dict[str, Any]) -> str:
        """
        Save benchmark results to file.
        
        Args:
            benchmark_name: Name of the benchmark
            db_type: Type of database used
            config: Benchmark configuration
            
        Returns:
            Path to the saved results file
        """
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{benchmark_name}_{db_type}_{timestamp}.json"
        file_path = self.output_dir / filename
        
        # Compile results
        results = {
            "benchmark_name": benchmark_name,
            "database_type": db_type,
            "timestamp": timestamp,
            "configuration": config,
            "query_statistics": self.calculate_query_statistics(),
            "insertion_statistics": self.calculate_insertion_statistics(),
            "raw_query_times": self.query_times,
            "raw_insertion_metrics": self.insertion_metrics
        }
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return str(file_path)
    
    def generate_latency_histogram(self, output_file: Optional[str] = None) -> str:
        """
        Generate a histogram of query latencies.
        
        Args:
            output_file: Path to save the histogram image
            
        Returns:
            Path to the saved image
        """
        if not self.query_times:
            return ""
        
        latencies = [q["latency"] * 1000 for q in self.query_times]  # Convert to ms
        
        plt.figure(figsize=(10, 6))
        plt.hist(latencies, bins=50, alpha=0.7)
        plt.xlabel('Latency (ms)')
        plt.ylabel('Frequency')
        plt.title('Query Latency Distribution')
        plt.grid(True, alpha=0.3)

        # Add percentile lines
        percentiles = [50, 95, 99, 99.9]
        colors = ['green', 'orange', 'red', 'purple']

        for p, color in zip(percentiles, colors):
            percentile_value = np.percentile(latencies, p)
            plt.axvline(x=percentile_value, color=color, linestyle='--',
                       label=f'P{p}: {percentile_value:.2f} ms')

        plt.legend()

        # Save or show
        if output_file is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_file = str(self.output_dir / f"latency_histogram_{timestamp}.png")

        plt.savefig(output_file)
        plt.close()

        return output_file

    def generate_throughput_timeline(self, window_size: int = 10,
                                   output_file: Optional[str] = None) -> str:
        """
        Generate a timeline of query throughput.

        Args:
            window_size: Size of the sliding window in seconds
            output_file: Path to save the timeline image

        Returns:
            Path to the saved image
        """
        if not self.query_times:
            return ""

        # Convert to DataFrame for easier time-based analysis
        df = pd.DataFrame(self.query_times)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.set_index('timestamp')

        # Calculate throughput in sliding windows
        throughput = []
        timestamps = []

        start_time = df.index.min()
        end_time = df.index.max()

        current_time = start_time
        while current_time <= end_time:
            window_end = current_time + pd.Timedelta(seconds=window_size)
            window_queries = df[(df.index >= current_time) & (df.index < window_end)]

            if not window_queries.empty:
                query_count = len(window_queries)
                throughput.append(query_count / window_size)
                timestamps.append(current_time)

            current_time += pd.Timedelta(seconds=window_size / 2)  # 50% overlap

        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, throughput, marker='o', markersize=3)
        plt.xlabel('Time')
        plt.ylabel('Throughput (queries/second)')
        plt.title('Query Throughput Over Time')
        plt.grid(True, alpha=0.3)

        # Format x-axis
        plt.gcf().autofmt_xdate()

        # Save or show
        if output_file is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_file = str(self.output_dir / f"throughput_timeline_{timestamp}.png")

        plt.savefig(output_file)
        plt.close()

        return output_file