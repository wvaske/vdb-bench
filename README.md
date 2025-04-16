# Vector Database Benchmark Tool
This tool allows you to benchmark and compare the performance of vector databases, with a focus on Milvus, Qdrant, and Pinecone.

## Installation

### Using Docker (recommended)
1. Clone the repository:
``` bash
git clone https://github.com/yourusername/vdb-bench.git
cd vdb-bench
```
2. Build and run the Docker container:
```bash
docker-compose up -d
```

### Manual Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/vdb-bench.git
cd vdb-bench
```

2. Install the package:
```bash
pip3 install ./
```

## Running the Benchmark
The benchmark process consists of three main steps:
1. Loading vectors into the database
2. Monitoring and compacting the database
3. Running the benchmark queries

### Step 1: Load Vectors into the Database
Use the load_vdb.py script to generate and load 10 million vectors into your vector database: (this process can take up to 8 hours)
```bash
python load_vdb.py --config configs/10m.yaml
```


For testing, I recommend using a smaller data by passing the num_vectors option:
```bash
python load_vdb.py --config configs/10m.yaml --collection_name mlps_500k_10shards_1536dim_uniform --num_vectors 500000
```

Key parameters:
* --collection-name: Name of the collection to create
* --dimension: Vector dimension
* --num-vectors: Number of vectors to generate
* --chunk-size: Number of vectors to generate in each chunk (for memory management)
* --distribution: Distribution for vector generation (uniform, normal)
* --batch-size: Batch size for insertion

Example configuration file (configs/10m.yaml):
```yaml
database:
  host: 127.0.0.1
  port: 19530
  database: milvus
  max_receive_message_length: 514_983_574
  max_send_message_length: 514_983_574

dataset:
  collection_name: mlps_10m_10shards_1536dim_uniform
  num_vectors: 10_000_000
  dimension: 1536
  distribution: uniform
  batch_size: 1000
  num_shards: 10
  vector_dtype: FLOAT_VECTOR

index:
  index_type: DISKANN
  metric_type: COSINE
  index_params:
    M: 64
    ef_construction: 200

workflow:
  compact: True
```

### Step 2: Monitor and Compact the Database
The compact_and_watch.py script monitors the database and performs compaction. You should only need this if the load process exits out while waiting. The load script will do compaction and will wait for it to complete.
```bash
python compact_and_watch.py --config configs/10m.yaml --interval 5
```
This step is automatically performed at the end of the loading process if you set compact: true in your configuration.

### Step 3: Run the Benchmark
Finally, run the benchmark using the simple_bench.py script:
```bash
python simple_bench.py --host 127.0.0.1 --collection <collection_name> --processes <N> --batch-size <batch_size> --runtime <length of benchmark run in seconds>
```

## Supported Databases
Milvus (currently implemented)

# Contributing
Contributions are welcome! Please feel free to submit a Pull Request.