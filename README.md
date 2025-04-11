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
Use the load_vdb.py script to generate and load vectors into your vector database:
`python -m vdbbench.load_vdb --config configs/10m.yaml`

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
The compact_and_watch.py script monitors the database and performs compaction:
```bash
python -m vdbbench.compact_and_watch --config configs/10m.yaml --interval 5
```
This step is automatically performed at the end of the loading process if you set compact: true in your configuration.

### Step 3: Run the Benchmark
Finally, run the benchmark using the simple_bench.py script:
```bash
python -m vdbbench.simple_bench --config configs/10m.yaml --processes <N> --batch-size <batch_size>
```

## Supported Databases
Milvus (currently implemented)

# Contributing
Contributions are welcome! Please feel free to submit a Pull Request.