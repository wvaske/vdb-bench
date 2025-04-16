#!/usr/bin/env python3
"""
Milvus Collection Lister

This script connects to a local Milvus database and lists all collections
along with the number of vectors in each collection.
"""

import argparse
import sys
from typing import Dict, List, Tuple

try:
    from pymilvus import connections, utility
    from pymilvus.exceptions import MilvusException
except ImportError:
    print("Error: pymilvus package not found. Please install it with 'pip install pymilvus'")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="List Milvus collections and their vector counts")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Milvus server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=str, default="19530",
                        help="Milvus server port (default: 19530)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed collection information")
    return parser.parse_args()


def connect_to_milvus(host: str, port: str) -> bool:
    """Establish connection to Milvus server"""
    try:
        connections.connect(
            alias="default",
            host=host,
            port=port,
            max_receive_message_length=514983574,
            max_send_message_length=514983574
        )
        return True
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        return False


def get_collections_info() -> List[Dict]:
    """Get information about all collections"""
    try:
        collection_names = utility.list_collections()
        collections_info = []

        for name in collection_names:
            from pymilvus import Collection
            collection = Collection(name)

            # Get collection statistics - using num_entities instead of get_stats()
            row_count = collection.num_entities

            # Get collection schema
            schema = collection.schema
            description = schema.description if schema.description else "No description"

            # Get vector field dimension
            vector_field = None
            vector_dim = None
            for field in schema.fields:
                if field.dtype == 100:  # DataType.FLOAT_VECTOR
                    vector_field = field.name
                    vector_dim = field.params.get("dim")
                    break

            # Get index information
            index_info = []
            try:
                for field_name in collection.schema.fields:
                    if collection.has_index(field_name.name):
                        index = collection.index(field_name.name)
                        index_info.append({
                            "field": field_name.name,
                            "index_type": index.params.get("index_type"),
                            "metric_type": index.params.get("metric_type"),
                            "params": index.params.get("params", {})
                        })
            except Exception as e:
                index_info = [{"error": str(e)}]

            collections_info.append({
                "name": name,
                "row_count": row_count,
                "description": description,
                "vector_field": vector_field,
                "vector_dim": vector_dim,
                "index_info": index_info
            })

        return collections_info
    except MilvusException as e:
        print(f"Error retrieving collection information: {e}")
        return []


def main() -> int:
    """Main function"""
    args = parse_args()

    # Connect to Milvus
    if not connect_to_milvus(args.host, args.port):
        return 1

    print(f"Connected to Milvus server at {args.host}:{args.port}")

    # Get collections information
    collections_info = get_collections_info()

    if not collections_info:
        print("No collections found.")
        return 0

    # Display collections information
    print(f"\nFound {len(collections_info)} collections:")
    print("-" * 80)

    for info in collections_info:
        print(f"Collection: {info['name']}")
        print(f"  Vectors:      {info['row_count']:,}")
        print(f"  Vector Field: {info['vector_field']} (dim: {info['vector_dim']})")

        if args.verbose:
            print(f"  Description:  {info['description']}")

            if info['index_info']:
                print("  Indexes:")
                for idx in info['index_info']:
                    if "error" in idx:
                        print(f"    Error retrieving index info: {idx['error']}")
                    else:
                        print(f"    Field: {idx['field']}")
                        print(f"      Type:   {idx['index_type']}")
                        print(f"      Metric: {idx['metric_type']}")
                        print(f"      Params: {idx['params']}")
            else:
                print("  Indexes: None")

        print("-" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())