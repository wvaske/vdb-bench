import psycopg2
import numpy as np
from psycopg2.extensions import register_adapter, AsIs
from time import time
from itertools import islice

# Configuration settings
TOTAL_INSERT = 10_000_000         # Total initial insert data volume
UPDATE_SAMPLE = 5_000_000         # Number of data to update in Scenario 1
DELETE_INSERT_SAMPLE = 5_000_000  # Number of delete and reinsert in Scenario 2
BATCH_SIZE = 10000                # Batch size (adjust according to device)
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'testdb',
    'user': 'postgres',
    'password': 'postgres'
}

def register_vector_adapter():
    """Register numpy array/vector adapters"""
    def adapt_array_to_vector(vec):
        # Use square brackets instead of curly braces
        return AsIs(f"'[{','.join(map(repr, vec))}]'")
    # Register for both numpy arrays and regular lists
    register_adapter(np.ndarray, adapt_array_to_vector)
    register_adapter(list, adapt_array_to_vector)

def create_table_and_index(conn):
    """Create vector table and DISKANN index"""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("DROP TABLE IF EXISTS vectors CASCADE;")
        cur.execute(
            "CREATE TABLE vectors (id SERIAL PRIMARY KEY, embedding vector(128));"
        )
        cur.execute(
            "CREATE INDEX vectors_index ON vectors "
            "USING diskann (embedding vector_cosine_ops) WITH (lists = 1000);"
        )
        conn.commit()

def chunked_batches(iterable, size):
    """Yield items in chunks (memory optimization)"""
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            return
        yield chunk

def generate_random_vector():
    """Generates random 1536-dimensional vector"""
    return np.random.rand(1536).tolist()

def insert_update_test(conn):
    """Scenario 1: Insert 10M then update 5M entries"""
    with conn.cursor() as cur:
        print("++ Scenario 1: Inserting initial data ++")
        start = time()
        # Insert initial data in batches
        for i in range(0, TOTAL_INSERT, BATCH_SIZE):
            vec_batch = [
                generate_random_vector()
                for _ in range(min(BATCH_SIZE, TOTAL_INSERT - i))
            ]
            cur.executemany(
                "INSERT INTO vectors (embedding) VALUES (%s)",
                [(vec,) for vec in vec_batch]
            )
            conn.commit()
        insert_time = time() - start
        print(f"Initial insert took: {insert_time:.2f}s")
        # Prepare random update IDs (select 5M from existing data)
        cur.execute("SELECT id FROM vectors")
        all_ids = [row[0] for row in cur.fetchall()]
        update_ids = np.random.choice(all_ids, UPDATE_SAMPLE, replace=False).tolist()
        print(f"++ Starting 5M updates ++")
        start_update = time()
        for batch_ids in chunked_batches(update_ids, BATCH_SIZE):
            new_vectors = [generate_random_vector() for _ in batch_ids]
            params = [(vec, vec_id) for vec, vec_id in zip(new_vectors, batch_ids)]
            cur.executemany(
                "UPDATE vectors SET embedding = %s WHERE id = %s",
                params
            )
            conn.commit()
        update_time = time() - start_update
        print(f"Update completed: {update_time:.2f}s. Total time: {(update_time + insert_time)/60:.2f} minutes")
    return {
        'scenario1_total': insert_time + update_time
    }

def insert_delete_reinsert_test(conn):
    """Scenario 2: Insert 10M → Delete 5M → Reinsert 5M"""
    with conn.cursor() as cur:
        print("++ Scenario 2: Inserting initial data ++")
        start = time()
        # Insert initial data
        for i in range(0, TOTAL_INSERT, BATCH_SIZE):
            vec_batch = [
                generate_random_vector()
                for _ in range(min(BATCH_SIZE, TOTAL_INSERT - i))
            ]
            cur.executemany(
                "INSERT INTO vectors (embedding) VALUES (%s)",
                [(vec,) for vec in vec_batch]
            )
            conn.commit()
        insert_time = time() - start
        print(f"Initial insert took: {insert_time:.2f}s")
        # Prepare IDs to delete (select 5M)
        cur.execute("SELECT id FROM vectors")
        all_ids = [row[0] for row in cur.fetchall()]
        delete_ids = np.random.choice(all_ids, DELETE_INSERT_SAMPLE, replace=False).tolist()
        # Batch delete (increasing batch size for large deletions)
        print("++ Deleting 5M entries ++")
        del_start = time()
        for batch in chunked_batches(delete_ids, 50000):  
            batch_str = ",".join(map(str, batch))
            cur.execute(f"DELETE FROM vectors WHERE id IN ({batch_str})")
            conn.commit()
        delete_time = time() - del_start
        print(f"Delete completed: {delete_time:.2f}s")
        # Reinsert 5M new entries
        print("++ Reinserting 5M entries ++")
        reinsert_start = time()
        for _ in range(DELETE_INSERT_SAMPLE // BATCH_SIZE):
            vec_batch = [generate_random_vector() for _ in range(BATCH_SIZE)]
            cur.executemany(
                "INSERT INTO vectors (embedding) VALUES (%s)",
                [(vec,) for vec in vec_batch]
            )
            conn.commit()
        # Process the remainder
        remainder = DELETE_INSERT_SAMPLE % BATCH_SIZE
        if remainder:
            vec_batch = [generate_random_vector() for _ in range(remainder)]
            cur.executemany(
                "INSERT INTO vectors (embedding) VALUES (%s)",
                [(vec,) for vec in vec_batch]
            )
            conn.commit()
        reinsert_time = time() - reinsert_start
        print(f"Reinsert completed: {reinsert_time:.2f}s. Total time: {(delete_time + reinsert_time + insert_time)/60:.2f} minutes")
    return {
        'scenario2_total': insert_time + delete_time + reinsert_time
    }

def main():
    # Register vector adapters
    register_vector_adapter()
    results = {}
    print("\n=== Scenario 1 begins ===")
    with psycopg2.connect(**DB_CONFIG) as conn:
        create_table_and_index(conn)
        results['scenario1'] = insert_update_test(conn)
    
    print("\n=== Scenario 2 begins ===")
    with psycopg2.connect(**DB_CONFIG) as conn:
        create_table_and_index(conn)
        results['scenario2'] = insert_delete_reinsert_test(conn)
    
    print("\nPerformance comparison:")
    print(f"Scenario 1 total time: {results['scenario1']['scenario1_total']/60:.2f} minutes")
    print(f"Scenario 2 total time: {results['scenario2']['scenario2_total']/60:.2f} minutes")
    print(f"Scenario 2 is {(results['scenario1']['scenario1_total']/results['scenario2']['scenario2_total']*100):.1f}% faster than Scenario 1")

if __name__ == "__main__":
    main()
