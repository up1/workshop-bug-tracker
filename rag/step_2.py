# Steps
# 1. Read data from  file `step_1_results.txt`
# 2. Embedding data with OpenAI API
# 3. Save to Vector database with Milvus server via APIs

import os
import json
from typing import List, Dict
from openai import OpenAI
from pymilvus import Collection, connections, FieldSchema, CollectionSchema, DataType
    
from pymilvus import MilvusClient

def read_data(file_path: str) -> List[Dict]:
    # read json lines from a file
    items = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                print(f"Processing line: {line.strip()}")
                item = json.loads(line.strip())
                items.append(item)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    
    return items

def embed_data(data: List[Dict]) -> List[Dict]:
    client = OpenAI()
    embedded_data = []
    for item in data:
        response = client.embeddings.create(
            input=item['description'],
            model="text-embedding-3-small"
        )
        item['embedding'] = response.data[0].embedding
        embedded_data.append(item)

    return embedded_data

def save_to_milvus(embedded_data: List[Dict], milvus_host: str, milvus_port: int):
    client = MilvusClient(
        uri="http://" + milvus_host + ":" + str(milvus_port)
    )

    # Drop collection if it exists
    if client.has_collection("bug_tracking"):
        client.drop_collection(collection_name="bug_tracking")

    # Define schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=len(embedded_data[0]['embedding']))
    ]
    
    schema = CollectionSchema(fields)

    # Prepare index parameters
    index_params = client.prepare_index_params()

    # Add indexes
    index_params.add_index(
        field_name="embedding",
        index_type="IVF_FLAT",
        metric_type="COSINE"
    )

    # Create collection
    client.create_collection(
        collection_name="bug_tracking",
        schema=schema,
        index_params=index_params
    )

    # Insert data into collection
    datas = [
        {"id": item['id'], "description": item['description'], "embedding": item['embedding']}
        for item in embedded_data
    ]
    client.insert(
        collection_name="bug_tracking",
        data=datas
    )

if __name__ == "__main__":
    # Read data from step_1_results.txt
    file_path = 'step_1_results.txt'
    data = read_data(file_path)
    
    # Embed data using OpenAI API
    embedded_data = embed_data(data)
    print(f"Embedded {len(embedded_data)} items.")

    # Save embedded data to Milvus
    milvus_host = os.getenv('MILVUS_HOST', 'localhost')
    milvus_port = int(os.getenv('MILVUS_PORT', 19530))
    save_to_milvus(embedded_data, milvus_host, milvus_port)
    print(f"Saved {len(embedded_data)} items to Milvus at {milvus_host}:{milvus_port}.")