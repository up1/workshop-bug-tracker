# Search data from Milvus
import os
import json
from typing import List, Dict
from openai import OpenAI
from pymilvus import Collection, connections


def search_data(query: str, top_k: int = 5) -> List[Dict]:
    client = OpenAI()
    response = client.embeddings.create(input=query, model="text-embedding-3-small")
    query_embedding = response.data[0].embedding

    connections.connect(
        host=os.getenv("MILVUS_HOST", "localhost"),
        port=int(os.getenv("MILVUS_PORT", 19530)),
    )
    collection = Collection(name="bug_tracking")

    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["id", "description"],
    )
    return [hit.entity for hit in results[0]]


if __name__ == "__main__":
    query = input("Enter a search query: ")
    top_k = int(input("Enter the number of results to return (default 5): ") or 5)

    results = search_data(query, top_k)

    print(f"Found {len(results)} results:")
    for item in results:
        print(item)
