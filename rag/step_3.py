# Search data from Milvus
import os
from typing import List, Dict
from openai import OpenAI
from pymilvus import Collection
from pymilvus import MilvusClient


def search_data(query: str, top_k: int = 5) -> List[Dict]:
    client = OpenAI()
    response = client.embeddings.create(input=query, model="text-embedding-3-small")
    query_embedding = response.data[0].embedding

    milvus_host = os.getenv('MILVUS_HOST', 'localhost')
    milvus_port = int(os.getenv('MILVUS_PORT', 19530))
    client = MilvusClient(
        uri="http://" + milvus_host + ":" + str(milvus_port)
    )

    search_params = {
        "params": {
            "radius": 0.4,
            "range_filter": 0.6
        }
    }

    # Search data in Milvus
    search_results = client.search(
        collection_name="bug_tracking",
        data=[query_embedding],
        anns_field="embedding",
        search_params=search_params,
        limit=top_k,
        output_fields=["id", "description"],
    )
    if not search_results or not search_results[0]:
        print("No results found.")
        return []

    return [hit.entity for hit in search_results[0]]


if __name__ == "__main__":
    query = input("Enter a search query: ")
    top_k = int(input("Enter the number of results to return (default 5): ") or 5)

    results = search_data(query, top_k)

    print(f"Found {len(results)} results:")
    for item in results:
        print(item)
