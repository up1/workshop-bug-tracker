# Search data from Milvus
import os
from typing import List, Dict
from openai import OpenAI
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker


def hybrid_search_data(query: str, top_k: int = 5) -> List[Dict]:
    client = OpenAI()
    response = client.embeddings.create(input=query, model="text-embedding-3-small")
    query_embedding = response.data[0].embedding

    milvus_host = os.getenv('MILVUS_HOST', 'localhost')
    milvus_port = int(os.getenv('MILVUS_PORT', 19530))
    client = MilvusClient(
        uri="http://" + milvus_host + ":" + str(milvus_port)
    )

    # Semantic search (embedding)
    search_param_1 = {
        "data": [query_embedding],
        "anns_field": "embedding",
        "param": {"nprobe": 10},
        "limit": top_k,
    }
    request_1 = AnnSearchRequest(**search_param_1)

    # Full-text search (sparse)
    search_param_2 = {
        "data": [query],
        "anns_field": "description_sparse",
        "param": {"drop_ratio_search": 0.2},
        "limit": top_k,
    }
    request_2 = AnnSearchRequest(**search_param_2)

    reqs = [request_1, request_2]

    # Re-rannking
    ranker = RRFRanker(100)

    # Search data in Milvus
    search_results = client.hybrid_search(
        collection_name="bug_tracking",
        reqs=reqs,
        ranker=ranker,
        limit=top_k,
        output_fields=["id", "description", "severity"]
    )

    if not search_results or not search_results[0]:
        print("No results found.")
        return []

    return [hit.entity for hit in search_results[0]]

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
        # search_params=search_params,
        limit=top_k,
        output_fields=["id", "description", "severity"],
    )
    if not search_results or not search_results[0]:
        print("No results found.")
        return []

    return [hit.entity for hit in search_results[0]]

if __name__ == "__main__":
    query = input("Enter a search query: ")
    top_k = int(input("Enter the number of results to return (default 5): ") or 5)

    results = hybrid_search_data(query, top_k)

    print(f"Found {len(results)} results:")
    for item in results:
        print(item)
