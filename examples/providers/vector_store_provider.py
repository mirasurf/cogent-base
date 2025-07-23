"""
Vector Store Provider Usage Example

This example demonstrates how to use vector store providers for
storing and searching vectors.
"""

import asyncio

import numpy as np

from cogent_base.providers.vector_store import CogentVectorStore


async def main():
    # Initialize the vector store
    store = CogentVectorStore()

    # Example vectors and metadata
    vectors = [
        np.random.rand(128).tolist(),  # Sample 128-dim vector
        np.random.rand(128).tolist(),
        np.random.rand(128).tolist(),
    ]

    metadata = [
        {"text": "Hello world", "source": "doc1"},
        {"text": "Vector search", "source": "doc2"},
        {"text": "Machine learning", "source": "doc3"},
    ]

    try:
        # Insert vectors
        await store.insert(vectors, metadata)
        print(f"Inserted {len(vectors)} vectors")

        # Search for similar vectors
        query_vector = np.random.rand(128).tolist()
        results = await store.search(query_vector, limit=10)

        print(f"Found {len(results)} similar vectors")

    except Exception as e:
        print(f"Error with vector store operations: {e}")


if __name__ == "__main__":
    asyncio.run(main())
