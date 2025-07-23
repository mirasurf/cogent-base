"""
Embedding Provider Usage Example

This example demonstrates how to use embedding providers for text embeddings.
"""

import asyncio

from cogent_base.providers.embedding import CogentEmbeddingModel


async def main():
    # Initialize the embedding model
    model = CogentEmbeddingModel("text-embedding-ada-002")

    # Example texts to embed
    texts = ["Hello", "World", "This is a test"]

    try:
        # Generate embeddings
        embeddings = await model.embed_texts(texts)

        print(f"Generated embeddings for {len(texts)} texts")
        print(f"Embedding dimensions: {len(embeddings[0]) if embeddings else 'N/A'}")

    except Exception as e:
        print(f"Error generating embeddings: {e}")


if __name__ == "__main__":
    asyncio.run(main())
