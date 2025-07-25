"""
Copyright (c) 2025 Mirasurf
Copyright (c) 2023-2025 morphik/morphik-core
Original code from https://github.com/morphik/morphik-core
"""

import logging
from typing import List, Union

import litellm

from cogent_base.config import get_cogent_config
from cogent_base.embedding.base_embedding import BaseEmbeddingModel
from cogent_base.models.chunk import Chunk

logger = logging.getLogger(__name__)
PGVECTOR_MAX_DIMENSIONS = 2000  # Maximum dimensions for pgvector


class LiteLLMEmbeddingModel(BaseEmbeddingModel):
    """
    LiteLLM embedding model implementation that provides unified access to
    various embedding providers. Uses registered models from the config file.
    """

    def __init__(self, model_key: str) -> None:
        """
        Initialize LiteLLM embedding model with a model key from registered_models.

        Args:
            model_key: The key of the model in the registered_models config
        """
        settings = get_cogent_config()
        self.model_key = model_key

        # Get the model configuration from registered_models
        if not hasattr(settings.llm, "registered_models") or model_key not in settings.llm.registered_models:
            raise ValueError(f"Model '{model_key}' not found in registered_models configuration")

        self.model_config = settings.llm.registered_models[model_key]
        self.dimensions = min(settings.llm.embedding_dimensions, 2000)

        logger.info(f"Initialized LiteLLM embedding model with model_key={model_key}, " f"config={self.model_config}")

    async def _embed_with_litellm(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using LiteLLM.

        Args:
            texts: List of text objects to embed

        Returns:
            List of embedding vectors (one per object)
        """
        if not texts:
            return []

        try:
            model_params = {"model": self.model_config["model_name"]}
            if self.model_config["model_name"] in [
                "text-embedding-3-large",
                "azure/text-embedding-3-large",
            ]:
                model_params["dimensions"] = PGVECTOR_MAX_DIMENSIONS

            # Add all model-specific parameters from the config
            for key, value in self.model_config.items():
                if key != "model_name":  # Skip as we've already handled it
                    model_params[key] = value

            # Call LiteLLM
            response = await litellm.aembedding(input=texts, **model_params)

            embeddings = [data["embedding"] for data in response.data]

            # Validate dimensions
            if embeddings and len(embeddings[0]) != self.dimensions:
                logger.warning(
                    f"Embedding dimension mismatch: got {len(embeddings[0])}, "
                    f"expected {self.dimensions}. Please update your VECTOR_DIMENSIONS "
                    f"setting to match the actual dimension."
                )

            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings with LiteLLM: {e}")
            raise

    async def embed_objects(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of objects using LiteLLM.

        Args:
            texts: List of text objects to embed

        Returns:
            List of embedding vectors (one per object)
        """
        return await self._embed_with_litellm(texts)

    async def embed_query(self, text: str) -> List[float]:
        """
        Generate an embedding for a single query using LiteLLM.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        result = await self.embed_objects([text])
        if not result:
            # In case of error, return zero vector
            return [0.0] * self.dimensions
        return result[0]

    async def embed_for_chunks(self, chunks: Union[Chunk, List[Chunk]]) -> List[List[float]]:
        """
        Generate embeddings for chunks to be ingested into the vector store.

        Args:
            chunks: Single chunk or list of chunks to embed

        Returns:
            List of embedding vectors (one per chunk)
        """
        if isinstance(chunks, Chunk):
            chunks = [chunks]

        texts = [chunk.content for chunk in chunks]
        # Batch embedding to respect token limits
        settings = get_cogent_config()
        batch_size = settings.llm.embedding_batch_size
        embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_embeddings = await self.embed_objects(batch_texts)
            embeddings.extend(batch_embeddings)
        return embeddings

    async def embed_for_query(self, text: str) -> List[float]:
        """
        Generate embedding for a query.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        return await self.embed_query(text)
