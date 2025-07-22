"""
Copyright (c) 2025 Mirasurf
"""

import logging
from typing import List, Union

from cogent_base.config import get_cogent_config
from cogent_base.config.consts import COGENT_RERANKER_EMBEDDING_LITTELM, COGENT_RERANKER_EMBEDDING_OLLAMA
from cogent_base.embedding.base_embedding import BaseEmbeddingModel
from cogent_base.models.chunk import Chunk

logger = logging.getLogger(__name__)


class CogentEmbeddingModel(BaseEmbeddingModel):
    """
    Cogent embedding model that provides a unified interface for different embedding providers.
    Uses registered models from the config file and routes to appropriate implementations.
    """

    def __init__(self, model_key: str) -> None:
        """
        Initialize Cogent embedding model with a model key from registered_models.

        Args:
            model_key: The key of the model in the registered_models config
        """
        settings = get_cogent_config()
        self.model_key = model_key
        self.embedding_impl: BaseEmbeddingModel = None

        # Get the model configuration from registered_models
        if not hasattr(settings.llm, "registered_models") or model_key not in settings.llm.registered_models:
            raise ValueError(f"Model '{model_key}' not found in registered_models configuration")

        self.model_config = settings.llm.registered_models[model_key]
        self.provider = settings.llm.embedding_provider
        self.dimensions = min(settings.llm.embedding_dimensions, 2000)

        # Initialize the appropriate embedding implementation
        if self.provider == COGENT_RERANKER_EMBEDDING_LITTELM:
            from cogent_base.embedding.litellm_embedding import LiteLLMEmbeddingModel

            self.embedding_impl = LiteLLMEmbeddingModel(model_key)
        elif self.provider == COGENT_RERANKER_EMBEDDING_OLLAMA:
            from cogent_base.embedding.ollama_embedding import OllamaEmbeddingModel

            self.embedding_impl = OllamaEmbeddingModel(model_key)
        else:
            raise ValueError(f"Embedding provider '{self.provider}' not supported")

        logger.info(f"Initialized Cogent embedding model with model_key={model_key}, config={self.model_config}")

    async def embed_objects(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of objects using the appropriate provider implementation.

        Args:
            texts: List of text objects to embed

        Returns:
            List of embedding vectors (one per object)
        """
        return await self.embedding_impl.embed_objects(texts)

    async def embed_query(self, text: str) -> List[float]:
        """
        Generate an embedding for a single query using the appropriate provider implementation.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        return await self.embedding_impl.embed_query(text)

    async def embed_for_chunks(self, chunks: Union[Chunk, List[Chunk]]) -> List[List[float]]:
        """
        Generate embeddings for chunks to be ingested into the vector store.

        Args:
            chunks: Single chunk or list of chunks to embed

        Returns:
            List of embedding vectors (one per chunk)
        """
        return await self.embedding_impl.embed_for_chunks(chunks)

    async def embed_for_query(self, text: str) -> List[float]:
        """
        Generate embedding for a query.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        return await self.embedding_impl.embed_for_query(text)
