"""
Cogent configuration module.
Provides extensible configuration management for agentic cognitive computing frameworks.
"""

from .base import BaseConfig, toml_config
from .consts import (
    COGENT_RERANKER_COMPLETION_LITTELM,
    COGENT_RERANKER_COMPLETION_OLLAMA,
    COGENT_RERANKER_EMBEDDING_LITTELM,
    COGENT_RERANKER_EMBEDDING_OLLAMA,
    COGENT_RERANKER_PROVIDER_FLAG,
    COGENT_RERANKER_PROVIDER_LITTELM,
    COGENT_RERANKER_PROVIDER_OLLAMA,
    COGENT_VECTOR_STORE_PROVIDER_PGVECTOR,
    COGENT_VECTOR_STORE_PROVIDER_WEAVIATE,
)
from .core import (
    CogentBaseConfig,
    LLMConfig,
    RerankerConfig,
    SensoryConfig,
    VectorStoreConfig,
    get_cogent_config,
)

__all__ = [
    # Base classes and decorators
    "BaseConfig",
    "toml_config",
    # Core configuration classes
    "LLMConfig",
    "VectorStoreConfig",
    "RerankerConfig",
    "SensoryConfig",
    # Main configuration
    "CogentBaseConfig",
    "get_cogent_config",
    # Constants
    "COGENT_RERANKER_COMPLETION_OLLAMA",
    "COGENT_RERANKER_COMPLETION_LITTELM",
    "COGENT_RERANKER_EMBEDDING_OLLAMA",
    "COGENT_RERANKER_EMBEDDING_LITTELM",
    "COGENT_RERANKER_PROVIDER_OLLAMA",
    "COGENT_RERANKER_PROVIDER_FLAG",
    "COGENT_RERANKER_PROVIDER_LITTELM",
    "COGENT_VECTOR_STORE_PROVIDER_PGVECTOR",
    "COGENT_VECTOR_STORE_PROVIDER_WEAVIATE",
]
