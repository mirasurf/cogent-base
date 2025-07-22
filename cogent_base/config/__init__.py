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
from .registry import ConfigRegistry
from .utils import (
    deep_merge_dicts,
    load_merged_toml_configs,
    load_toml_config,
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
    # Registry
    "ConfigRegistry",
    # Main configuration
    "CogentBaseConfig",
    "get_cogent_config",
    # Utility functions
    "load_toml_config",
    "load_merged_toml_configs",
    "deep_merge_dicts",
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
