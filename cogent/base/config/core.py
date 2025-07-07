"""
Core configuration classes.
Contains the main configuration classes for LLM, VectorStore, Reranker, and Sensory.
"""

from typing import Any, Dict

from pydantic import Field

from .base import BaseConfig, toml_config
from .utils import _safe_bool, _safe_int


@toml_config("llm")
class LLMConfig(BaseConfig):
    """LLM configuration."""

    registered_models: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # Completion configuration
    completion_provider: str = "litellm"
    completion_model: str = "openai_gpt4-1-mini"
    completion_max_tokens: int = 2000
    completion_temperature: float = 0.7

    # Embedding configuration
    embedding_provider: str = "litellm"
    embedding_model: str = "openai_embedding"
    embedding_dimensions: int = 768
    embedding_similarity_metric: str = "cosine"
    embedding_batch_size: int = 100

    def get_toml_section(self) -> str:
        return "llm"

    @classmethod
    def _from_toml(cls, toml_data: Dict[str, Any]) -> "LLMConfig":
        """Custom TOML loading implementation for LLMConfig."""

        def get(key: str, section: Dict[str, Any], cast=None, default=None):
            val = section.get(key, default)
            if cast:
                try:
                    return cast(val)
                except (ValueError, TypeError):
                    return default
            return val if val is not None else default

        return cls(
            registered_models=toml_data.get("registered_models", {}),
            completion_provider=get("provider", toml_data.get("completion", {}), str, cls().completion_provider),
            completion_model=get("model", toml_data.get("completion", {}), str, cls().completion_model),
            completion_max_tokens=get(
                "default_max_tokens", toml_data.get("completion", {}), int, cls().completion_max_tokens
            ),
            completion_temperature=get(
                "default_temperature", toml_data.get("completion", {}), float, cls().completion_temperature
            ),
            embedding_provider=get("provider", toml_data.get("embedding", {}), str, cls().embedding_provider),
            embedding_model=get("model", toml_data.get("embedding", {}), str, cls().embedding_model),
            embedding_dimensions=get("dimensions", toml_data.get("embedding", {}), int, cls().embedding_dimensions),
            embedding_similarity_metric=get(
                "similarity_metric", toml_data.get("embedding", {}), str, cls().embedding_similarity_metric
            ),
            embedding_batch_size=get("batch_size", toml_data.get("embedding", {}), int, cls().embedding_batch_size),
        )


@toml_config("vector_store")
class VectorStoreConfig(BaseConfig):
    """Configuration for vector stores from REGISTERED_VECTOR_STORES."""

    registered_vector_stores: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    provider: str = "pgvector"
    collection_name: str = "cogent"
    embedding_model_dims: int = 768

    def get_toml_section(self) -> str:
        return "vector_store"

    @classmethod
    def _from_toml(cls, toml_data: Dict[str, Any]) -> "VectorStoreConfig":
        """Custom TOML loading implementation for VectorStoreConfig."""
        vector_store_cfg = toml_data.get("vector_store", {})

        return cls(
            registered_vector_stores=toml_data.get("registered_vector_stores", {}),
            provider=vector_store_cfg.get("provider", cls().provider),
            collection_name=vector_store_cfg.get("collection_name", cls().collection_name),
            embedding_model_dims=_safe_int(vector_store_cfg.get("embedding_model_dims"), cls().embedding_model_dims),
        )


@toml_config("reranker")
class RerankerConfig(BaseConfig):
    """Configuration for rerankers from REGISTERED_RERANKERS."""

    registered_rerankers: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    enable_reranker: bool = False
    reranker_provider: str = "litellm"
    reranker_model: str = "ollama_reranker"

    def get_toml_section(self) -> str:
        return "reranker"

    @classmethod
    def _from_toml(cls, toml_data: Dict[str, Any]) -> "RerankerConfig":
        """Custom TOML loading implementation for RerankerConfig."""
        reranker_cfg = toml_data.get("reranker", {})
        return cls(
            registered_rerankers=toml_data.get("registered_rerankers", {}),
            enable_reranker=reranker_cfg.get("enable_reranker", cls().enable_reranker),
            reranker_provider=reranker_cfg.get("provider", cls().reranker_provider),
            reranker_model=reranker_cfg.get("model", cls().reranker_model),
        )


@toml_config("sensory")
class SensoryConfig(BaseConfig):
    """Sensory configuration."""

    # parser config
    chunk_size: int = Field(default=6000)
    chunk_overlap: int = Field(default=300)
    use_unstructured_api: bool = Field(default=False)
    use_contextual_chunking: bool = Field(default=False)
    contextual_chunking_model: str = Field(default="ollama_qwen_vision")

    def get_toml_section(self) -> str:
        return "sensory"

    @classmethod
    def _from_toml(cls, toml_data: Dict[str, Any]) -> "SensoryConfig":
        """Custom TOML loading implementation for SensoryConfig."""
        parser_cfg = toml_data.get("sensory", {}).get("parser", {})
        default_config = cls()
        return cls(
            chunk_size=_safe_int(parser_cfg.get("chunk_size"), default_config.chunk_size),
            chunk_overlap=_safe_int(parser_cfg.get("chunk_overlap"), default_config.chunk_overlap),
            use_unstructured_api=_safe_bool(
                parser_cfg.get("use_unstructured_api"), default_config.use_unstructured_api
            ),
            use_contextual_chunking=_safe_bool(
                parser_cfg.get("use_contextual_chunking"), default_config.use_contextual_chunking
            ),
            contextual_chunking_model=parser_cfg.get(
                "contextual_chunking_model", default_config.contextual_chunking_model
            ),
        )
