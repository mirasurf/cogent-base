"""
Core configuration classes.
Contains the main configuration classes for LLM, VectorStore, Reranker, and Sensory.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from .base import BaseConfig, toml_config
from .registry import ConfigRegistry
from .utils import get_user_cogent_toml_path, load_toml_config


@toml_config("llm")
class LLMConfig(BaseConfig):
    """LLM configuration."""

    registered_models: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # Completion configuration
    completion_provider: str = "ollama"
    completion_model: str = "ollama_qwen_vision"
    completion_max_tokens: int = 5000
    completion_temperature: float = 0.3

    # Embedding configuration
    embedding_provider: str = "ollama"
    embedding_model: str = "ollama_embedding"
    embedding_dimensions: int = 768
    embedding_similarity_metric: str = "cosine"
    embedding_batch_size: int = 100

    @classmethod
    def _from_toml(cls, toml_data: Dict[str, Any], section_name: Optional[str] = None) -> "LLMConfig":
        """Custom TOML loading implementation for LLMConfig."""

        def get(key: str, section: Dict[str, Any], cast=None, default=None) -> Any:
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


@toml_config("reranker")
class RerankerConfig(BaseConfig):
    """Configuration for rerankers from REGISTERED_RERANKERS."""

    registered_rerankers: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    enable_reranker: bool = False
    provider: str = "ollama"
    model: str = "ollama_reranker"


@toml_config("vector_store")
class VectorStoreConfig(BaseConfig):
    """Configuration for vector stores from REGISTERED_VECTOR_STORES."""

    registered_vector_stores: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    provider: str = "pgvector"
    collection_name: str = "cogent"
    embedding_model_dims: int = 768


@toml_config("sensory")
class SensoryConfig(BaseConfig):
    """Sensory configuration."""

    # parser config
    chunk_size: int = Field(default=6000)
    chunk_overlap: int = Field(default=300)
    use_unstructured_api: bool = Field(default=False)
    use_contextual_chunking: bool = Field(default=False)
    contextual_chunking_model: str = Field(default="ollama_qwen_vision")
    vision_model: str = Field(default="ollama_qwen_vision")
    vision_frame_sample_rate: int = Field(default=120)

    @classmethod
    def _from_toml(cls, toml_data: Dict[str, Any], section_name: Optional[str] = None) -> "SensoryConfig":
        parser_cfg = dict(toml_data.get("sensory", {}).get("parser", {}))
        return cls(**parser_cfg)


class CogentBaseConfig(BaseModel):
    """Main configuration class that combines all module configurations."""

    # Config registry for extensible submodule configs
    registry: ConfigRegistry = Field(default_factory=ConfigRegistry)

    def __init__(self, config_dir: Optional[Path] = None, **data) -> None:
        super().__init__(**data)
        self._load_default_configs()
        self._load_dot_cogent_toml(config_dir=config_dir)

    def _load_default_configs(self) -> None:
        """Load default submodule configurations (class defaults)."""
        self.registry.register("llm", LLMConfig())
        self.registry.register("vector_store", VectorStoreConfig())
        self.registry.register("reranker", RerankerConfig())
        self.registry.register("sensory", SensoryConfig())

    def _load_dot_cogent_toml(self, config_dir: Optional[Path] = None) -> None:
        """Load user runtime configuration that can override package defaults."""
        # Check for user runtime config in current working directory or provided config_dir
        runtime_config_path = get_user_cogent_toml_path(config_dir)
        toml_data = load_toml_config(runtime_config_path)
        if toml_data:
            self.registry.update_from_toml(toml_data)

    def register_config(self, name: str, config: BaseConfig) -> None:
        """Register a new submodule configuration."""
        self.registry.register(name, config)

    def get_config(self, name: str) -> Optional[BaseConfig]:
        """Get a submodule configuration by name."""
        return self.registry.get(name)

    def get_all_configs(self) -> Dict[str, BaseConfig]:
        """Get all registered submodule configurations."""
        return self.registry.get_all()

    # Convenience properties for backward compatibility
    @property
    def llm(self) -> LLMConfig:
        """Get LLM configuration."""
        return self.registry.get("llm")

    @property
    def vector_store(self) -> VectorStoreConfig:
        """Get vector store configuration."""
        return self.registry.get("vector_store")

    @property
    def reranker(self) -> RerankerConfig:
        """Get reranker configuration."""
        return self.registry.get("reranker")

    @property
    def sensory(self) -> SensoryConfig:
        """Get sensory configuration."""
        return self.registry.get("sensory")


# Global config instance
_config: Optional[CogentBaseConfig] = None


def get_cogent_config() -> CogentBaseConfig:
    """
    Get the global configuration instance.

    If no config has been initialized, creates one with default settings.

    Returns:
        CogentBaseConfig: The global configuration instance
    """
    global _config
    if _config is None:
        _config = CogentBaseConfig()
    return _config


def set_cogent_config_dir(config_dir: Path) -> None:
    """
    Set the configuration directory and reinitialize the global config.

    This will reload the configuration from the specified directory.

    Args:
        config_dir: Path to the directory containing .cogent.toml
    """
    global _config
    _config = CogentBaseConfig(config_dir=config_dir)


def init_cogent_config(config_dir: Optional[Path] = None) -> CogentBaseConfig:
    """
    Initialize the global configuration with optional config directory.

    Args:
        config_dir: Optional path to the directory containing .cogent.toml

    Returns:
        CogentBaseConfig: The initialized configuration instance
    """
    global _config
    _config = CogentBaseConfig(config_dir=config_dir)
    return _config
