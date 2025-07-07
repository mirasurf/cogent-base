"""
Common configuration module for the Cogent project.
Supports environment variables and provides structured configuration management.
"""

import copy
import tomllib
from pathlib import Path
from typing import Any, Dict, List, Mapping

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from cogent.base.rootdir import ROOT_DIR

# Load environment variables from .env file
load_dotenv(override=True)


def load_toml_config(toml_path: Path) -> Dict[str, Any]:
    """Load configuration from TOML file."""
    try:
        with open(toml_path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        print(f"Warning: TOML config file not found at {toml_path}")
        return {}
    except Exception as e:
        print(f"Error loading TOML config: {e}")
        return {}


def deep_merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge dict b into dict a without modifying inputs."""
    result = copy.deepcopy(a)
    for key, value in b.items():
        if key in result and isinstance(result[key], Mapping) and isinstance(value, Mapping):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_merged_toml_configs(toml_paths: List[Path]) -> Dict[str, Any]:
    """Load and merge multiple TOML config files into a unified settings dictionary."""
    merged_config: Dict[str, Any] = {}
    for path in toml_paths:
        config = load_toml_config(path)
        merged_config = deep_merge_dicts(merged_config, config)
    return merged_config




def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _safe_bool(value: Any, default: bool) -> bool:
    """Safely convert value to boolean, falling back to default if conversion fails."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes", "on")
    try:
        return bool(value)
    except (ValueError, TypeError):
        return default
    

class LLMConfig(BaseModel):
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

    @classmethod
    def from_toml(cls, toml_data: Dict[str, Any]) -> "LLMConfig":
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


class RerankerConfig(BaseModel):
    """Configuration for rerankers from REGISTERED_RERANKERS."""

    registered_rerankers: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    enable_reranker: bool = False
    reranker_provider: str = "litellm"
    reranker_model: str = "ollama_reranker"

    @classmethod
    def from_toml(cls, toml_data: Dict[str, Any]) -> "RerankerConfig":
        reranker_cfg = toml_data.get("reranker", {})
        return cls(
            registered_rerankers=toml_data.get("registered_rerankers", {}),
            enable_reranker=reranker_cfg.get("enable_reranker", cls().enable_reranker),
            reranker_provider=reranker_cfg.get("provider", cls().reranker_provider),
            reranker_model=reranker_cfg.get("model", cls().reranker_model),
        )


class VectorStoreConfig(BaseModel):
    """Configuration for vector stores from REGISTERED_VECTOR_STORES."""

    registered_vector_stores: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    provider: str = "pgvector"
    collection_name: str = "cogent"
    embedding_model_dims: int = 768

    @classmethod
    def from_toml(cls, toml_data: Dict[str, Any]) -> "VectorStoreConfig":
        """Create VectorStoreConfig from TOML data."""
        vector_store_cfg = toml_data.get("vector_store", {})

        return cls(
            registered_vector_stores=toml_data.get("registered_vector_stores", {}),
            provider=vector_store_cfg.get("provider", cls().provider),
            collection_name=vector_store_cfg.get("collection_name", cls().collection_name),
            embedding_model_dims=_safe_int(vector_store_cfg.get("embedding_model_dims"), cls().embedding_model_dims),
        )

class SensoryConfig(BaseModel):
    """Sensory configuration."""

    # parser config
    chunk_size: int = Field(default=6000)
    chunk_overlap: int = Field(default=300)
    use_unstructured_api: bool = Field(default=False)
    use_contextual_chunking: bool = Field(default=False)
    contextual_chunking_model: str = Field(default="ollama_qwen_vision")

    @classmethod
    def from_toml(cls, toml_data: Dict[str, Any]) -> "SensoryConfig":
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


# Cogent all configurations
class CogentConfig(BaseModel):
    """Main configuration class that combines all module configurations."""

    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=True)

    # Environment
    env: str = Field(default="development", validation_alias="ENV")
    debug: bool = Field(default=False, validation_alias="DEBUG")

    # Optional override paths
    config_dir: Path | None = Field(default=None, validation_alias="CONFIG_DIR")
    log_dir: Path | None = Field(default=None, validation_alias="LOG_DIR")

    # Derived paths (set in post-init)
    base_toml: Path = Field(default=None)
    providers_toml: Path = Field(default=None)
    sensory_toml: Path = Field(default=None)

    # Logging
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", validation_alias="LOG_FORMAT"
    )

    # Module configurations
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    sensory: SensoryConfig = Field(default_factory=SensoryConfig)

    def model_post_init(self, __context):
        """Post-init hook to compute derived paths."""
        self.config_dir = self.config_dir or ROOT_DIR / "config"
        self.log_dir = self.log_dir or ROOT_DIR / "logs"
        self.base_toml = self.config_dir / "base.toml"
        self.providers_toml = self.config_dir / "providers.toml"
        self.sensory_toml = self.config_dir / "sensory.toml"

    def __init__(self, **data):
        super().__init__(**data)
        self._load_toml_config()

    def _load_toml_config(self):
        """Load and merge TOML configurations, updating submodules."""
        toml_paths = [self.base_toml, self.providers_toml, self.sensory_toml]
        toml_data = load_merged_toml_configs(toml_paths)
        if toml_data:
            self.llm = LLMConfig.from_toml(toml_data)
            self.vector_store = VectorStoreConfig.from_toml(toml_data)
            self.reranker = RerankerConfig.from_toml(toml_data)
            self.sensory = SensoryConfig.from_toml(toml_data)


# Create global config instance
config = CogentConfig()


def get_config() -> CogentConfig:
    """
    Get the global configuration instance.

    Returns:
        CogentConfig: The global configuration instance
    """
    return config
