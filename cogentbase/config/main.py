"""
Main configuration module.
Contains the main CogentBaseConfig class and global configuration instance.
"""

from pathlib import Path
from typing import Dict, Optional

from pydantic import BaseModel, Field

from .base import BaseConfig
from .core import LLMConfig, RerankerConfig, SensoryConfig, VectorStoreConfig
from .registry import ConfigRegistry
from .utils import load_toml_config


class CogentBaseConfig(BaseModel):
    """Main configuration class that combines all module configurations."""

    # Config registry for extensible submodule configs
    registry: ConfigRegistry = Field(default_factory=ConfigRegistry)

    def __init__(self, **data):
        super().__init__(**data)
        self._load_default_configs()
        self._load_package_defaults()
        self._load_user_runtime_config()

    def _load_default_configs(self):
        """Load default submodule configurations (class defaults)."""
        self.registry.register("llm", LLMConfig())
        self.registry.register("vector_store", VectorStoreConfig())
        self.registry.register("reranker", RerankerConfig())
        self.registry.register("sensory", SensoryConfig())

    def _load_package_defaults(self):
        """Load package default configuration from base.toml."""
        package_config_path = Path(__file__).parent / "base.toml"
        toml_data = load_toml_config(package_config_path)
        if toml_data:
            self.registry.update_from_toml(toml_data)

    def _load_user_runtime_config(self):
        """Load user runtime configuration that can override package defaults."""
        # Check for user runtime config in current working directory
        runtime_config_path = Path.cwd() / "base.toml"
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


# Create global config instance
config = CogentBaseConfig()


def get_cogent_config() -> CogentBaseConfig:
    """
    Get the global configuration instance.

    Returns:
        CogentBaseConfig: The global configuration instance
    """
    return config
