"""
Main configuration module.
Contains the main CogentConfig class and global configuration instance.
"""

from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from cogent.base.rootdir import ROOT_DIR

from .base import BaseConfig
from .core import LLMConfig, RerankerConfig, SensoryConfig, VectorStoreConfig
from .registry import ConfigRegistry
from .utils import load_merged_toml_configs

# Load environment variables from .env file
load_dotenv(override=True)


class CogentConfig(BaseModel):
    """Main configuration class that combines all module configurations."""

    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=True)

    # Environment
    env: str = Field(default="development", validation_alias="ENV")
    debug: bool = Field(default=False, validation_alias="DEBUG")

    # Optional override paths
    config_dir: Path | None = Field(default=None, validation_alias="CONFIG_DIR")

    # Derived paths (set in post-init)
    base_toml: Path = Field(default=None)
    providers_toml: Path = Field(default=None)
    sensory_toml: Path = Field(default=None)

    # Config registry for extensible submodule configs
    registry: ConfigRegistry = Field(default_factory=ConfigRegistry)

    def model_post_init(self, __context):
        """Post-init hook to compute derived paths."""
        self.config_dir = self.config_dir or ROOT_DIR / "config"
        self.base_toml = self.config_dir / "base.toml"
        self.providers_toml = self.config_dir / "providers.toml"
        self.sensory_toml = self.config_dir / "sensory.toml"

    def __init__(self, **data):
        super().__init__(**data)
        self._load_default_configs()
        self._load_toml_config()

    def _load_default_configs(self):
        """Load default submodule configurations."""
        self.registry.register("llm", LLMConfig())
        self.registry.register("vector_store", VectorStoreConfig())
        self.registry.register("reranker", RerankerConfig())
        self.registry.register("sensory", SensoryConfig())

    def _load_toml_config(self):
        """Load and merge TOML configurations, updating submodules."""
        toml_paths = [self.base_toml, self.providers_toml, self.sensory_toml]
        toml_data = load_merged_toml_configs(toml_paths)
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
config = CogentConfig()


def get_cogent_config() -> CogentConfig:
    """
    Get the global configuration instance.

    Returns:
        CogentConfig: The global configuration instance
    """
    return config
