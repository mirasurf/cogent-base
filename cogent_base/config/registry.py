"""
Configuration registry.
Manages submodule configurations and provides extensibility.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

from .base import BaseConfig


class ConfigRegistry(BaseModel):
    """Registry for managing submodule configurations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    configs: Dict[str, BaseConfig] = Field(default_factory=dict)

    def register(self, name: str, config: BaseConfig) -> None:
        """Register a configuration instance."""
        self.configs[name] = config

    def get(self, name: str) -> Optional[BaseConfig]:
        """Get a configuration by name."""
        return self.configs.get(name)

    def get_all(self) -> Dict[str, BaseConfig]:
        """Get all registered configurations."""
        return self.configs.copy()

    def update_from_toml(self, toml_data: Dict[str, Any]) -> None:
        """Update all registered configs from TOML data."""
        for name, config in self.configs.items():
            if hasattr(config, "from_toml"):
                updated_config = config.from_toml(toml_data)
                self.configs[name] = updated_config
