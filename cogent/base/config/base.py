"""
Base configuration classes and decorators.
Provides the foundation for all configuration classes.
"""

from typing import Any, Callable, Dict, Optional, Type, TypeVar

from pydantic import BaseModel, ConfigDict

# Type variable for config classes
T = TypeVar("T", bound="BaseConfig")


def toml_config(section_name: str, default_factory: Optional[Callable[[], T]] = None):
    """
    Decorator to add TOML loading capability to config classes.

    Args:
        section_name: The section name in TOML file to load from
        default_factory: Optional factory function to create default instance

    Returns:
        Decorated class with from_toml method
    """

    def decorator(cls: Type[T]) -> Type[T]:
        @classmethod
        def from_toml(cls, toml_data: Dict[str, Any]) -> T:
            """Load config from TOML data."""
            section_data = toml_data.get(section_name, {})

            # If the class has a custom from_toml implementation, use it
            if hasattr(cls, "_from_toml"):
                return cls._from_toml(toml_data)

            # Default implementation: create instance with section data
            if default_factory:
                default_instance = default_factory()
            else:
                default_instance = cls()

            # Try to create instance with section data, fallback to defaults
            try:
                return cls(**section_data)
            except Exception:
                return default_instance

        # Add the from_toml method to the class
        cls.from_toml = from_toml
        return cls

    return decorator


class BaseConfig(BaseModel):
    """Base configuration class that provides common functionality."""

    model_config = ConfigDict(extra="allow")

    def get_toml_section(self) -> str:
        """Get the TOML section name for this config. Override in subclasses."""
        return self.__class__.__name__.lower().replace("config", "")
