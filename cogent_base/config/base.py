"""
Base configuration classes and decorators.
Provides the foundation for all configuration classes.
"""

from typing import Any, Callable, Dict, Optional, Type, TypeVar

from pydantic import BaseModel, ConfigDict

# Type variable for config classes
T = TypeVar("T", bound="BaseConfig")


def toml_config(section_name: str, default_factory: Optional[Callable[[], T]] = None) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator to add TOML loading capability to config classes.
    Supports nested section names like 'nova.agent.tools'.
    """

    def decorator(cls: Type[T]) -> Type[T]:
        @classmethod
        def from_toml(cls, toml_data: Dict[str, Any]) -> T:
            # Always require section_name
            if hasattr(cls, "_from_toml") and cls._from_toml != BaseConfig._from_toml:
                return cls._from_toml(toml_data, section_name=section_name)
            return BaseConfig._from_toml.__func__(cls, toml_data, section_name=section_name)

        cls.from_toml = from_toml
        return cls

    return decorator


class BaseConfig(BaseModel):
    """Base configuration class that provides common functionality."""

    model_config = ConfigDict(extra="allow")

    @classmethod
    def _from_toml(cls, toml_data: Dict[str, Any], section_name: str) -> "BaseConfig":
        # Support nested section names like 'nova.agent.tools'
        section_keys = section_name.split(".")
        section_data = toml_data
        for key in section_keys:
            if not isinstance(section_data, dict):
                section_data = {}
                break
            section_data = section_data.get(key, {})
        merged = {**{k: v for k, v in toml_data.items() if k in cls.model_fields}, **section_data}
        return cls(**merged)
