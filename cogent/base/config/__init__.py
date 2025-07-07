"""
Cogent configuration module.
Provides extensible configuration management for agentic cognitive computing frameworks.
"""

from .base import BaseConfig, toml_config
from .core import (
    LLMConfig,
    RerankerConfig,
    SensoryConfig,
    VectorStoreConfig,
)
from .main import CogentBaseConfig, get_cogent_config
from .registry import ConfigRegistry
from .utils import (
    _safe_bool,
    _safe_int,
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
    "_safe_int",
    "_safe_bool",
]
