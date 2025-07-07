"""
Configuration utilities.
Provides TOML loading and helper functions for configuration management.
"""

import copy
import tomllib
from pathlib import Path
from typing import Any, Dict, List, Mapping


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
    """Safely convert value to integer, falling back to default if conversion fails."""
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
