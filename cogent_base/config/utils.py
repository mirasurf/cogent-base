"""
Configuration utilities.
Provides TOML loading and helper functions for configuration management.
"""

import copy
import os
import tomllib
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


def load_toml_config(toml_path: Path) -> Dict[str, Any]:
    """Load configuration from TOML file."""
    try:
        with open(toml_path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
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


def get_user_cogent_toml_path(config_dir: Optional[Path] = None) -> Path:
    """Get the path to the user's .cogent.toml file."""
    if config_dir is not None:
        return Path(config_dir) / ".cogent.toml"
    cogent_config_env = os.environ.get("COGENT_CONFIG_DIR")
    if cogent_config_env:
        return Path(cogent_config_env) / ".cogent.toml"
    return Path.cwd() / ".cogent.toml"
