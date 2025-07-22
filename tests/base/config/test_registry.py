"""
Tests for configuration registry.
"""

import unittest

import pytest

from cogent.base.config import BaseConfig, ConfigRegistry, toml_config


class TestConfigRegistry(unittest.TestCase):
    """Test the ConfigRegistry class."""

    @pytest.mark.unit
    def test_config_registry_creation(self):
        """Test creating a ConfigRegistry instance."""
        registry = ConfigRegistry()
        self.assertIsInstance(registry, ConfigRegistry)

    @pytest.mark.unit
    def test_register_and_get_config(self):
        """Test registering and retrieving configurations."""
        registry = ConfigRegistry()
        config = BaseConfig()

        registry.register("test", config)
        retrieved = registry.get("test")

        self.assertEqual(retrieved, config)

    @pytest.mark.unit
    def test_get_nonexistent_config(self):
        """Test getting a non-existent configuration."""
        registry = ConfigRegistry()
        config = registry.get("nonexistent")

        self.assertIsNone(config)

    @pytest.mark.unit
    def test_get_all_configs(self):
        """Test getting all registered configurations."""
        registry = ConfigRegistry()
        config1 = BaseConfig()
        config2 = BaseConfig()

        registry.register("test1", config1)
        registry.register("test2", config2)

        all_configs = registry.get_all()

        self.assertEqual(len(all_configs), 2)
        self.assertEqual(all_configs["test1"], config1)
        self.assertEqual(all_configs["test2"], config2)

    @pytest.mark.unit
    def test_update_from_toml(self):
        """Test updating configurations from TOML data."""
        registry = ConfigRegistry()

        # Create a config with from_toml capability
        @toml_config("test_section")
        class TestConfig(BaseConfig):
            value: str = "default"

        config = TestConfig()
        registry.register("test", config)

        # Update from TOML
        toml_data = {"test_section": {"value": "updated"}}
        registry.update_from_toml(toml_data)

        updated_config = registry.get("test")
        self.assertEqual(updated_config.value, "updated")
