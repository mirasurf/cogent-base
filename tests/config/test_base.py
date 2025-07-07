"""
Tests for base configuration classes and decorators.
"""

import unittest
from typing import Any, Dict

import pytest

from cogentbase.config import BaseConfig, toml_config


class TestBaseConfig(unittest.TestCase):
    """Test the BaseConfig class."""

    @pytest.mark.unit
    def test_base_config_creation(self):
        """Test creating a BaseConfig instance."""
        config = BaseConfig()
        self.assertIsInstance(config, BaseConfig)

    @pytest.mark.unit
    def test_get_toml_section(self):
        """Test the default get_toml_section method."""
        config = BaseConfig()
        section = config.get_toml_section()
        self.assertEqual(section, "base")


class TestTomlConfigDecorator(unittest.TestCase):
    """Test the toml_config decorator."""

    @pytest.mark.unit
    def test_toml_config_decorator(self):
        """Test the toml_config decorator functionality."""

        @toml_config("test_section")
        class TestConfig(BaseConfig):
            value: str = "default"
            number: int = 42

        # Test that from_toml method was added
        self.assertTrue(hasattr(TestConfig, "from_toml"))

        # Test loading from TOML
        toml_data = {"test_section": {"value": "custom", "number": 100}}
        config = TestConfig.from_toml(toml_data)

        self.assertEqual(config.value, "custom")
        self.assertEqual(config.number, 100)

    @pytest.mark.unit
    def test_toml_config_decorator_with_custom_implementation(self):
        """Test toml_config decorator with custom _from_toml method."""

        @toml_config("test_section")
        class TestConfig(BaseConfig):
            value: str = "default"

            @classmethod
            def _from_toml(cls, toml_data: Dict[str, Any]) -> "TestConfig":
                return cls(value="custom_from_toml")

        toml_data = {"test_section": {"value": "ignored"}}
        config = TestConfig.from_toml(toml_data)

        # Should use custom implementation
        self.assertEqual(config.value, "custom_from_toml")
