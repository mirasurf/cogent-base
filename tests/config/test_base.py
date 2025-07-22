"""
Tests for base configuration classes and decorators.
"""

import unittest
from typing import Any, Dict, Optional

import pytest

from cogent_base.config import BaseConfig, toml_config


class TestBaseConfig(unittest.TestCase):
    """Test the BaseConfig class."""

    @pytest.mark.unit
    def test_base_config_creation(self):
        """Test creating a BaseConfig instance."""
        config = BaseConfig()
        self.assertIsInstance(config, BaseConfig)


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
            def _from_toml(cls, toml_data: Dict[str, Any], section_name: Optional[str] = None) -> "TestConfig":
                return cls(value="custom_from_toml")

        toml_data = {"test_section": {"value": "ignored"}}
        config = TestConfig.from_toml(toml_data)

        # Should use custom implementation
        self.assertEqual(config.value, "custom_from_toml")

    @pytest.mark.unit
    def test_toml_config_nested_section(self):
        """Test toml_config decorator with nested section names."""

        @toml_config("agent")
        class AgentConfig(BaseConfig):
            value: str = "default"

        @toml_config("nova.agent")
        class NovaAgentConfig(BaseConfig):
            value: str = "default"
            number: int = 0

        @toml_config("nova.agent.tools")
        class NovaAgentToolsConfig(BaseConfig):
            enabled: bool = False
            threshold: float = 0.0

        toml_data = {
            "agent": {"value": "agent_value"},
            "nova": {
                "agent": {"value": "nova_agent_value", "number": 123, "tools": {"enabled": True, "threshold": 0.75}}
            },
        }

        # Test single-level section
        agent_config = AgentConfig.from_toml(toml_data)
        self.assertEqual(agent_config.value, "agent_value")

        # Test two-level section
        nova_agent_config = NovaAgentConfig.from_toml(toml_data)
        self.assertEqual(nova_agent_config.value, "nova_agent_value")
        self.assertEqual(nova_agent_config.number, 123)

        # Test three-level section
        nova_agent_tools_config = NovaAgentToolsConfig.from_toml(toml_data)
        self.assertTrue(nova_agent_tools_config.enabled)
        self.assertAlmostEqual(nova_agent_tools_config.threshold, 0.75)
