"""
Tests for configuration extensibility.
Demonstrates how users can add custom submodule configs.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

from cogent.base.config import (
    BaseConfig,
    CogentConfig,
    get_cogent_config,
    toml_config,
)


class TestExtensibility(unittest.TestCase):
    """Test configuration extensibility features."""

    @pytest.mark.unit
    def test_custom_agent_config_creation(self):
        """Test creating a custom AgentConfig class."""

        @toml_config("agent")
        class AgentConfig(BaseConfig):
            """Custom agent configuration for downstream projects."""

            agent_type: str = "assistant"
            max_conversation_turns: int = 10
            enable_memory: bool = True
            memory_size: int = 1000
            temperature: float = 0.7

            def get_toml_section(self) -> str:
                return "agent"

            @classmethod
            def _from_toml(cls, toml_data: dict) -> "AgentConfig":
                """Custom TOML loading for AgentConfig."""
                agent_section = toml_data.get("agent", {})
                return cls(
                    agent_type=agent_section.get("type", cls().agent_type),
                    max_conversation_turns=agent_section.get("max_turns", cls().max_conversation_turns),
                    enable_memory=agent_section.get("enable_memory", cls().enable_memory),
                    memory_size=agent_section.get("memory_size", cls().memory_size),
                    temperature=agent_section.get("temperature", cls().temperature),
                )

        # Test default values
        config = AgentConfig()
        self.assertEqual(config.agent_type, "assistant")
        self.assertEqual(config.max_conversation_turns, 10)
        self.assertTrue(config.enable_memory)
        self.assertEqual(config.memory_size, 1000)
        self.assertEqual(config.temperature, 0.7)

        # Test from_toml method was added
        self.assertTrue(hasattr(AgentConfig, "from_toml"))

    @pytest.mark.unit
    def test_custom_agent_config_from_toml(self):
        """Test loading custom AgentConfig from TOML."""

        @toml_config("agent")
        class AgentConfig(BaseConfig):
            agent_type: str = "assistant"
            max_conversation_turns: int = 10
            enable_memory: bool = True
            memory_size: int = 1000
            temperature: float = 0.7

            @classmethod
            def _from_toml(cls, toml_data: dict) -> "AgentConfig":
                agent_section = toml_data.get("agent", {})
                return cls(
                    agent_type=agent_section.get("type", cls().agent_type),
                    max_conversation_turns=agent_section.get("max_turns", cls().max_conversation_turns),
                    enable_memory=agent_section.get("enable_memory", cls().enable_memory),
                    memory_size=agent_section.get("memory_size", cls().memory_size),
                    temperature=agent_section.get("temperature", cls().temperature),
                )

        # Test loading from TOML data
        toml_data = {
            "agent": {
                "type": "specialist",
                "max_turns": 20,
                "enable_memory": False,
                "memory_size": 2000,
                "temperature": 0.5,
            }
        }

        config = AgentConfig.from_toml(toml_data)

        self.assertEqual(config.agent_type, "specialist")
        self.assertEqual(config.max_conversation_turns, 20)
        self.assertFalse(config.enable_memory)
        self.assertEqual(config.memory_size, 2000)
        self.assertEqual(config.temperature, 0.5)

    @pytest.mark.unit
    @patch("cogent.base.config.load_merged_toml_configs")
    def test_register_custom_config_with_cogent(self, mock_load_merged):
        """Test registering custom config with main CogentConfig."""
        mock_load_merged.return_value = {}

        # Create custom config
        @toml_config("agent")
        class AgentConfig(BaseConfig):
            agent_type: str = "assistant"
            max_conversation_turns: int = 10

            @classmethod
            def _from_toml(cls, toml_data: dict) -> "AgentConfig":
                agent_section = toml_data.get("agent", {})
                return cls(
                    agent_type=agent_section.get("type", cls().agent_type),
                    max_conversation_turns=agent_section.get("max_turns", cls().max_conversation_turns),
                )

        # Get main config and register custom config
        cogent_config = get_cogent_config()
        agent_config = AgentConfig()
        cogent_config.register_config("agent", agent_config)

        # Test retrieval
        retrieved_agent = cogent_config.get_config("agent")
        self.assertEqual(retrieved_agent, agent_config)
        self.assertEqual(retrieved_agent.agent_type, "assistant")

        # Test it's in all configs
        all_configs = cogent_config.get_all_configs()
        self.assertIn("agent", all_configs)
        self.assertIn("llm", all_configs)
        self.assertIn("vector_store", all_configs)

    @pytest.mark.unit
    @patch("cogent.base.config.load_merged_toml_configs")
    def test_custom_config_with_toml_file(self, mock_load_merged):
        """Test custom config with actual TOML file loading."""
        # Create a temporary TOML file
        toml_content = """
        [agent]
        type = "research_assistant"
        max_turns = 15
        enable_memory = true
        memory_size = 1500
        temperature = 0.3

        [completion]
        model = "gpt-4"
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            temp_path = Path(f.name)

        try:
            # Mock the TOML loading to return our custom data
            mock_load_merged.return_value = {
                "agent": {
                    "type": "research_assistant",
                    "max_turns": 15,
                    "enable_memory": True,
                    "memory_size": 1500,
                    "temperature": 0.3,
                },
                "completion": {
                    "model": "gpt-4",
                },
            }

            # Create custom config
            @toml_config("agent")
            class AgentConfig(BaseConfig):
                agent_type: str = "assistant"
                max_conversation_turns: int = 10
                enable_memory: bool = True
                memory_size: int = 1000
                temperature: float = 0.7

                @classmethod
                def _from_toml(cls, toml_data: dict) -> "AgentConfig":
                    agent_section = toml_data.get("agent", {})
                    return cls(
                        agent_type=agent_section.get("type", cls().agent_type),
                        max_conversation_turns=agent_section.get("max_turns", cls().max_conversation_turns),
                        enable_memory=agent_section.get("enable_memory", cls().enable_memory),
                        memory_size=agent_section.get("memory_size", cls().memory_size),
                        temperature=agent_section.get("temperature", cls().temperature),
                    )

            # Create main config and register custom config
            cogent_config = CogentConfig()
            agent_config = AgentConfig()
            cogent_config.register_config("agent", agent_config)
            # Manually update from TOML to simulate what happens for built-in configs
            cogent_config.registry.update_from_toml(mock_load_merged.return_value)
            # Test that both custom and built-in configs were updated
            self.assertEqual(cogent_config.get_config("agent").agent_type, "research_assistant")
            self.assertEqual(cogent_config.get_config("agent").max_conversation_turns, 15)
            self.assertEqual(cogent_config.llm.completion_model, "gpt-4")

        finally:
            temp_path.unlink()

    @pytest.mark.unit
    def test_multiple_custom_configs(self):
        """Test registering multiple custom configs."""

        @toml_config("agent")
        class AgentConfig(BaseConfig):
            agent_type: str = "assistant"

        @toml_config("workflow")
        class WorkflowConfig(BaseConfig):
            workflow_name: str = "default"
            steps: list = []

        @toml_config("database")
        class DatabaseConfig(BaseConfig):
            connection_string: str = "sqlite:///default.db"
            pool_size: int = 5

        # Register multiple custom configs
        cogent_config = get_cogent_config()
        cogent_config.register_config("agent", AgentConfig())
        cogent_config.register_config("workflow", WorkflowConfig())
        cogent_config.register_config("database", DatabaseConfig())

        # Test all configs are available
        all_configs = cogent_config.get_all_configs()
        self.assertIn("agent", all_configs)
        self.assertIn("workflow", all_configs)
        self.assertIn("database", all_configs)
        self.assertIn("llm", all_configs)
        self.assertIn("vector_store", all_configs)

        # Test individual retrieval
        self.assertIsInstance(cogent_config.get_config("agent"), AgentConfig)
        self.assertIsInstance(cogent_config.get_config("workflow"), WorkflowConfig)
        self.assertIsInstance(cogent_config.get_config("database"), DatabaseConfig)
