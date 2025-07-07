"""
Tests for CogentBaseConfig extensibility.
Demonstrates how to extend CogentBaseConfig with custom submodule configurations.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

from cogent.base.config import (
    BaseConfig,
    CogentBaseConfig,
    toml_config,
)
from cogent.base.rootdir import ROOT_DIR


@toml_config("agent")
class AgentConfig(BaseConfig):
    """Custom agent configuration for testing extensibility."""

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


@toml_config("workflow")
class WorkflowConfig(BaseConfig):
    """Custom workflow configuration for testing extensibility."""

    workflow_name: str = "default"
    steps: list = []
    max_retries: int = 3
    timeout: int = 300

    def get_toml_section(self) -> str:
        return "workflow"

    @classmethod
    def _from_toml(cls, toml_data: dict) -> "WorkflowConfig":
        """Custom TOML loading for WorkflowConfig."""
        workflow_section = toml_data.get("workflow", {})
        return cls(
            workflow_name=workflow_section.get("name", cls().workflow_name),
            steps=workflow_section.get("steps", cls().steps),
            max_retries=workflow_section.get("max_retries", cls().max_retries),
            timeout=workflow_section.get("timeout", cls().timeout),
        )


class CogentAgentConfig(CogentBaseConfig):
    """Extended configuration class that demonstrates CogentBaseConfig extensibility."""

    def _load_default_configs(self):
        """Override to load default submodule configurations plus custom ones."""
        # Load parent default configs
        super()._load_default_configs()

        # Register custom submodule configs
        self.register_config("agent", AgentConfig())
        self.register_config("workflow", WorkflowConfig())


class TestCogentBaseConfigExtensibility(unittest.TestCase):
    """Test CogentBaseConfig extensibility through CogentAgentConfig."""

    @pytest.mark.unit
    @patch("cogent.base.config.main.load_merged_toml_configs")
    def test_cogent_agent_config_creation(self, mock_load_merged):
        """Test creating CogentAgentConfig with custom submodule configs."""
        mock_load_merged.return_value = {}
        config = CogentAgentConfig()

        # Test that parent configs are loaded
        self.assertIsInstance(config.llm, BaseConfig)
        self.assertIsInstance(config.vector_store, BaseConfig)
        self.assertIsInstance(config.reranker, BaseConfig)
        self.assertIsInstance(config.sensory, BaseConfig)

        # Test that custom configs are loaded
        self.assertIsInstance(config.get_config("agent"), AgentConfig)
        self.assertIsInstance(config.get_config("workflow"), WorkflowConfig)

    @pytest.mark.unit
    @patch("cogent.base.config.main.load_merged_toml_configs")
    def test_cogent_agent_config_default_values(self, mock_load_merged):
        """Test CogentAgentConfig default values for custom configs."""
        mock_load_merged.return_value = {}
        config = CogentAgentConfig()

        # Test agent config defaults
        agent_config = config.get_config("agent")
        self.assertEqual(agent_config.agent_type, "assistant")
        self.assertEqual(agent_config.max_conversation_turns, 10)
        self.assertTrue(agent_config.enable_memory)
        self.assertEqual(agent_config.memory_size, 1000)
        self.assertEqual(agent_config.temperature, 0.7)

        # Test workflow config defaults
        workflow_config = config.get_config("workflow")
        self.assertEqual(workflow_config.workflow_name, "default")
        self.assertEqual(workflow_config.steps, [])
        self.assertEqual(workflow_config.max_retries, 3)
        self.assertEqual(workflow_config.timeout, 300)

    @pytest.mark.unit
    @patch("cogent.base.config.main.load_merged_toml_configs")
    def test_cogent_agent_config_with_toml_data(self, mock_load_merged):
        """Test CogentAgentConfig with TOML data for custom configs."""
        mock_load_merged.return_value = {
            "agent": {
                "type": "specialist",
                "max_turns": 20,
                "enable_memory": False,
                "memory_size": 2000,
                "temperature": 0.5,
            },
            "workflow": {
                "name": "custom_workflow",
                "steps": ["step1", "step2"],
                "max_retries": 5,
                "timeout": 600,
            },
            "completion": {"model": "test_model"},
        }
        config = CogentAgentConfig()

        # Test that custom configs were updated from TOML
        agent_config = config.get_config("agent")
        self.assertEqual(agent_config.agent_type, "specialist")
        self.assertEqual(agent_config.max_conversation_turns, 20)
        self.assertFalse(agent_config.enable_memory)
        self.assertEqual(agent_config.memory_size, 2000)
        self.assertEqual(agent_config.temperature, 0.5)

        workflow_config = config.get_config("workflow")
        self.assertEqual(workflow_config.workflow_name, "custom_workflow")
        self.assertEqual(workflow_config.steps, ["step1", "step2"])
        self.assertEqual(workflow_config.max_retries, 5)
        self.assertEqual(workflow_config.timeout, 600)

        # Test that parent configs were also updated
        self.assertEqual(config.llm.completion_model, "test_model")

    @pytest.mark.unit
    @patch("cogent.base.config.main.load_merged_toml_configs")
    def test_cogent_agent_config_register_config(self, mock_load_merged):
        """Test registering additional configs with CogentAgentConfig."""
        mock_load_merged.return_value = {}
        config = CogentAgentConfig()

        # Create and register an additional custom config
        @toml_config("database")
        class DatabaseConfig(BaseConfig):
            connection_string: str = "sqlite:///default.db"
            pool_size: int = 5

        database_config = DatabaseConfig()
        config.register_config("database", database_config)

        # Test retrieval
        retrieved_db = config.get_config("database")
        self.assertEqual(retrieved_db, database_config)
        self.assertEqual(retrieved_db.connection_string, "sqlite:///default.db")
        self.assertEqual(retrieved_db.pool_size, 5)

    @pytest.mark.unit
    @patch("cogent.base.config.main.load_merged_toml_configs")
    def test_cogent_agent_config_get_all_configs(self, mock_load_merged):
        """Test getting all configs from CogentAgentConfig."""
        mock_load_merged.return_value = {}
        config = CogentAgentConfig()

        all_configs = config.get_all_configs()

        # Test that all expected configs are present
        expected_configs = ["llm", "vector_store", "reranker", "sensory", "agent", "workflow"]
        for config_name in expected_configs:
            self.assertIn(config_name, all_configs)
            self.assertIsInstance(all_configs[config_name], BaseConfig)

        # Test that custom configs are the right types
        self.assertIsInstance(all_configs["agent"], AgentConfig)
        self.assertIsInstance(all_configs["workflow"], WorkflowConfig)

    @pytest.mark.unit
    @patch("cogent.base.config.main.load_merged_toml_configs")
    def test_cogent_agent_config_get_config_nonexistent(self, mock_load_merged):
        """Test getting nonexistent config from CogentAgentConfig."""
        mock_load_merged.return_value = {}
        config = CogentAgentConfig()

        # Test getting nonexistent config
        nonexistent = config.get_config("nonexistent")
        self.assertIsNone(nonexistent)

    @pytest.mark.unit
    @patch("cogent.base.config.main.load_merged_toml_configs")
    def test_cogent_agent_config_convenience_properties(self, mock_load_merged):
        """Test convenience properties of CogentAgentConfig."""
        mock_load_merged.return_value = {}
        config = CogentAgentConfig()

        # Test parent convenience properties still work
        self.assertIsInstance(config.llm, BaseConfig)
        self.assertIsInstance(config.vector_store, BaseConfig)
        self.assertIsInstance(config.reranker, BaseConfig)
        self.assertIsInstance(config.sensory, BaseConfig)

        # Test that custom configs are accessible via get_config
        self.assertIsInstance(config.get_config("agent"), AgentConfig)
        self.assertIsInstance(config.get_config("workflow"), WorkflowConfig)

    @pytest.mark.unit
    @patch("cogent.base.config.main.load_merged_toml_configs")
    def test_cogent_agent_config_paths(self, mock_load_merged):
        """Test that config paths are set correctly in CogentAgentConfig."""
        mock_load_merged.return_value = {}
        config = CogentAgentConfig()

        self.assertEqual(config.base_toml, ROOT_DIR / "config" / "base.toml")
        self.assertEqual(config.config_dir, ROOT_DIR / "config")

    @pytest.mark.unit
    @patch("cogent.base.config.main.load_merged_toml_configs")
    def test_cogent_agent_config_environment_variables(self, mock_load_merged):
        """Test environment variable handling in CogentAgentConfig."""
        mock_load_merged.return_value = {}

        # Test default environment values
        config = CogentAgentConfig()
        self.assertEqual(config.env, "development")
        self.assertFalse(config.debug)

    @pytest.mark.unit
    @patch("cogent.base.config.main.load_merged_toml_configs")
    def test_cogent_agent_config_custom_config_dir(self, mock_load_merged):
        """Test CogentAgentConfig with custom config directory."""
        mock_load_merged.return_value = {}

        # Test default config directory
        config_default = CogentAgentConfig()
        self.assertEqual(config_default.config_dir, ROOT_DIR / "config")
        self.assertEqual(config_default.base_toml, ROOT_DIR / "config" / "base.toml")

        # Test custom config directory - the config_dir should be preserved
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_config_dir = Path(temp_dir)
            config = CogentAgentConfig(config_dir=custom_config_dir)

            # The base_toml should be set to the custom directory
            self.assertEqual(config.base_toml, custom_config_dir / "base.toml")

    @pytest.mark.unit
    @patch("cogent.base.config.main.load_merged_toml_configs")
    def test_cogent_agent_config_toml_loading_called(self, mock_load_merged):
        """Test that TOML loading is called during CogentAgentConfig initialization."""
        mock_load_merged.return_value = {}
        CogentAgentConfig()
        mock_load_merged.assert_called_once()

    @pytest.mark.unit
    @patch("cogent.base.config.main.load_merged_toml_configs")
    def test_cogent_agent_config_multiple_custom_configs(self, mock_load_merged):
        """Test CogentAgentConfig with multiple custom configs."""
        mock_load_merged.return_value = {}
        config = CogentAgentConfig()

        # Create additional custom configs
        @toml_config("monitoring")
        class MonitoringConfig(BaseConfig):
            log_level: str = "INFO"
            metrics_enabled: bool = True

        @toml_config("security")
        class SecurityConfig(BaseConfig):
            encryption_enabled: bool = True
            key_rotation_days: int = 30

        # Register additional configs
        config.register_config("monitoring", MonitoringConfig())
        config.register_config("security", SecurityConfig())

        # Test all configs are present
        all_configs = config.get_all_configs()
        expected_configs = ["llm", "vector_store", "reranker", "sensory", "agent", "workflow", "monitoring", "security"]

        for config_name in expected_configs:
            self.assertIn(config_name, all_configs)

        # Test specific configs
        monitoring = config.get_config("monitoring")
        self.assertEqual(monitoring.log_level, "INFO")
        self.assertTrue(monitoring.metrics_enabled)

        security = config.get_config("security")
        self.assertTrue(security.encryption_enabled)
        self.assertEqual(security.key_rotation_days, 30)

    @pytest.mark.unit
    @patch("cogent.base.config.main.load_merged_toml_configs")
    def test_cogent_agent_config_inheritance_structure(self, mock_load_merged):
        """Test that CogentAgentConfig properly inherits from CogentBaseConfig."""
        mock_load_merged.return_value = {}
        config = CogentAgentConfig()

        # Test inheritance
        self.assertIsInstance(config, CogentBaseConfig)

        # Test that all parent methods are available
        self.assertTrue(hasattr(config, "register_config"))
        self.assertTrue(hasattr(config, "get_config"))
        self.assertTrue(hasattr(config, "get_all_configs"))
        self.assertTrue(hasattr(config, "llm"))
        self.assertTrue(hasattr(config, "vector_store"))
        self.assertTrue(hasattr(config, "reranker"))
        self.assertTrue(hasattr(config, "sensory"))
        self.assertTrue(hasattr(config, "base_toml"))
        self.assertTrue(hasattr(config, "config_dir"))
        self.assertTrue(hasattr(config, "env"))
        self.assertTrue(hasattr(config, "debug"))
