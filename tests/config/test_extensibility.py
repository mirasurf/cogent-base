"""
Tests for configuration extensibility and config loading order.
Demonstrates how users can add custom submodule configs and how config precedence works.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

from cogentbase.config import (
    BaseConfig,
    CogentBaseConfig,
    get_cogent_config,
    toml_config,
)

# --- Custom Config Classes for Extensibility ---


@toml_config("agent")
class AgentConfig(BaseConfig):
    agent_type: str = "assistant"
    max_conversation_turns: int = 10
    enable_memory: bool = True
    memory_size: int = 1000
    temperature: float = 0.7

    def get_toml_section(self) -> str:
        return "agent"

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


@toml_config("workflow")
class WorkflowConfig(BaseConfig):
    workflow_name: str = "default"
    steps: list = []
    max_retries: int = 3
    timeout: int = 300

    def get_toml_section(self) -> str:
        return "workflow"

    @classmethod
    def _from_toml(cls, toml_data: dict) -> "WorkflowConfig":
        workflow_section = toml_data.get("workflow", {})
        return cls(
            workflow_name=workflow_section.get("name", cls().workflow_name),
            steps=workflow_section.get("steps", cls().steps),
            max_retries=workflow_section.get("max_retries", cls().max_retries),
            timeout=workflow_section.get("timeout", cls().timeout),
        )


# --- Extensible Cogent Config Example ---


class CogentAgentConfig(CogentBaseConfig):
    def _load_default_configs(self):
        super()._load_default_configs()
        self.register_config("agent", AgentConfig())
        self.register_config("workflow", WorkflowConfig())


# --- Main Test Class ---


class TestExtensibility(unittest.TestCase):
    """Test configuration extensibility and config loading order."""

    @pytest.mark.unit
    def test_custom_agent_config_creation(self):
        config = AgentConfig()
        self.assertEqual(config.agent_type, "assistant")
        self.assertEqual(config.max_conversation_turns, 10)
        self.assertTrue(config.enable_memory)
        self.assertEqual(config.memory_size, 1000)
        self.assertEqual(config.temperature, 0.7)
        self.assertTrue(hasattr(AgentConfig, "from_toml"))

    @pytest.mark.unit
    def test_custom_agent_config_from_toml(self):
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
    @patch("cogentbase.config.main.load_toml_config")
    def test_register_custom_config_with_cogent(self, mock_load_toml):
        mock_load_toml.return_value = {}
        cogent_config = CogentBaseConfig()
        agent_config = AgentConfig()
        cogent_config.register_config("agent", agent_config)
        retrieved_agent = cogent_config.get_config("agent")
        self.assertEqual(retrieved_agent, agent_config)
        self.assertEqual(retrieved_agent.agent_type, "assistant")
        all_configs = cogent_config.get_all_configs()
        self.assertIn("agent", all_configs)
        self.assertIn("llm", all_configs)
        self.assertIn("vector_store", all_configs)

    @pytest.mark.unit
    @patch("cogentbase.config.main.load_toml_config")
    def test_custom_config_with_toml_file(self, mock_load_toml):
        mock_load_toml.return_value = {
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
        cogent_config = CogentBaseConfig()
        agent_config = AgentConfig()
        cogent_config.register_config("agent", agent_config)
        cogent_config.registry.update_from_toml(mock_load_toml.return_value)
        self.assertEqual(cogent_config.get_config("agent").agent_type, "research_assistant")
        self.assertEqual(cogent_config.get_config("agent").max_conversation_turns, 15)
        self.assertEqual(cogent_config.llm.completion_model, "gpt-4")

    @pytest.mark.unit
    def test_multiple_custom_configs(self):
        @toml_config("database")
        class DatabaseConfig(BaseConfig):
            connection_string: str = "sqlite:///default.db"
            pool_size: int = 5

        cogent_config = get_cogent_config()
        cogent_config.register_config("agent", AgentConfig())
        cogent_config.register_config("workflow", WorkflowConfig())
        cogent_config.register_config("database", DatabaseConfig())
        all_configs = cogent_config.get_all_configs()
        self.assertIn("agent", all_configs)
        self.assertIn("workflow", all_configs)
        self.assertIn("database", all_configs)
        self.assertIn("llm", all_configs)
        self.assertIn("vector_store", all_configs)
        self.assertIsInstance(cogent_config.get_config("agent"), AgentConfig)
        self.assertIsInstance(cogent_config.get_config("workflow"), WorkflowConfig)
        self.assertIsInstance(cogent_config.get_config("database"), DatabaseConfig)

    # --- Extensibility and Precedence Tests from test_cogent_base_config_extensibility.py ---

    @pytest.mark.unit
    @patch("cogentbase.config.main.load_toml_config")
    def test_cogent_agent_config_creation(self, mock_load_toml):
        mock_load_toml.return_value = {}
        config = CogentAgentConfig()
        self.assertIsInstance(config.llm, BaseConfig)
        self.assertIsInstance(config.vector_store, BaseConfig)
        self.assertIsInstance(config.reranker, BaseConfig)
        self.assertIsInstance(config.sensory, BaseConfig)
        self.assertIsInstance(config.get_config("agent"), AgentConfig)
        self.assertIsInstance(config.get_config("workflow"), WorkflowConfig)

    @pytest.mark.unit
    @patch("cogentbase.config.main.load_toml_config")
    def test_cogent_agent_config_default_values(self, mock_load_toml):
        mock_load_toml.return_value = {}
        config = CogentAgentConfig()
        agent_config = config.get_config("agent")
        self.assertEqual(agent_config.agent_type, "assistant")
        self.assertEqual(agent_config.max_conversation_turns, 10)
        self.assertTrue(agent_config.enable_memory)
        self.assertEqual(agent_config.memory_size, 1000)
        self.assertEqual(agent_config.temperature, 0.7)
        workflow_config = config.get_config("workflow")
        self.assertEqual(workflow_config.workflow_name, "default")
        self.assertEqual(workflow_config.steps, [])
        self.assertEqual(workflow_config.max_retries, 3)
        self.assertEqual(workflow_config.timeout, 300)

    @pytest.mark.unit
    @patch("cogentbase.config.main.load_toml_config")
    def test_cogent_agent_config_with_toml_data(self, mock_load_toml):
        mock_load_toml.return_value = {
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
        self.assertEqual(config.llm.completion_model, "test_model")

    @pytest.mark.unit
    @patch("cogentbase.config.main.load_toml_config")
    def test_cogent_agent_config_register_config(self, mock_load_toml):
        mock_load_toml.return_value = {}
        config = CogentAgentConfig()

        @toml_config("database")
        class DatabaseConfig(BaseConfig):
            connection_string: str = "sqlite:///default.db"
            pool_size: int = 5

        database_config = DatabaseConfig()
        config.register_config("database", database_config)
        retrieved_db = config.get_config("database")
        self.assertEqual(retrieved_db, database_config)
        self.assertEqual(retrieved_db.connection_string, "sqlite:///default.db")
        self.assertEqual(retrieved_db.pool_size, 5)
        all_configs = config.get_all_configs()
        self.assertIn("database", all_configs)
        self.assertIn("llm", all_configs)
        self.assertIn("vector_store", all_configs)
        self.assertIn("reranker", all_configs)
        self.assertIn("sensory", all_configs)

    @pytest.mark.unit
    @patch("cogentbase.config.main.load_toml_config")
    def test_cogent_agent_config_get_all_configs(self, mock_load_toml):
        mock_load_toml.return_value = {}
        config = CogentAgentConfig()
        all_configs = config.get_all_configs()
        expected_configs = ["llm", "vector_store", "reranker", "sensory", "agent", "workflow"]
        for config_name in expected_configs:
            self.assertIn(config_name, all_configs)
            self.assertIsInstance(all_configs[config_name], BaseConfig)
        self.assertIsInstance(all_configs["agent"], AgentConfig)
        self.assertIsInstance(all_configs["workflow"], WorkflowConfig)

    @pytest.mark.unit
    @patch("cogentbase.config.main.load_toml_config")
    def test_cogent_agent_config_get_config_nonexistent(self, mock_load_toml):
        mock_load_toml.return_value = {}
        config = CogentAgentConfig()
        nonexistent = config.get_config("nonexistent")
        self.assertIsNone(nonexistent)

    @pytest.mark.unit
    @patch("cogentbase.config.main.load_toml_config")
    def test_cogent_agent_config_convenience_properties(self, mock_load_toml):
        mock_load_toml.return_value = {}
        config = CogentAgentConfig()
        self.assertIsInstance(config.llm, BaseConfig)
        self.assertIsInstance(config.vector_store, BaseConfig)
        self.assertIsInstance(config.reranker, BaseConfig)
        self.assertIsInstance(config.sensory, BaseConfig)
        self.assertIsInstance(config.get_config("agent"), AgentConfig)
        self.assertIsInstance(config.get_config("workflow"), WorkflowConfig)

    @pytest.mark.unit
    @patch("cogentbase.config.main.load_toml_config")
    def test_cogent_agent_config_toml_loading_called(self, mock_load_toml):
        mock_load_toml.return_value = {}
        CogentAgentConfig()
        self.assertEqual(mock_load_toml.call_count, 2)

    @pytest.mark.unit
    @patch("cogentbase.config.main.load_toml_config")
    def test_cogent_agent_config_multiple_custom_configs(self, mock_load_toml):
        mock_load_toml.return_value = {}
        config = CogentAgentConfig()

        @toml_config("monitoring")
        class MonitoringConfig(BaseConfig):
            log_level: str = "INFO"
            metrics_enabled: bool = True

        @toml_config("security")
        class SecurityConfig(BaseConfig):
            encryption_enabled: bool = True
            key_rotation_days: int = 30

        config.register_config("monitoring", MonitoringConfig())
        config.register_config("security", SecurityConfig())
        all_configs = config.get_all_configs()
        expected_configs = ["llm", "vector_store", "reranker", "sensory", "agent", "workflow", "monitoring", "security"]
        for config_name in expected_configs:
            self.assertIn(config_name, all_configs)
        monitoring = config.get_config("monitoring")
        self.assertEqual(monitoring.log_level, "INFO")
        self.assertTrue(monitoring.metrics_enabled)
        security = config.get_config("security")
        self.assertTrue(security.encryption_enabled)
        self.assertEqual(security.key_rotation_days, 30)

    @pytest.mark.unit
    @patch("cogentbase.config.main.load_toml_config")
    def test_cogent_agent_config_inheritance_structure(self, mock_load_toml):
        mock_load_toml.return_value = {}
        config = CogentAgentConfig()
        self.assertIsInstance(config, CogentBaseConfig)
        self.assertTrue(hasattr(config, "register_config"))
        self.assertTrue(hasattr(config, "get_config"))
        self.assertTrue(hasattr(config, "get_all_configs"))
        self.assertTrue(hasattr(config, "llm"))
        self.assertTrue(hasattr(config, "vector_store"))
        self.assertTrue(hasattr(config, "reranker"))
        self.assertTrue(hasattr(config, "sensory"))

    # --- Config Loading Order and Precedence Tests ---

    @pytest.mark.unit
    def test_user_base_toml_overrides_package(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            user_toml = temp_dir_path / "base.toml"
            user_toml.write_text(
                """
[completion]
provider = "user-provider"
model = "user-model"

[embedding]
dimensions = 1234
                """
            )
            old_cwd = os.getcwd()
            os.chdir(temp_dir)
            try:
                config = CogentBaseConfig()
                self.assertEqual(config.llm.completion_provider, "user-provider")
                self.assertEqual(config.llm.completion_model, "user-model")
                self.assertEqual(config.llm.embedding_dimensions, 1234)
            finally:
                os.chdir(old_cwd)

    @pytest.mark.unit
    def test_package_base_toml_used_when_no_user_toml(self):
        user_toml = Path.cwd() / "base.toml"
        if user_toml.exists():
            user_toml.unlink()
        config = CogentBaseConfig()
        self.assertEqual(config.llm.completion_provider, "litellm")
        self.assertEqual(config.llm.completion_model, "ollama_qwen_vision")
        self.assertEqual(config.llm.embedding_dimensions, 768)

    @pytest.mark.unit
    @patch("cogentbase.config.main.load_toml_config")
    def test_class_defaults_used_if_not_in_toml(self, mock_load_toml):
        mock_load_toml.return_value = {}
        config = CogentBaseConfig()
        self.assertEqual(config.llm.completion_provider, "litellm")
        self.assertEqual(config.llm.completion_model, "openai_gpt4-1-mini")
        self.assertEqual(config.llm.embedding_dimensions, 768)

    @pytest.mark.unit
    def test_class_defaults_vs_package_defaults(self):
        from cogentbase.config.core import LLMConfig

        llm_config = LLMConfig()
        self.assertEqual(llm_config.completion_model, "openai_gpt4-1-mini")
        self.assertEqual(llm_config.embedding_dimensions, 768)
        config = CogentBaseConfig()
        self.assertEqual(config.llm.completion_model, "ollama_qwen_vision")
        self.assertEqual(config.llm.embedding_dimensions, 768)
