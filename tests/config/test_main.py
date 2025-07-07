"""
Tests for main configuration module.
"""

import unittest
from unittest.mock import patch

import pytest

from cogentbase.config import (
    BaseConfig,
    CogentBaseConfig,
    get_cogent_config,
    toml_config,
)


class TestCogentBaseConfig(unittest.TestCase):
    """Test the CogentBaseConfig class."""

    @pytest.mark.unit
    @patch("cogentbase.config.main.load_toml_config")
    def test_default_values(self, mock_load_toml):
        """Test CogentBaseConfig default values."""
        mock_load_toml.return_value = {}
        config = CogentBaseConfig()

        self.assertIsInstance(config.llm, BaseConfig)
        self.assertIsInstance(config.vector_store, BaseConfig)
        self.assertIsInstance(config.reranker, BaseConfig)
        self.assertIsInstance(config.sensory, BaseConfig)

    @pytest.mark.unit
    @patch("cogentbase.config.main.load_toml_config")
    def test_load_toml_config_called(self, mock_load_toml):
        """Test that load_toml_config is called during initialization."""
        mock_load_toml.return_value = {}
        CogentBaseConfig()
        # Should be called twice: once for package defaults, once for user runtime
        self.assertEqual(mock_load_toml.call_count, 2)

    @pytest.mark.unit
    @patch("cogentbase.config.main.load_toml_config")
    def test_load_toml_config_with_data(self, mock_load_toml):
        """Test CogentBaseConfig with TOML data."""
        mock_load_toml.return_value = {
            "completion": {"model": "test_model"},
            "embedding": {"dimensions": 1024},
            "vector_store": {"provider": "test_provider"},
            "reranker": {"enable_reranker": True},
            "sensory": {"parser": {"chunk_size": 8000}},
        }
        config = CogentBaseConfig()
        # Check that configs were updated from TOML
        self.assertEqual(config.llm.completion_model, "test_model")
        self.assertEqual(config.llm.embedding_dimensions, 1024)
        self.assertEqual(config.vector_store.provider, "test_provider")
        self.assertTrue(config.reranker.enable_reranker)
        self.assertEqual(config.sensory.chunk_size, 8000)

    @pytest.mark.unit
    @patch("cogentbase.config.main.load_toml_config")
    def test_register_config(self, mock_load_toml):
        """Test registering a new configuration."""
        mock_load_toml.return_value = {}
        config = CogentBaseConfig()

        # Create a custom config
        @toml_config("custom_section")
        class CustomConfig(BaseConfig):
            value: str = "default"

        custom_config = CustomConfig()
        config.register_config("custom", custom_config)

        # Test retrieval
        retrieved = config.get_config("custom")
        self.assertEqual(retrieved, custom_config)

        # Test getting all configs
        all_configs = config.get_all_configs()
        self.assertIn("custom", all_configs)
        self.assertIn("llm", all_configs)
        self.assertIn("vector_store", all_configs)
        self.assertIn("reranker", all_configs)
        self.assertIn("sensory", all_configs)


class TestGetCogentConfig(unittest.TestCase):
    """Test the get_cogent_config function."""

    @pytest.mark.unit
    def test_get_cogent_config_returns_singleton(self):
        """Test that get_cogent_config returns the same instance."""
        config1 = get_cogent_config()
        config2 = get_cogent_config()
        self.assertIs(config1, config2)
