"""
Tests for main configuration module.
"""

import unittest
from unittest.mock import patch

import pytest

from cogent.base.config import BaseConfig, CogentConfig, get_cogent_config, toml_config
from cogent.base.rootdir import ROOT_DIR


class TestCogentConfig(unittest.TestCase):
    """Test the CogentConfig class."""

    @pytest.mark.unit
    @patch("cogent.base.config.load_merged_toml_configs")
    def test_default_values(self, mock_load_merged):
        """Test CogentConfig default values."""
        mock_load_merged.return_value = {}
        config = CogentConfig()

        self.assertEqual(config.env, "development")
        self.assertFalse(config.debug)
        self.assertIsInstance(config.llm, BaseConfig)
        self.assertIsInstance(config.vector_store, BaseConfig)
        self.assertIsInstance(config.reranker, BaseConfig)
        self.assertIsInstance(config.sensory, BaseConfig)

    @pytest.mark.unit
    @patch("cogent.base.config.main.load_merged_toml_configs")
    def test_load_merged_toml_configs_called(self, mock_load_merged):
        """Test that load_merged_toml_configs is called during initialization."""
        mock_load_merged.return_value = {}
        CogentConfig()
        mock_load_merged.assert_called_once()

    @pytest.mark.unit
    @patch("cogent.base.config.main.load_merged_toml_configs")
    def test_load_merged_toml_configs_with_data(self, mock_load_merged):
        """Test CogentConfig with TOML data."""
        mock_load_merged.return_value = {
            "completion": {"model": "test_model"},
            "embedding": {"dimensions": 1024},
            "vector_store": {"provider": "test_provider"},
            "reranker": {"enable_reranker": True},
            "sensory": {"parser": {"chunk_size": 8000}},
        }
        config = CogentConfig()
        # Check that configs were updated from TOML
        self.assertEqual(config.llm.completion_model, "test_model")
        self.assertEqual(config.llm.embedding_dimensions, 1024)
        self.assertEqual(config.vector_store.provider, "test_provider")
        self.assertTrue(config.reranker.enable_reranker)
        self.assertEqual(config.sensory.chunk_size, 8000)

    @pytest.mark.unit
    @patch("cogent.base.config.load_merged_toml_configs")
    def test_config_paths(self, mock_load_merged):
        """Test that config paths are set correctly."""
        mock_load_merged.return_value = {}
        config = CogentConfig()

        self.assertEqual(config.base_toml, ROOT_DIR / "config" / "base.toml")
        self.assertEqual(config.providers_toml, ROOT_DIR / "config" / "providers.toml")
        self.assertEqual(config.sensory_toml, ROOT_DIR / "config" / "sensory.toml")

    @pytest.mark.unit
    @patch("cogent.base.config.load_merged_toml_configs")
    def test_register_config(self, mock_load_merged):
        """Test registering a new configuration."""
        mock_load_merged.return_value = {}
        config = CogentConfig()

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
