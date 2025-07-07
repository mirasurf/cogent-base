"""
Unit tests for the configuration module.
Tests all configuration classes and their methods including TOML loading,
environment variable priority, and error handling.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, mock_open
from typing import Dict, Any
import pytest

# Import the config module
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from cogent.base.rootdir import ROOT_DIR
from cogent.base.config import (
    LLMConfig,
    VectorStoreConfig,
    RerankerConfig,
    SensoryConfig,
    CogentConfig,
    load_toml_config,
    load_merged_toml_configs,
    get_config,
)


class TestLoadTomlConfig(unittest.TestCase):
    """Test the load_toml_config function."""

    @pytest.mark.unit
    def test_load_valid_toml(self):
        """Test loading a valid TOML file."""
        toml_content = """
        [registered_models]
        openai_gpt4 = { model_name = "gpt-4" }

        [completion]
        model = "openai_gpt4"
        default_max_tokens = 1000

        [embedding]
        model = "text-embedding-3-small"
        dimensions = 1536
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            temp_path = Path(f.name)

        try:
            result = load_toml_config(temp_path)
            self.assertIn("registered_models", result)
            self.assertIn("completion", result)
            self.assertIn("embedding", result)
            self.assertEqual(result["registered_models"]["openai_gpt4"]["model_name"], "gpt-4")
        finally:
            temp_path.unlink()

    @pytest.mark.unit
    def test_load_nonexistent_file(self):
        """Test loading a non-existent TOML file."""
        result = load_toml_config(Path("/nonexistent/file.toml"))
        self.assertEqual(result, {})

    @pytest.mark.unit
    def test_load_invalid_toml(self):
        """Test loading an invalid TOML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("invalid toml content [")
            temp_path = Path(f.name)

        try:
            result = load_toml_config(temp_path)
            self.assertEqual(result, {})
        finally:
            temp_path.unlink()


class TestLoadMergedTomlConfigs(unittest.TestCase):
    """Test the load_merged_toml_configs function."""

    @pytest.mark.unit
    def test_merge_multiple_toml_files(self):
        """Test merging multiple TOML files."""
        base_toml = """
        [sensory.parser]
        chunk_size = 6000
        chunk_overlap = 300

        [graph]
        model = "ollama_qwen_vision"
        enable_entity_resolution = true
        """

        providers_toml = """
        [registered_models]
        openai_gpt4 = { model_name = "gpt-4" }

        [completion]
        model = "openai_gpt4"
        default_max_tokens = 1000

        [embedding]
        model = "text-embedding-3-small"
        dimensions = 1536
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f1:
            f1.write(base_toml)
            base_path = Path(f1.name)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f2:
            f2.write(providers_toml)
            providers_path = Path(f2.name)

        try:
            result = load_merged_toml_configs([base_path, providers_path])

            # Check that both files are merged
            self.assertIn("sensory", result)
            self.assertIn("graph", result)
            self.assertIn("registered_models", result)
            self.assertIn("completion", result)
            self.assertIn("embedding", result)

            # Check specific values
            self.assertEqual(result["sensory"]["parser"]["chunk_size"], 6000)
            self.assertEqual(result["graph"]["model"], "ollama_qwen_vision")
            self.assertEqual(result["registered_models"]["openai_gpt4"]["model_name"], "gpt-4")
            self.assertEqual(result["completion"]["model"], "openai_gpt4")
            self.assertEqual(result["embedding"]["dimensions"], 1536)
        finally:
            base_path.unlink()
            providers_path.unlink()

    @pytest.mark.unit
    def test_merge_with_overlapping_keys(self):
        """Test merging TOML files with overlapping keys."""
        file1_content = """
        [section]
        key1 = "value1"
        key2 = "value2"
        """

        file2_content = """
        [section]
        key2 = "overwritten"
        key3 = "value3"
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f1:
            f1.write(file1_content)
            path1 = Path(f1.name)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f2:
            f2.write(file2_content)
            path2 = Path(f2.name)

        try:
            result = load_merged_toml_configs([path1, path2])

            # Later file should overwrite earlier file's values
            self.assertEqual(result["section"]["key1"], "value1")
            self.assertEqual(result["section"]["key2"], "overwritten")
            self.assertEqual(result["section"]["key3"], "value3")
        finally:
            path1.unlink()
            path2.unlink()

    @pytest.mark.unit
    def test_merge_with_nonexistent_files(self):
        """Test merging with some non-existent files."""
        valid_toml = """
        [test]
        key = "value"
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(valid_toml)
            valid_path = Path(f.name)

        try:
            nonexistent_path = Path("/nonexistent/file.toml")
            result = load_merged_toml_configs([valid_path, nonexistent_path])

            # Should still load the valid file
            self.assertIn("test", result)
            self.assertEqual(result["test"]["key"], "value")
        finally:
            valid_path.unlink()


class TestLLMConfig(unittest.TestCase):
    """Test the LLMConfig class."""

    @pytest.mark.unit
    def test_default_values(self):
        """Test default values."""
        config = LLMConfig()
        self.assertEqual(config.registered_models, {})
        self.assertEqual(config.completion_provider, "litellm")
        self.assertEqual(config.completion_model, "openai_gpt4-1-mini")
        self.assertEqual(config.completion_max_tokens, 2000)
        self.assertEqual(config.completion_temperature, 0.7)
        self.assertEqual(config.embedding_provider, "litellm")
        self.assertEqual(config.embedding_model, "openai_embedding")
        self.assertEqual(config.embedding_dimensions, 768)
        self.assertEqual(config.embedding_similarity_metric, "cosine")
        self.assertEqual(config.embedding_batch_size, 100)

    @pytest.mark.unit
    def test_from_toml_empty_data(self):
        """Test from_toml with empty data."""
        config = LLMConfig.from_toml({})
        self.assertEqual(config.registered_models, {})
        self.assertEqual(config.completion_provider, "litellm")  # Default value

    @pytest.mark.unit
    def test_from_toml_with_data(self):
        """Test from_toml with valid data."""
        toml_data = {
            "registered_models": {"openai_gpt4": {"model_name": "gpt-4"}, "claude": {"model_name": "claude-3"}},
            "completion": {
                "provider": "openai",
                "model": "gpt-4",
                "default_max_tokens": "1500",
                "default_temperature": "0.5",
            },
            "embedding": {
                "provider": "openai",
                "model": "text-embedding-3-large",
                "dimensions": "3072",
                "similarity_metric": "dot_product",
                "batch_size": "200",
            },
        }

        config = LLMConfig.from_toml(toml_data)

        self.assertEqual(config.registered_models, toml_data["registered_models"])
        self.assertEqual(config.completion_provider, "openai")
        self.assertEqual(config.completion_model, "gpt-4")
        self.assertEqual(config.completion_max_tokens, 1500)
        self.assertEqual(config.completion_temperature, 0.5)
        self.assertEqual(config.embedding_provider, "openai")
        self.assertEqual(config.embedding_model, "text-embedding-3-large")
        self.assertEqual(config.embedding_dimensions, 3072)
        self.assertEqual(config.embedding_similarity_metric, "dot_product")
        self.assertEqual(config.embedding_batch_size, 200)

    @pytest.mark.unit
    def test_from_toml_invalid_numeric_values(self):
        """Test from_toml with invalid numeric values."""
        toml_data = {
            "completion": {"default_max_tokens": "invalid_number", "default_temperature": "not_a_float"},
            "embedding": {"dimensions": "invalid_dimensions", "batch_size": "not_a_number"},
        }

        config = LLMConfig.from_toml(toml_data)

        # Should use default values when TOML values are invalid
        self.assertEqual(config.completion_max_tokens, 2000)
        self.assertEqual(config.completion_temperature, 0.7)
        self.assertEqual(config.embedding_dimensions, 768)
        self.assertEqual(config.embedding_batch_size, 100)


class TestVectorStoreConfig(unittest.TestCase):
    """Test the VectorStoreConfig class."""

    @pytest.mark.unit
    def test_default_values(self):
        """Test default values."""
        config = VectorStoreConfig()
        self.assertEqual(config.registered_vector_stores, {})
        self.assertEqual(config.provider, "pgvector")
        self.assertEqual(config.collection_name, "cogent")
        self.assertEqual(config.embedding_model_dims, 768)

    @pytest.mark.unit
    def test_from_toml_empty_data(self):
        """Test from_toml with empty data."""
        config = VectorStoreConfig.from_toml({})
        self.assertEqual(config.registered_vector_stores, {})
        self.assertEqual(config.provider, "pgvector")  # Default value

    @pytest.mark.unit
    def test_from_toml_with_data(self):
        """Test from_toml with valid data."""
        toml_data = {
            "registered_vector_stores": {
                "pgvector": {"provider": "pgvector", "collection_name": "test_collection", "embedding_model_dims": 768},
                "weaviate": {
                    "provider": "weaviate",
                    "collection_name": "weaviate_collection",
                    "embedding_model_dims": 1536,
                },
            },
            "vector_store": {
                "provider": "weaviate",
                "collection_name": "main_collection",
                "embedding_model_dims": "1024",
            },
        }

        config = VectorStoreConfig.from_toml(toml_data)

        self.assertEqual(config.registered_vector_stores, toml_data["registered_vector_stores"])
        self.assertEqual(config.provider, "weaviate")
        self.assertEqual(config.collection_name, "main_collection")
        self.assertEqual(config.embedding_model_dims, 1024)

    @pytest.mark.unit
    def test_from_toml_invalid_numeric_values(self):
        """Test from_toml with invalid numeric values."""
        toml_data = {"vector_store": {"embedding_model_dims": "invalid_number"}}

        config = VectorStoreConfig.from_toml(toml_data)

        # Should use default value when TOML value is invalid
        self.assertEqual(config.embedding_model_dims, 768)


class TestRerankerConfig(unittest.TestCase):
    """Test the RerankerConfig class."""

    @pytest.mark.unit
    def test_reranker_config_from_toml_full(self):
        toml_data = {
            "registered_rerankers": {"foo": {"model_name": "bar"}},
            "reranker": {
                "enable_reranker": True,
                "provider": "ollama",
                "model_name": "ollama_reranker"
            }
        }
        cfg = RerankerConfig.from_toml(toml_data)
        self.assertEqual(cfg.registered_rerankers, {"foo": {"model_name": "bar"}})
        self.assertTrue(cfg.enable_reranker)
        self.assertEqual(cfg.reranker_provider, "ollama")
        self.assertEqual(cfg.reranker_model, "ollama_reranker")

    @pytest.mark.unit
    def test_reranker_config_from_toml_defaults(self):
        toml_data = {}
        cfg = RerankerConfig.from_toml(toml_data)
        self.assertEqual(cfg.registered_rerankers, {})
        self.assertFalse(cfg.enable_reranker)
        self.assertEqual(cfg.reranker_provider, "litellm")
        self.assertEqual(cfg.reranker_model, "ollama_reranker")

    @pytest.mark.unit
    def test_reranker_config_from_toml_partial(self):
        toml_data = {"reranker": {"provider": "flag"}}
        cfg = RerankerConfig.from_toml(toml_data)
        self.assertEqual(cfg.reranker_provider, "flag")
        self.assertFalse(cfg.enable_reranker)
        self.assertEqual(cfg.reranker_model, "ollama_reranker")


class TestSensoryConfig(unittest.TestCase):
    """Test the SensoryConfig class."""

    @pytest.mark.unit
    def test_default_values(self):
        """Test default values."""
        config = SensoryConfig()
        self.assertEqual(config.chunk_size, 6000)
        self.assertEqual(config.chunk_overlap, 300)
        self.assertFalse(config.use_unstructured_api)
        self.assertFalse(config.use_contextual_chunking)
        self.assertEqual(config.contextual_chunking_model, "ollama_qwen_vision")

    @pytest.mark.unit
    def test_custom_values(self):
        """Test custom values."""
        config = SensoryConfig(
            chunk_size=8000,
            chunk_overlap=500,
            use_unstructured_api=True,
            use_contextual_chunking=True,
            contextual_chunking_model="custom_model"
        )
        self.assertEqual(config.chunk_size, 8000)
        self.assertEqual(config.chunk_overlap, 500)
        self.assertTrue(config.use_unstructured_api)
        self.assertTrue(config.use_contextual_chunking)
        self.assertEqual(config.contextual_chunking_model, "custom_model")

    @pytest.mark.unit
    def test_from_toml_empty_data(self):
        """Test from_toml with empty data."""
        config = SensoryConfig.from_toml({})
        self.assertEqual(config.chunk_size, 6000)  # Default value
        self.assertEqual(config.chunk_overlap, 300)  # Default value
        self.assertFalse(config.use_unstructured_api)  # Default value
        self.assertFalse(config.use_contextual_chunking)  # Default value
        self.assertEqual(config.contextual_chunking_model, "ollama_qwen_vision")  # Default value

    @pytest.mark.unit
    def test_from_toml_with_data(self):
        """Test from_toml with valid data."""
        toml_data = {
            "sensory": {
                "parser": {
                    "chunk_size": 8000,
                    "chunk_overlap": 500,
                    "use_unstructured_api": True,
                    "use_contextual_chunking": True,
                    "contextual_chunking_model": "custom_vision_model"
                }
            }
        }

        config = SensoryConfig.from_toml(toml_data)

        self.assertEqual(config.chunk_size, 8000)
        self.assertEqual(config.chunk_overlap, 500)
        self.assertTrue(config.use_unstructured_api)
        self.assertTrue(config.use_contextual_chunking)
        self.assertEqual(config.contextual_chunking_model, "custom_vision_model")

    @pytest.mark.unit
    def test_from_toml_partial_data(self):
        """Test from_toml with partial data."""
        toml_data = {
            "sensory": {
                "parser": {
                    "chunk_size": 7000,
                    "use_contextual_chunking": True
                }
            }
        }

        config = SensoryConfig.from_toml(toml_data)

        # Values from TOML
        self.assertEqual(config.chunk_size, 7000)
        self.assertTrue(config.use_contextual_chunking)

        # Default values for missing fields
        self.assertEqual(config.chunk_overlap, 300)
        self.assertFalse(config.use_unstructured_api)
        self.assertEqual(config.contextual_chunking_model, "ollama_qwen_vision")

    @pytest.mark.unit
    def test_from_toml_missing_sensory_section(self):
        """Test from_toml when sensory section is missing."""
        toml_data = {
            "other_section": {
                "some_key": "some_value"
            }
        }

        config = SensoryConfig.from_toml(toml_data)

        # Should use all default values
        self.assertEqual(config.chunk_size, 6000)
        self.assertEqual(config.chunk_overlap, 300)
        self.assertFalse(config.use_unstructured_api)
        self.assertFalse(config.use_contextual_chunking)
        self.assertEqual(config.contextual_chunking_model, "ollama_qwen_vision")

    @pytest.mark.unit
    def test_from_toml_missing_parser_section(self):
        """Test from_toml when parser section is missing."""
        toml_data = {
            "sensory": {
                "other_parser": {
                    "some_key": "some_value"
                }
            }
        }

        config = SensoryConfig.from_toml(toml_data)

        # Should use all default values
        self.assertEqual(config.chunk_size, 6000)
        self.assertEqual(config.chunk_overlap, 300)
        self.assertFalse(config.use_unstructured_api)
        self.assertFalse(config.use_contextual_chunking)
        self.assertEqual(config.contextual_chunking_model, "ollama_qwen_vision")

    @pytest.mark.unit
    def test_from_toml_with_string_values(self):
        """Test from_toml with string values that should be converted."""
        toml_data = {
            "sensory": {
                "parser": {
                    "chunk_size": "8000",
                    "chunk_overlap": "500",
                    "use_unstructured_api": "true",
                    "use_contextual_chunking": "false"
                }
            }
        }

        config = SensoryConfig.from_toml(toml_data)

        # String values should be converted to appropriate types
        self.assertEqual(config.chunk_size, 8000)
        self.assertEqual(config.chunk_overlap, 500)
        self.assertTrue(config.use_unstructured_api)
        self.assertFalse(config.use_contextual_chunking)
        self.assertEqual(config.contextual_chunking_model, "ollama_qwen_vision")  # Default value

    @pytest.mark.unit
    def test_from_toml_with_invalid_numeric_values(self):
        """Test from_toml with invalid numeric values."""
        toml_data = {
            "sensory": {
                "parser": {
                    "chunk_size": "invalid_number",
                    "chunk_overlap": "not_a_number"
                }
            }
        }

        config = SensoryConfig.from_toml(toml_data)

        # Should use default values when TOML values are invalid
        self.assertEqual(config.chunk_size, 6000)
        self.assertEqual(config.chunk_overlap, 300)
        self.assertFalse(config.use_unstructured_api)
        self.assertFalse(config.use_contextual_chunking)
        self.assertEqual(config.contextual_chunking_model, "ollama_qwen_vision")


class TestCogentConfig(unittest.TestCase):
    """Test the main CogentConfig class."""

    @pytest.mark.unit
    @patch("cogent.base.config.load_merged_toml_configs")
    def test_default_values(self, mock_load_merged):
        """Test default values."""
        mock_load_merged.return_value = {}
        config = CogentConfig()
        self.assertEqual(config.env, "development")
        self.assertFalse(config.debug)
        self.assertIsInstance(config.config_dir, Path)
        self.assertIsInstance(config.base_toml, Path)
        self.assertIsInstance(config.providers_toml, Path)
        self.assertIsInstance(config.llm, LLMConfig)
        self.assertIsInstance(config.vector_store, VectorStoreConfig)
        self.assertIsInstance(config.reranker, RerankerConfig)
        self.assertIsInstance(config.sensory, SensoryConfig)

    @pytest.mark.unit
    @patch("cogent.base.config.load_merged_toml_configs")
    def test_load_merged_toml_configs_called(self, mock_load_merged):
        """Test that merged TOML config is loaded during initialization."""
        mock_load_merged.return_value = {}
        config = CogentConfig()
        mock_load_merged.assert_called_once_with([config.base_toml, config.providers_toml, config.sensory_toml])

    @pytest.mark.unit
    @patch("cogent.base.config.load_merged_toml_configs")
    def test_load_merged_toml_configs_with_data(self, mock_load_merged):
        """Test loading merged TOML config with actual data."""
        mock_toml_data = {
            "registered_models": {"test_model": {"model_name": "test"}},
            "registered_vector_stores": {"test_store": {"provider": "test"}},
            "completion": {"model": "test_completion_model"},
            "embedding": {"model": "test_embedding_model"},
            "vector_store": {"provider": "test_vector_provider"},
            "sensory": {"parser": {"chunk_size": 5000}},
            "graph": {"model": "test_graph_model"},
        }
        mock_load_merged.return_value = mock_toml_data

        config = CogentConfig()

        # Check that TOML data was loaded
        self.assertEqual(config.llm.registered_models, mock_toml_data["registered_models"])
        self.assertEqual(config.vector_store.registered_vector_stores, mock_toml_data["registered_vector_stores"])
        self.assertEqual(config.llm.completion_model, "test_completion_model")
        self.assertEqual(config.llm.embedding_model, "test_embedding_model")
        self.assertEqual(config.vector_store.provider, "test_vector_provider")
        self.assertEqual(config.sensory.chunk_size, 5000)

    @pytest.mark.unit
    @patch("cogent.base.config.load_merged_toml_configs")
    def test_config_paths(self, mock_load_merged):
        """Test that config paths are correctly set."""
        mock_load_merged.return_value = {}
        config = CogentConfig()

        # Test that paths are correctly constructed
        self.assertEqual(config.config_dir, ROOT_DIR / "config")
        self.assertEqual(config.base_toml, config.config_dir / "base.toml")
        self.assertEqual(config.providers_toml, config.config_dir / "providers.toml")


class TestGetConfig(unittest.TestCase):
    """Test the get_config function."""

    @pytest.mark.unit
    def test_get_config_returns_singleton(self):
        """Test that get_config returns the same instance."""
        config1 = get_config()
        config2 = get_config()
        self.assertIs(config1, config2)


if __name__ == "__main__":
    # Run tests
    unittest.main()
