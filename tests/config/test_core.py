"""
Tests for core configuration classes.
"""

import unittest

import pytest

from cogent.base.config import (
    LLMConfig,
    RerankerConfig,
    SensoryConfig,
    VectorStoreConfig,
)


class TestLLMConfig(unittest.TestCase):
    """Test the LLMConfig class."""

    @pytest.mark.unit
    def test_default_values(self):
        """Test LLMConfig default values."""
        config = LLMConfig()
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
        """Test LLMConfig.from_toml with empty data."""
        config = LLMConfig.from_toml({})
        self.assertIsInstance(config, LLMConfig)

    @pytest.mark.unit
    def test_from_toml_with_data(self):
        """Test LLMConfig.from_toml with valid data."""
        toml_data = {
            "registered_models": {"test_model": {"model_name": "test"}},
            "completion": {
                "provider": "test_provider",
                "model": "test_model",
                "default_max_tokens": 1500,
                "default_temperature": 0.5,
            },
            "embedding": {
                "provider": "test_embedding_provider",
                "model": "test_embedding_model",
                "dimensions": 1024,
                "similarity_metric": "euclidean",
                "batch_size": 200,
            },
        }

        config = LLMConfig.from_toml(toml_data)

        self.assertEqual(config.registered_models["test_model"]["model_name"], "test")
        self.assertEqual(config.completion_provider, "test_provider")
        self.assertEqual(config.completion_model, "test_model")
        self.assertEqual(config.completion_max_tokens, 1500)
        self.assertEqual(config.completion_temperature, 0.5)
        self.assertEqual(config.embedding_provider, "test_embedding_provider")
        self.assertEqual(config.embedding_model, "test_embedding_model")
        self.assertEqual(config.embedding_dimensions, 1024)
        self.assertEqual(config.embedding_similarity_metric, "euclidean")
        self.assertEqual(config.embedding_batch_size, 200)

    @pytest.mark.unit
    def test_from_toml_invalid_numeric_values(self):
        """Test LLMConfig.from_toml with invalid numeric values."""
        toml_data = {
            "completion": {"default_max_tokens": "invalid", "default_temperature": "invalid"},
            "embedding": {"dimensions": "invalid", "batch_size": "invalid"},
        }

        config = LLMConfig.from_toml(toml_data)

        # Should fall back to defaults
        self.assertEqual(config.completion_max_tokens, 2000)
        self.assertEqual(config.completion_temperature, 0.7)
        self.assertEqual(config.embedding_dimensions, 768)
        self.assertEqual(config.embedding_batch_size, 100)

    @pytest.mark.unit
    def test_get_toml_section(self):
        """Test LLMConfig.get_toml_section method."""
        config = LLMConfig()
        section = config.get_toml_section()
        self.assertEqual(section, "llm")


class TestVectorStoreConfig(unittest.TestCase):
    """Test the VectorStoreConfig class."""

    @pytest.mark.unit
    def test_default_values(self):
        """Test VectorStoreConfig default values."""
        config = VectorStoreConfig()
        self.assertEqual(config.provider, "pgvector")
        self.assertEqual(config.collection_name, "cogent")
        self.assertEqual(config.embedding_model_dims, 768)

    @pytest.mark.unit
    def test_from_toml_empty_data(self):
        """Test VectorStoreConfig.from_toml with empty data."""
        config = VectorStoreConfig.from_toml({})
        self.assertIsInstance(config, VectorStoreConfig)

    @pytest.mark.unit
    def test_from_toml_with_data(self):
        """Test VectorStoreConfig.from_toml with valid data."""
        toml_data = {
            "registered_vector_stores": {"test_store": {"host": "localhost"}},
            "vector_store": {
                "provider": "test_provider",
                "collection_name": "test_collection",
                "embedding_model_dims": 1024,
            },
        }

        config = VectorStoreConfig.from_toml(toml_data)

        self.assertEqual(config.registered_vector_stores["test_store"]["host"], "localhost")
        self.assertEqual(config.provider, "test_provider")
        self.assertEqual(config.collection_name, "test_collection")
        self.assertEqual(config.embedding_model_dims, 1024)

    @pytest.mark.unit
    def test_from_toml_invalid_numeric_values(self):
        """Test VectorStoreConfig.from_toml with invalid numeric values."""
        toml_data = {"vector_store": {"embedding_model_dims": "invalid"}}

        config = VectorStoreConfig.from_toml(toml_data)

        # Should fall back to default
        self.assertEqual(config.embedding_model_dims, 768)

    @pytest.mark.unit
    def test_get_toml_section(self):
        """Test VectorStoreConfig.get_toml_section method."""
        config = VectorStoreConfig()
        section = config.get_toml_section()
        self.assertEqual(section, "vector_store")


class TestRerankerConfig(unittest.TestCase):
    """Test the RerankerConfig class."""

    @pytest.mark.unit
    def test_reranker_config_from_toml_full(self):
        """Test RerankerConfig.from_toml with full data."""
        toml_data = {
            "registered_rerankers": {"test_reranker": {"model_name": "test"}},
            "reranker": {
                "enable_reranker": True,
                "provider": "test_provider",
                "model": "test_model",
            },
        }

        config = RerankerConfig.from_toml(toml_data)

        self.assertEqual(config.registered_rerankers["test_reranker"]["model_name"], "test")
        self.assertTrue(config.enable_reranker)
        self.assertEqual(config.reranker_provider, "test_provider")
        self.assertEqual(config.reranker_model, "test_model")

    @pytest.mark.unit
    def test_reranker_config_from_toml_defaults(self):
        """Test RerankerConfig.from_toml with empty data."""
        config = RerankerConfig.from_toml({})
        self.assertFalse(config.enable_reranker)
        self.assertEqual(config.reranker_provider, "litellm")
        self.assertEqual(config.reranker_model, "ollama_reranker")

    @pytest.mark.unit
    def test_reranker_config_from_toml_partial(self):
        """Test RerankerConfig.from_toml with partial data."""
        toml_data = {"reranker": {"enable_reranker": True}}

        config = RerankerConfig.from_toml(toml_data)

        self.assertTrue(config.enable_reranker)
        self.assertEqual(config.reranker_provider, "litellm")  # default
        self.assertEqual(config.reranker_model, "ollama_reranker")  # default

    @pytest.mark.unit
    def test_get_toml_section(self):
        """Test RerankerConfig.get_toml_section method."""
        config = RerankerConfig()
        section = config.get_toml_section()
        self.assertEqual(section, "reranker")


class TestSensoryConfig(unittest.TestCase):
    """Test the SensoryConfig class."""

    @pytest.mark.unit
    def test_default_values(self):
        """Test SensoryConfig default values."""
        config = SensoryConfig()
        self.assertEqual(config.chunk_size, 6000)
        self.assertEqual(config.chunk_overlap, 300)
        self.assertFalse(config.use_unstructured_api)
        self.assertFalse(config.use_contextual_chunking)
        self.assertEqual(config.contextual_chunking_model, "ollama_qwen_vision")

    @pytest.mark.unit
    def test_custom_values(self):
        """Test SensoryConfig with custom values."""
        config = SensoryConfig(
            chunk_size=8000,
            chunk_overlap=500,
            use_unstructured_api=True,
            use_contextual_chunking=True,
            contextual_chunking_model="custom_model",
        )
        self.assertEqual(config.chunk_size, 8000)
        self.assertEqual(config.chunk_overlap, 500)
        self.assertTrue(config.use_unstructured_api)
        self.assertTrue(config.use_contextual_chunking)
        self.assertEqual(config.contextual_chunking_model, "custom_model")

    @pytest.mark.unit
    def test_from_toml_empty_data(self):
        """Test SensoryConfig.from_toml with empty data."""
        config = SensoryConfig.from_toml({})
        self.assertIsInstance(config, SensoryConfig)
        # Should use defaults
        self.assertEqual(config.chunk_size, 6000)
        self.assertEqual(config.chunk_overlap, 300)
        self.assertFalse(config.use_unstructured_api)
        self.assertFalse(config.use_contextual_chunking)
        self.assertEqual(config.contextual_chunking_model, "ollama_qwen_vision")

    @pytest.mark.unit
    def test_from_toml_with_data(self):
        """Test SensoryConfig.from_toml with valid data."""
        toml_data = {
            "sensory": {
                "parser": {
                    "chunk_size": 8000,
                    "chunk_overlap": 500,
                    "use_unstructured_api": True,
                    "use_contextual_chunking": True,
                    "contextual_chunking_model": "custom_model",
                }
            }
        }

        config = SensoryConfig.from_toml(toml_data)

        self.assertEqual(config.chunk_size, 8000)
        self.assertEqual(config.chunk_overlap, 500)
        self.assertTrue(config.use_unstructured_api)
        self.assertTrue(config.use_contextual_chunking)
        self.assertEqual(config.contextual_chunking_model, "custom_model")

    @pytest.mark.unit
    def test_from_toml_partial_data(self):
        """Test SensoryConfig.from_toml with partial data."""
        toml_data = {
            "sensory": {
                "parser": {
                    "chunk_size": 8000,
                    "use_contextual_chunking": True,
                }
            }
        }

        config = SensoryConfig.from_toml(toml_data)

        self.assertEqual(config.chunk_size, 8000)
        self.assertEqual(config.chunk_overlap, 300)  # default
        self.assertFalse(config.use_unstructured_api)  # default
        self.assertTrue(config.use_contextual_chunking)
        self.assertEqual(config.contextual_chunking_model, "ollama_qwen_vision")  # default

    @pytest.mark.unit
    def test_from_toml_missing_sensory_section(self):
        """Test SensoryConfig.from_toml with missing sensory section."""
        toml_data = {"other_section": {"key": "value"}}

        config = SensoryConfig.from_toml(toml_data)

        # Should use defaults
        self.assertEqual(config.chunk_size, 6000)
        self.assertEqual(config.chunk_overlap, 300)
        self.assertFalse(config.use_unstructured_api)
        self.assertFalse(config.use_contextual_chunking)
        self.assertEqual(config.contextual_chunking_model, "ollama_qwen_vision")

    @pytest.mark.unit
    def test_from_toml_missing_parser_section(self):
        """Test SensoryConfig.from_toml with missing parser section."""
        toml_data = {"sensory": {"other_subsection": {"key": "value"}}}

        config = SensoryConfig.from_toml(toml_data)

        # Should use defaults
        self.assertEqual(config.chunk_size, 6000)
        self.assertEqual(config.chunk_overlap, 300)
        self.assertFalse(config.use_unstructured_api)
        self.assertFalse(config.use_contextual_chunking)
        self.assertEqual(config.contextual_chunking_model, "ollama_qwen_vision")

    @pytest.mark.unit
    def test_from_toml_with_string_values(self):
        """Test SensoryConfig.from_toml with string values that should be converted."""
        toml_data = {
            "sensory": {
                "parser": {
                    "chunk_size": "8000",
                    "chunk_overlap": "500",
                    "use_unstructured_api": "true",
                    "use_contextual_chunking": "true",
                }
            }
        }

        config = SensoryConfig.from_toml(toml_data)

        self.assertEqual(config.chunk_size, 8000)
        self.assertEqual(config.chunk_overlap, 500)
        self.assertTrue(config.use_unstructured_api)
        self.assertTrue(config.use_contextual_chunking)

    @pytest.mark.unit
    def test_from_toml_with_invalid_numeric_values(self):
        """Test SensoryConfig.from_toml with invalid numeric values."""
        toml_data = {
            "sensory": {
                "parser": {
                    "chunk_size": "invalid",
                    "chunk_overlap": "invalid",
                    "use_unstructured_api": "invalid",
                    "use_contextual_chunking": "invalid",
                }
            }
        }

        config = SensoryConfig.from_toml(toml_data)

        # Should fall back to defaults
        self.assertEqual(config.chunk_size, 6000)
        self.assertEqual(config.chunk_overlap, 300)
        self.assertFalse(config.use_unstructured_api)
        self.assertFalse(config.use_contextual_chunking)

    @pytest.mark.unit
    def test_get_toml_section(self):
        """Test SensoryConfig.get_toml_section method."""
        config = SensoryConfig()
        section = config.get_toml_section()
        self.assertEqual(section, "sensory")
