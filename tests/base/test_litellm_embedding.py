import os
from unittest.mock import AsyncMock, patch

import pytest

from cogent.base.embedding.litellm_embedding import LiteLLMEmbeddingModel
from cogent.base.models.chunk import Chunk


class TestIntegrationLiteLLMEmbedding:
    """Integration tests for LiteLLMEmbeddingModel with external services."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration with test models."""
        return {
            "llm": {
                "registered_models": {
                    "test_openai_embedding": {"model_name": "text-embedding-3-small"},
                    "test_openai_embedding_large": {"model_name": "text-embedding-3-large"},
                }
            },
            "embedding": {"embedding_dimensions": 768},
        }

    @pytest.fixture
    def test_chunks(self):
        """Test chunks for embedding."""
        return [
            Chunk(content="This is a test object about artificial intelligence.", metadata={"source": "test"}),
            Chunk(content="Machine learning is a subset of AI.", metadata={"source": "test"}),
            Chunk(content="Deep learning uses neural networks.", metadata={"source": "test"}),
        ]

    @pytest.fixture
    def test_texts(self):
        """Test texts for embedding."""
        return [
            "This is a test object about artificial intelligence.",
            "Machine learning is a subset of AI.",
            "Deep learning uses neural networks.",
        ]

    @pytest.fixture
    def single_text(self):
        """Single test text for query embedding."""
        return "What is artificial intelligence?"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_openai_object_embedding(self, mock_config, test_texts):
        """Test object embedding with OpenAI model."""
        # Skip if OpenAI API key is not available
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OpenAI API key not available")

        with patch("cogent.base.embedding.litellm_embedding.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]
            mock_get_config.return_value.llm.embedding_dimensions = mock_config["embedding"]["embedding_dimensions"]

            model = LiteLLMEmbeddingModel("test_openai_embedding")
            embeddings = await model.embed_objects(test_texts)

            assert isinstance(embeddings, list)
            assert len(embeddings) == len(test_texts)
            assert all(isinstance(emb, list) for emb in embeddings)
            assert all(isinstance(val, float) for emb in embeddings for val in emb)
            assert all(len(emb) > 0 for emb in embeddings)

            # Check that embeddings are not all zeros
            assert not all(all(val == 0.0 for val in emb) for emb in embeddings)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_openai_large_object_embedding(self, mock_config, test_texts):
        """Test object embedding with large OpenAI model."""
        # Skip if OpenAI API key is not available
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OpenAI API key not available")

        with patch("cogent.base.embedding.litellm_embedding.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]
            mock_get_config.return_value.llm.embedding_dimensions = 2000  # Large model dimension

            model = LiteLLMEmbeddingModel("test_openai_embedding_large")
            embeddings = await model.embed_objects(test_texts)

            assert isinstance(embeddings, list)
            assert len(embeddings) == len(test_texts)
            assert all(isinstance(emb, list) for emb in embeddings)
            assert all(isinstance(val, float) for emb in embeddings for val in emb)
            assert all(len(emb) > 0 for emb in embeddings)

            # For large models, expect higher dimensionality
            assert all(len(emb) >= 1536 for emb in embeddings)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_openai_query_embedding(self, mock_config, single_text):
        """Test query embedding with OpenAI model."""
        # Skip if OpenAI API key is not available
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OpenAI API key not available")

        with patch("cogent.base.embedding.litellm_embedding.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]
            mock_get_config.return_value.llm.embedding_dimensions = mock_config["embedding"]["embedding_dimensions"]

            model = LiteLLMEmbeddingModel("test_openai_embedding")
            embedding = await model.embed_query(single_text)

            assert isinstance(embedding, list)
            assert len(embedding) > 0
            assert all(isinstance(val, float) for val in embedding)

            # Check that embedding is not all zeros
            assert not all(val == 0.0 for val in embedding)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_openai_chunk_embedding_for_ingestion(self, mock_config, test_chunks):
        """Test chunk embedding for ingestion with OpenAI model."""
        # Skip if OpenAI API key is not available
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OpenAI API key not available")

        with patch("cogent.base.embedding.litellm_embedding.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]
            mock_get_config.return_value.llm.embedding_dimensions = mock_config["embedding"]["embedding_dimensions"]
            mock_get_config.return_value.llm.embedding_batch_size = 100

            model = LiteLLMEmbeddingModel("test_openai_embedding")
            embeddings = await model.embed_for_chunks(test_chunks)

            assert isinstance(embeddings, list)
            assert len(embeddings) == len(test_chunks)
            assert all(isinstance(emb, list) for emb in embeddings)
            assert all(isinstance(val, float) for emb in embeddings for val in emb)
            assert all(len(emb) > 0 for emb in embeddings)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_openai_query_embedding_for_query(self, mock_config, single_text):
        """Test query embedding for query with OpenAI model."""
        # Skip if OpenAI API key is not available
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OpenAI API key not available")

        with patch("cogent.base.embedding.litellm_embedding.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]
            mock_get_config.return_value.llm.embedding_dimensions = mock_config["embedding"]["embedding_dimensions"]

            model = LiteLLMEmbeddingModel("test_openai_embedding")
            embedding = await model.embed_for_query(single_text)

            assert isinstance(embedding, list)
            assert len(embedding) > 0
            assert all(isinstance(val, float) for val in embedding)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_openai_single_chunk_embedding(self, mock_config, test_chunks):
        """Test single chunk embedding with OpenAI model."""
        # Skip if OpenAI API key is not available
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OpenAI API key not available")

        with patch("cogent.base.embedding.litellm_embedding.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]
            mock_get_config.return_value.llm.embedding_dimensions = mock_config["embedding"]["embedding_dimensions"]
            mock_get_config.return_value.llm.embedding_batch_size = 100

            model = LiteLLMEmbeddingModel("test_openai_embedding")
            # Test with single chunk
            embeddings = await model.embed_for_chunks(test_chunks[0])

            assert isinstance(embeddings, list)
            assert len(embeddings) == 1
            assert isinstance(embeddings[0], list)
            assert all(isinstance(val, float) for val in embeddings[0])
            assert len(embeddings[0]) > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_openai_batch_embedding(self, mock_config):
        """Test batch embedding with large number of texts."""
        # Skip if OpenAI API key is not available
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OpenAI API key not available")

        # Create larger set of test chunks
        large_chunk_set = [Chunk(content=f"Test content number {i}", metadata={"source": "test"}) for i in range(10)]

        with patch("cogent.base.embedding.litellm_embedding.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]
            mock_get_config.return_value.llm.embedding_dimensions = mock_config["embedding"]["embedding_dimensions"]
            mock_get_config.return_value.llm.embedding_batch_size = 3  # Small batch size to test batching

            model = LiteLLMEmbeddingModel("test_openai_embedding")
            embeddings = await model.embed_for_chunks(large_chunk_set)

            assert isinstance(embeddings, list)
            assert len(embeddings) == len(large_chunk_set)
            assert all(isinstance(emb, list) for emb in embeddings)
            assert all(len(emb) > 0 for emb in embeddings)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_error_handling_invalid_model(self):
        """Test error handling for invalid model."""
        with patch("cogent.base.embedding.litellm_embedding.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = {}

            with pytest.raises(ValueError, match="Model 'invalid_model' not found"):
                LiteLLMEmbeddingModel("invalid_model")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_error_handling_empty_texts(self, mock_config):
        """Test handling of empty text list."""
        with patch("cogent.base.embedding.litellm_embedding.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]
            mock_get_config.return_value.llm.embedding_dimensions = mock_config["embedding"]["embedding_dimensions"]

            model = LiteLLMEmbeddingModel("test_openai_embedding")
            embeddings = await model.embed_objects([])

            assert isinstance(embeddings, list)
            assert len(embeddings) == 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_error_handling_empty_chunks(self, mock_config):
        """Test handling of empty chunk list."""
        with patch("cogent.base.embedding.litellm_embedding.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]
            mock_get_config.return_value.llm.embedding_dimensions = mock_config["embedding"]["embedding_dimensions"]
            mock_get_config.return_value.llm.embedding_batch_size = 100

            model = LiteLLMEmbeddingModel("test_openai_embedding")
            embeddings = await model.embed_for_chunks([])

            assert isinstance(embeddings, list)
            assert len(embeddings) == 0


class TestUnitLiteLLMEmbedding:
    """Unit tests for LiteLLMEmbeddingModel with mocked dependencies."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration with test models."""
        return {
            "llm": {
                "registered_models": {
                    "test_openai_embedding": {"model_name": "text-embedding-3-small"},
                    "test_openai_embedding_large": {"model_name": "text-embedding-3-large"},
                }
            },
            "embedding": {"embedding_dimensions": 768},
        }

    @pytest.fixture
    def mock_litellm_response(self):
        """Mock LiteLLM embedding response generator for batch tests."""

        def _make_response(num_embeddings=3, dim=765):
            class MockResponse:
                def __init__(self, num_embeddings, dim):
                    self.data = [{"embedding": [float(i % 10) for i in range(dim)]} for _ in range(num_embeddings)]

            return MockResponse(num_embeddings, dim)

        return _make_response

    @pytest.fixture
    def mock_litellm_response_object(self):
        """Mock LiteLLM embedding response as object with data attribute."""

        class MockResponse:
            def __init__(self):
                self.data = [
                    {"embedding": [0.1, 0.2, 0.3, 0.4, 0.5] * 153},  # 765 dimensions
                    {"embedding": [0.2, 0.3, 0.4, 0.5, 0.6] * 153},  # 765 dimensions
                ]

        return MockResponse()

    @pytest.fixture
    def mock_litellm_response_direct(self):
        """Mock LiteLLM embedding response as direct list."""
        return [
            [0.1, 0.2, 0.3, 0.4, 0.5] * 153,  # 765 dimensions
            [0.2, 0.3, 0.4, 0.5, 0.6] * 153,  # 765 dimensions
        ]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_model_initialization(self, mock_config):
        """Test model initialization with valid config."""
        with patch("cogent.base.embedding.litellm_embedding.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]
            mock_get_config.return_value.llm.embedding_dimensions = mock_config["embedding"]["embedding_dimensions"]

            model = LiteLLMEmbeddingModel("test_openai_embedding")
            assert model.model_key == "test_openai_embedding"
            assert model.model_config == mock_config["llm"]["registered_models"]["test_openai_embedding"]
            assert model.dimensions == mock_config["embedding"]["embedding_dimensions"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_model_initialization_invalid_model(self, mock_config):
        """Test model initialization with invalid model key."""
        with patch("cogent.base.embedding.litellm_embedding.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]

            with pytest.raises(ValueError, match="Model 'invalid_model' not found in registered_models"):
                LiteLLMEmbeddingModel("invalid_model")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mocked_litellm_embedding(self, mock_config, mock_litellm_response_object):
        """Test embedding with mocked LiteLLM."""
        with patch("cogent.base.embedding.litellm_embedding.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]
            mock_get_config.return_value.llm.embedding_dimensions = mock_config["embedding"]["embedding_dimensions"]

            with patch("cogent.base.embedding.litellm_embedding.litellm") as mock_litellm:
                mock_litellm.aembedding = AsyncMock(return_value=mock_litellm_response_object)

                model = LiteLLMEmbeddingModel("test_openai_embedding")
                test_texts = ["test text 1", "test text 2"]
                embeddings = await model.embed_objects(test_texts)

                assert isinstance(embeddings, list)
                assert len(embeddings) == 2
                assert all(isinstance(emb, list) for emb in embeddings)
                assert all(len(emb) == 765 for emb in embeddings)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mocked_large_embedding(self, mock_config):
        """Test embedding with large model configuration."""
        with patch("cogent.base.embedding.litellm_embedding.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]
            mock_get_config.return_value.llm.embedding_dimensions = 2000

            # Mock response for large model
            class MockLargeResponse:
                def __init__(self):
                    self.data = [{"embedding": [0.1] * 2000}]

            with patch("cogent.base.embedding.litellm_embedding.litellm") as mock_litellm:
                mock_litellm.aembedding = AsyncMock(return_value=MockLargeResponse())

                model = LiteLLMEmbeddingModel("test_openai_embedding_large")
                await model.embed_objects(["test text"])

                # Verify that dimensions parameter is set for large models
                call_kwargs = mock_litellm.aembedding.call_args[1]
                assert "dimensions" in call_kwargs
                assert call_kwargs["dimensions"] == 2000

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mocked_query_embedding(self, mock_config, mock_litellm_response_object):
        """Test query embedding with mocked LiteLLM."""
        with patch("cogent.base.embedding.litellm_embedding.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]
            mock_get_config.return_value.llm.embedding_dimensions = mock_config["embedding"]["embedding_dimensions"]

            with patch("cogent.base.embedding.litellm_embedding.litellm") as mock_litellm:
                mock_litellm.aembedding = AsyncMock(return_value=mock_litellm_response_object)

                model = LiteLLMEmbeddingModel("test_openai_embedding")
                embedding = await model.embed_query("test query")

                assert isinstance(embedding, list)
                assert len(embedding) == 765
                assert all(isinstance(val, float) for val in embedding)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mocked_chunk_embedding_for_ingestion(self, mock_config, mock_litellm_response_object):
        """Test chunk embedding for ingestion with mocked LiteLLM."""
        test_chunks = [
            Chunk(content="Test content 1", metadata={"source": "test"}),
            Chunk(content="Test content 2", metadata={"source": "test"}),
        ]

        with patch("cogent.base.embedding.litellm_embedding.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]
            mock_get_config.return_value.llm.embedding_dimensions = mock_config["embedding"]["embedding_dimensions"]
            mock_get_config.return_value.llm.embedding_batch_size = 100

            with patch("cogent.base.embedding.litellm_embedding.litellm") as mock_litellm:
                mock_litellm.aembedding = AsyncMock(return_value=mock_litellm_response_object)

                model = LiteLLMEmbeddingModel("test_openai_embedding")
                embeddings = await model.embed_for_chunks(test_chunks)

                assert isinstance(embeddings, list)
                assert len(embeddings) == 2

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mocked_embed_for_query(self, mock_config, mock_litellm_response_object):
        """Test embed_for_query method with mocked LiteLLM."""
        with patch("cogent.base.embedding.litellm_embedding.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]
            mock_get_config.return_value.llm.embedding_dimensions = mock_config["embedding"]["embedding_dimensions"]

            with patch("cogent.base.embedding.litellm_embedding.litellm") as mock_litellm:
                mock_litellm.aembedding = AsyncMock(return_value=mock_litellm_response_object)

                model = LiteLLMEmbeddingModel("test_openai_embedding")
                embedding = await model.embed_for_query("test query")

                assert isinstance(embedding, list)
                assert len(embedding) == 765

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mocked_batch_embedding_processing(self, mock_config, mock_litellm_response):
        """Test batch processing with small batch size."""
        test_chunks = [Chunk(content=f"Test content {i}", metadata={"source": "test"}) for i in range(5)]

        with patch("cogent.base.embedding.litellm_embedding.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]
            mock_get_config.return_value.llm.embedding_dimensions = mock_config["embedding"]["embedding_dimensions"]
            mock_get_config.return_value.llm.embedding_batch_size = 2  # Small batch size

            with patch("cogent.base.embedding.litellm_embedding.litellm") as mock_litellm:
                # Mock different responses for different batch calls
                mock_litellm.aembedding = AsyncMock(
                    side_effect=[
                        mock_litellm_response(2, 765),  # First batch
                        mock_litellm_response(2, 765),  # Second batch
                        mock_litellm_response(1, 765),  # Third batch
                    ]
                )

                model = LiteLLMEmbeddingModel("test_openai_embedding")
                embeddings = await model.embed_for_chunks(test_chunks)

                assert isinstance(embeddings, list)
                assert len(embeddings) == 5
                # Verify that multiple calls were made due to batching
                assert mock_litellm.aembedding.call_count == 3

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mocked_error_handling(self, mock_config):
        """Test error handling in embedding methods."""
        with patch("cogent.base.embedding.litellm_embedding.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]
            mock_get_config.return_value.llm.embedding_dimensions = mock_config["embedding"]["embedding_dimensions"]

            with patch("cogent.base.embedding.litellm_embedding.litellm") as mock_litellm:
                mock_litellm.aembedding = AsyncMock(side_effect=Exception("API Error"))

                model = LiteLLMEmbeddingModel("test_openai_embedding")

                with pytest.raises(Exception, match="API Error"):
                    await model.embed_objects(["test"])

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mocked_empty_input_handling(self, mock_config):
        """Test handling of empty inputs."""
        with patch("cogent.base.embedding.litellm_embedding.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]
            mock_get_config.return_value.llm.embedding_dimensions = mock_config["embedding"]["embedding_dimensions"]

            model = LiteLLMEmbeddingModel("test_openai_embedding")

            # Test empty lists
            embeddings = await model.embed_objects([])
            assert embeddings == []

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mocked_single_chunk_processing(self, mock_config, mock_litellm_response_object):
        """Test processing of single chunk."""
        test_chunk = Chunk(content="Single test content", metadata={"source": "test"})

        with patch("cogent.base.embedding.litellm_embedding.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]
            mock_get_config.return_value.llm.embedding_dimensions = mock_config["embedding"]["embedding_dimensions"]
            mock_get_config.return_value.llm.embedding_batch_size = 100

            with patch("cogent.base.embedding.litellm_embedding.litellm") as mock_litellm:
                # Mock response for single embedding
                class MockSingleResponse:
                    def __init__(self):
                        self.data = [{"embedding": [0.1] * 765}]

                mock_litellm.aembedding = AsyncMock(return_value=MockSingleResponse())

                model = LiteLLMEmbeddingModel("test_openai_embedding")
                embeddings = await model.embed_for_chunks(test_chunk)

                assert isinstance(embeddings, list)
                assert len(embeddings) == 1
                assert len(embeddings[0]) == 765
