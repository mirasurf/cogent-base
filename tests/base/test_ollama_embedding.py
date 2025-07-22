from unittest.mock import AsyncMock, patch

import pytest

from cogent.base.embedding.ollama_embedding import OllamaEmbeddingModel
from cogent.base.models.chunk import Chunk


class TestIntegrationOllamaEmbedding:
    """Integration tests for OllamaEmbeddingModel with external services."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration with test models."""
        return {
            "llm": {
                "registered_models": {
                    "test_ollama_embedding": {"model_name": "nomic-embed-text", "api_base": "http://localhost:11434"},
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
    async def test_ollama_object_embedding(self, mock_config, test_texts):
        """Test object embedding with Ollama model."""
        with patch("cogent.base.embedding.ollama_embedding.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]
            mock_get_config.return_value.llm.embedding_dimensions = mock_config["embedding"]["embedding_dimensions"]

            # Skip if Ollama is not available
            try:
                import ollama  # noqa: F401
            except ImportError:
                pytest.skip("Ollama library not available")

            # Check if Ollama service is running
            try:
                import httpx

                async with httpx.AsyncClient() as client:
                    response = await client.get("http://localhost:11434/api/tags", timeout=5.0)
                    if response.status_code != 200:
                        pytest.skip("Ollama service not running")
            except Exception:
                pytest.skip("Ollama service not accessible")

            model = OllamaEmbeddingModel("test_ollama_embedding")
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
    async def test_ollama_query_embedding(self, mock_config, single_text):
        """Test query embedding with Ollama model."""
        with patch("cogent.base.embedding.ollama_embedding.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]
            mock_get_config.return_value.llm.embedding_dimensions = mock_config["embedding"]["embedding_dimensions"]

            # Skip if Ollama is not available
            try:
                import ollama  # noqa: F401
            except ImportError:
                pytest.skip("Ollama library not available")

            # Check if Ollama service is running
            try:
                import httpx

                async with httpx.AsyncClient() as client:
                    response = await client.get("http://localhost:11434/api/tags", timeout=5.0)
                    if response.status_code != 200:
                        pytest.skip("Ollama service not running")
            except Exception:
                pytest.skip("Ollama service not accessible")

            model = OllamaEmbeddingModel("test_ollama_embedding")
            embedding = await model.embed_query(single_text)

            assert isinstance(embedding, list)
            assert all(isinstance(val, float) for val in embedding)
            assert len(embedding) > 0
            assert not all(val == 0.0 for val in embedding)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ollama_chunk_embedding_for_ingestion(self, mock_config, test_chunks):
        """Test chunk embedding for ingestion with Ollama model."""
        with patch("cogent.base.embedding.ollama_embedding.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]
            mock_get_config.return_value.llm.embedding_dimensions = mock_config["embedding"]["embedding_dimensions"]
            mock_get_config.return_value.llm.embedding_batch_size = 100

            # Skip if Ollama is not available
            try:
                import ollama  # noqa: F401
            except ImportError:
                pytest.skip("Ollama library not available")

            # Check if Ollama service is running
            try:
                import httpx

                async with httpx.AsyncClient() as client:
                    response = await client.get("http://localhost:11434/api/tags", timeout=5.0)
                    if response.status_code != 200:
                        pytest.skip("Ollama service not running")
            except Exception:
                pytest.skip("Ollama service not accessible")

            model = OllamaEmbeddingModel("test_ollama_embedding")
            embeddings = await model.embed_for_chunks(test_chunks)

            assert isinstance(embeddings, list)
            assert len(embeddings) == len(test_chunks)
            assert all(isinstance(emb, list) for emb in embeddings)
            assert all(isinstance(val, float) for emb in embeddings for val in emb)
            assert all(len(emb) > 0 for emb in embeddings)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ollama_single_chunk_embedding(self, mock_config):
        """Test single chunk embedding with Ollama model."""
        with patch("cogent.base.embedding.ollama_embedding.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]
            mock_get_config.return_value.llm.embedding_dimensions = mock_config["embedding"]["embedding_dimensions"]
            mock_get_config.return_value.llm.embedding_batch_size = 100

            # Skip if Ollama is not available
            try:
                import ollama  # noqa: F401
            except ImportError:
                pytest.skip("Ollama library not available")

            # Check if Ollama service is running
            try:
                import httpx

                async with httpx.AsyncClient() as client:
                    response = await client.get("http://localhost:11434/api/tags", timeout=5.0)
                    if response.status_code != 200:
                        pytest.skip("Ollama service not running")
            except Exception:
                pytest.skip("Ollama service not accessible")

            model = OllamaEmbeddingModel("test_ollama_embedding")
            single_chunk = Chunk(content="This is a single test chunk.", metadata={"source": "test"})
            embeddings = await model.embed_for_chunks(single_chunk)

            assert isinstance(embeddings, list)
            assert len(embeddings) == 1
            assert isinstance(embeddings[0], list)
            assert all(isinstance(val, float) for val in embeddings[0])
            assert len(embeddings[0]) > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ollama_embed_for_query(self, mock_config, single_text):
        """Test embed_for_query with Ollama model."""
        with patch("cogent.base.embedding.ollama_embedding.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]
            mock_get_config.return_value.llm.embedding_dimensions = mock_config["embedding"]["embedding_dimensions"]

            # Skip if Ollama is not available
            try:
                import ollama  # noqa: F401
            except ImportError:
                pytest.skip("Ollama library not available")

            # Check if Ollama service is running
            try:
                import httpx

                async with httpx.AsyncClient() as client:
                    response = await client.get("http://localhost:11434/api/tags", timeout=5.0)
                    if response.status_code != 200:
                        pytest.skip("Ollama service not running")
            except Exception:
                pytest.skip("Ollama service not accessible")

            model = OllamaEmbeddingModel("test_ollama_embedding")
            embedding = await model.embed_for_query(single_text)

            assert isinstance(embedding, list)
            assert all(isinstance(val, float) for val in embedding)
            assert len(embedding) > 0
            assert not all(val == 0.0 for val in embedding)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ollama_batch_processing(self, mock_config):
        """Test batch processing with Ollama model."""
        with patch("cogent.base.embedding.ollama_embedding.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]
            mock_get_config.return_value.llm.embedding_dimensions = mock_config["embedding"]["embedding_dimensions"]
            mock_get_config.return_value.llm.embedding_batch_size = 2  # Small batch size for testing

            # Skip if Ollama is not available
            try:
                import ollama  # noqa: F401
            except ImportError:
                pytest.skip("Ollama library not available")

            # Check if Ollama service is running
            try:
                import httpx

                async with httpx.AsyncClient() as client:
                    response = await client.get("http://localhost:11434/api/tags", timeout=5.0)
                    if response.status_code != 200:
                        pytest.skip("Ollama service not running")
            except Exception:
                pytest.skip("Ollama service not accessible")

            model = OllamaEmbeddingModel("test_ollama_embedding")
            large_chunk_list = [
                Chunk(content=f"This is test chunk number {i}.", metadata={"source": "test", "index": i})
                for i in range(5)  # More chunks than batch size
            ]
            embeddings = await model.embed_for_chunks(large_chunk_list)

            assert isinstance(embeddings, list)
            assert len(embeddings) == len(large_chunk_list)
            assert all(isinstance(emb, list) for emb in embeddings)
            assert all(len(emb) > 0 for emb in embeddings)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ollama_invalid_model_error(self, mock_config):
        """Test error handling with invalid model configuration."""
        with patch("cogent.base.embedding.ollama_embedding.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]

            with pytest.raises(ValueError, match="Model 'invalid_model' not found in registered_models"):
                OllamaEmbeddingModel("invalid_model")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ollama_context_handling(self, mock_config):
        """Test context handling with Ollama model."""
        with patch("cogent.base.embedding.ollama_embedding.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]
            mock_get_config.return_value.llm.embedding_dimensions = mock_config["embedding"]["embedding_dimensions"]

            # Skip if Ollama is not available
            try:
                import ollama  # noqa: F401
            except ImportError:
                pytest.skip("Ollama library not available")

            # Skip if Ollama service is not running
            try:
                import httpx

                async with httpx.AsyncClient() as client:
                    response = await client.get("http://localhost:11434/api/tags", timeout=5.0)
                    if response.status_code != 200:
                        pytest.skip("Ollama service not running")
            except Exception:
                pytest.skip("Ollama service not accessible")

            model = OllamaEmbeddingModel("test_ollama_embedding")

            # Test without context
            embeddings = await model.embed_objects([])
            assert isinstance(embeddings, list)
            assert len(embeddings) == 0


class TestUnitOllamaEmbedding:
    """Unit tests for OllamaEmbeddingModel with mocked dependencies."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration with test models."""
        return {
            "llm": {
                "registered_models": {
                    "test_ollama_embedding": {"model_name": "nomic-embed-text", "api_base": "http://localhost:11434"},
                }
            },
            "embedding": {"embedding_dimensions": 768},
        }

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_model_initialization(self, mock_config):
        """Test model initialization with valid config."""
        with patch("cogent.base.embedding.ollama_embedding.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]
            mock_get_config.return_value.llm.embedding_dimensions = mock_config["embedding"]["embedding_dimensions"]

            with patch("cogent.base.embedding.ollama_embedding.ollama") as mock_ollama:
                mock_ollama.__version__ = "0.1.0"

                model = OllamaEmbeddingModel("test_ollama_embedding")
                assert model.model_key == "test_ollama_embedding"
                assert model.model_config == mock_config["llm"]["registered_models"]["test_ollama_embedding"]
                assert model.dimensions == mock_config["embedding"]["embedding_dimensions"]
                assert model.ollama_api_base == "http://localhost:11434"
                assert model.ollama_base_model_name == "nomic-embed-text"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_ollama_library_not_installed(self, mock_config):
        """Test error when Ollama library is not installed."""
        with patch("cogent.base.embedding.ollama_embedding.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]

            with patch("cogent.base.embedding.ollama_embedding.ollama", None):
                with pytest.raises(ImportError, match="Ollama library not installed"):
                    OllamaEmbeddingModel("test_ollama_embedding")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_non_ollama_model_error(self, mock_config):
        """Test error when model is not configured as Ollama model."""
        mock_config["llm"]["registered_models"]["test_non_ollama"] = {"model_name": "text-embedding-3-small"}
        mock_config["llm"]["registered_models"]["test_non_ollama"]["embedding_dimensions"] = 768

        with patch("cogent.base.embedding.ollama_embedding.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]
            mock_get_config.return_value.llm.embedding_dimensions = mock_config["embedding"]["embedding_dimensions"]

            with patch("cogent.base.embedding.ollama_embedding.ollama") as mock_ollama:
                mock_ollama.__version__ = "0.1.0"

                with patch("cogent.base.embedding.ollama_embedding.initialize_ollama_model") as mock_init:
                    mock_init.return_value = (False, None, None)

                    with pytest.raises(ValueError, match="is not configured as an Ollama model"):
                        OllamaEmbeddingModel("test_non_ollama")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_model_initialization_invalid_model(self, mock_config):
        """Test model initialization with invalid model key."""
        with patch("cogent.base.embedding.ollama_embedding.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]

            with patch("cogent.base.embedding.ollama_embedding.ollama") as mock_ollama:
                mock_ollama.__version__ = "0.1.0"

                with pytest.raises(ValueError, match="Model 'invalid_model' not found in registered_models"):
                    OllamaEmbeddingModel("invalid_model")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mocked_ollama_embedding(self, mock_config):
        """Test Ollama embedding with mocked Ollama client."""
        with patch("cogent.base.embedding.ollama_embedding.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]
            mock_get_config.return_value.llm.embedding_dimensions = mock_config["embedding"]["embedding_dimensions"]

            with patch("cogent.base.embedding.ollama_embedding.ollama") as mock_ollama:
                mock_ollama.__version__ = "0.1.0"

                # Mock Ollama client and its embeddings method
                mock_client_instance = AsyncMock()
                mock_ollama.AsyncClient.return_value = mock_client_instance

                # Mock the embeddings response
                test_texts = ["Test text 1", "Test text 2"]
                mock_embeddings = [[0.1, 0.2, 0.3] * 256] * len(test_texts)  # 768 dimensions
                mock_client_instance.embeddings.side_effect = [
                    {"embedding": embedding} for embedding in mock_embeddings
                ]

                model = OllamaEmbeddingModel("test_ollama_embedding")
                embeddings = await model.embed_objects(test_texts)

                assert len(embeddings) == 2
                assert all(len(emb) == 768 for emb in embeddings)

                # Verify Ollama client was called correctly
                mock_ollama.AsyncClient.assert_called_once_with(host="http://localhost:11434")
                assert mock_client_instance.embeddings.call_count == 2
                mock_client_instance.embeddings.assert_any_call(model="nomic-embed-text", prompt="Test text 1")
                mock_client_instance.embeddings.assert_any_call(model="nomic-embed-text", prompt="Test text 2")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mocked_query_embedding(self, mock_config):
        """Test query embedding with mocked Ollama."""
        with patch("cogent.base.embedding.ollama_embedding.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]
            mock_get_config.return_value.llm.embedding_dimensions = mock_config["embedding"]["embedding_dimensions"]

            with patch("cogent.base.embedding.ollama_embedding.ollama") as mock_ollama:
                mock_ollama.__version__ = "0.1.0"

                mock_client_instance = AsyncMock()
                mock_ollama.AsyncClient.return_value = mock_client_instance
                mock_client_instance.embeddings.return_value = {"embedding": [0.1, 0.2, 0.3] * 256}  # 768 dimensions

                model = OllamaEmbeddingModel("test_ollama_embedding")
                embedding = await model.embed_query("test query")

                assert isinstance(embedding, list)
                assert len(embedding) == 768
                assert all(isinstance(val, float) for val in embedding)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mocked_chunk_embedding_for_ingestion(self, mock_config):
        """Test chunk embedding for ingestion with mocked Ollama."""
        test_chunks = [
            Chunk(content="Test content 1", metadata={"source": "test"}),
            Chunk(content="Test content 2", metadata={"source": "test"}),
        ]

        with patch("cogent.base.embedding.ollama_embedding.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]
            mock_get_config.return_value.llm.embedding_dimensions = mock_config["embedding"]["embedding_dimensions"]
            mock_get_config.return_value.llm.embedding_batch_size = 100

            with patch("cogent.base.embedding.ollama_embedding.ollama") as mock_ollama:
                mock_ollama.__version__ = "0.1.0"

                mock_client_instance = AsyncMock()
                mock_ollama.AsyncClient.return_value = mock_client_instance
                mock_client_instance.embeddings.side_effect = [
                    {"embedding": [0.1] * 768},
                    {"embedding": [0.2] * 768},
                ]

                model = OllamaEmbeddingModel("test_ollama_embedding")
                embeddings = await model.embed_for_chunks(test_chunks)

                assert isinstance(embeddings, list)
                assert len(embeddings) == 2

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mocked_embed_for_query(self, mock_config):
        """Test embed_for_query method with mocked Ollama."""
        with patch("cogent.base.embedding.ollama_embedding.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]
            mock_get_config.return_value.llm.embedding_dimensions = mock_config["embedding"]["embedding_dimensions"]

            with patch("cogent.base.embedding.ollama_embedding.ollama") as mock_ollama:
                mock_ollama.__version__ = "0.1.0"

                mock_client_instance = AsyncMock()
                mock_ollama.AsyncClient.return_value = mock_client_instance
                mock_client_instance.embeddings.return_value = {"embedding": [0.1] * 768}

                model = OllamaEmbeddingModel("test_ollama_embedding")
                embedding = await model.embed_for_query("test query")

                assert isinstance(embedding, list)
                assert len(embedding) == 768

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mocked_batch_embedding_processing(self, mock_config):
        """Test batch processing with small batch size."""
        test_chunks = [Chunk(content=f"Test content {i}", metadata={"source": "test"}) for i in range(5)]

        with patch("cogent.base.embedding.ollama_embedding.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]
            mock_get_config.return_value.llm.embedding_dimensions = mock_config["embedding"]["embedding_dimensions"]
            mock_get_config.return_value.llm.embedding_batch_size = 2  # Small batch size

            with patch("cogent.base.embedding.ollama_embedding.ollama") as mock_ollama:
                mock_ollama.__version__ = "0.1.0"

                mock_client_instance = AsyncMock()
                mock_ollama.AsyncClient.return_value = mock_client_instance
                mock_client_instance.embeddings.side_effect = [
                    {"embedding": [0.1] * 768},
                    {"embedding": [0.2] * 768},
                    {"embedding": [0.3] * 768},
                    {"embedding": [0.4] * 768},
                    {"embedding": [0.5] * 768},
                ]

                model = OllamaEmbeddingModel("test_ollama_embedding")
                embeddings = await model.embed_for_chunks(test_chunks)

                assert isinstance(embeddings, list)
                assert len(embeddings) == 5
                # Verify that multiple calls were made due to batching
                assert mock_client_instance.embeddings.call_count == 5

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mocked_error_handling(self, mock_config):
        """Test error handling in embedding methods."""
        with patch("cogent.base.embedding.ollama_embedding.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]
            mock_get_config.return_value.llm.embedding_dimensions = mock_config["embedding"]["embedding_dimensions"]

            with patch("cogent.base.embedding.ollama_embedding.ollama") as mock_ollama:
                mock_ollama.__version__ = "0.1.0"

                mock_client_instance = AsyncMock()
                mock_ollama.AsyncClient.return_value = mock_client_instance
                mock_client_instance.embeddings.side_effect = Exception("Ollama Error")

                model = OllamaEmbeddingModel("test_ollama_embedding")

                with pytest.raises(Exception, match="Ollama Error"):
                    await model.embed_objects(["test"])

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mocked_empty_input_handling(self, mock_config):
        """Test handling of empty inputs."""
        with patch("cogent.base.embedding.ollama_embedding.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]
            mock_get_config.return_value.llm.embedding_dimensions = mock_config["embedding"]["embedding_dimensions"]

            with patch("cogent.base.embedding.ollama_embedding.ollama") as mock_ollama:
                mock_ollama.__version__ = "0.1.0"

                model = OllamaEmbeddingModel("test_ollama_embedding")

                # Test empty lists
                embeddings = await model.embed_objects([])
                assert embeddings == []

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mocked_single_chunk_processing(self, mock_config):
        """Test processing of single chunk."""
        test_chunk = Chunk(content="Single test content", metadata={"source": "test"})

        with patch("cogent.base.embedding.ollama_embedding.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]
            mock_get_config.return_value.llm.embedding_dimensions = mock_config["embedding"]["embedding_dimensions"]
            mock_get_config.return_value.llm.embedding_batch_size = 100

            with patch("cogent.base.embedding.ollama_embedding.ollama") as mock_ollama:
                mock_ollama.__version__ = "0.1.0"

                mock_client_instance = AsyncMock()
                mock_ollama.AsyncClient.return_value = mock_client_instance
                mock_client_instance.embeddings.return_value = {"embedding": [0.1] * 768}

                model = OllamaEmbeddingModel("test_ollama_embedding")
                embeddings = await model.embed_for_chunks(test_chunk)

                assert isinstance(embeddings, list)
                assert len(embeddings) == 1
                assert len(embeddings[0]) == 768
