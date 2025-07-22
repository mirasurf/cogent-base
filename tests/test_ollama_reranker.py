"""
Tests for Ollama reranker provider.
"""

from unittest.mock import AsyncMock, patch

import pytest

try:
    import httpx
except ImportError:
    httpx = None

from cogent_base.models.chunk import ObjectChunk
from cogent_base.reranker.ollama_reranker import OllamaReranker


class TestIntegrationOllamaReranker:
    """Integration tests for OllamaReranker with external services."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ollama_reranker_integration(self):
        """Integration test for Ollama reranker with real server and ollama_reranker model."""
        # Skip if httpx is not available
        if httpx is None:
            pytest.skip("httpx library not available")

        # Skip if Ollama is not available
        try:
            import ollama  # noqa: F401
        except ImportError:
            pytest.skip("Ollama library not available")

        # Check if Ollama service is running
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get("http://localhost:11434/api/tags", timeout=5.0)
                if response.status_code != 200:
                    pytest.skip("Ollama service not running")
        except Exception:
            pytest.skip("Ollama service not accessible")

        # Check if the specific model is available
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get("http://localhost:11434/api/tags", timeout=5.0)
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    model_names = [model.get("name", "") for model in models]
                    if "linux6200/bge-reranker-v2-m3:latest" not in model_names:
                        pytest.skip(
                            "Required Ollama model 'linux6200/bge-reranker-v2-m3:latest' not found. "
                            f"Available models: {model_names}"
                        )
        except Exception:
            pytest.skip("Could not check available Ollama models")

        class DummyReranker:
            def __init__(self):
                self.registered_rerankers = {
                    "ollama_reranker": {
                        "model_name": "linux6200/bge-reranker-v2-m3:latest",
                        "api_base": "http://localhost:11434",
                    }
                }

        class DummyConfig:
            def __init__(self):
                self.reranker = DummyReranker()

        mock_config = DummyConfig()
        with patch("cogent_base.reranker.ollama_reranker.get_cogent_config", return_value=mock_config):
            reranker = OllamaReranker("ollama_reranker")
            query = "What is machine learning?"
            chunks = [
                ObjectChunk(
                    object_id="doc1",
                    content="Machine learning is a subset of AI.",
                    embedding=[0.1] * 768,
                    chunk_number=0,
                    score=0.0,
                ),
                ObjectChunk(
                    object_id="doc2",
                    content="Deep learning uses neural networks.",
                    embedding=[0.2] * 768,
                    chunk_number=0,
                    score=0.0,
                ),
            ]

            try:
                result = await reranker.rerank(query, chunks)

                assert len(result) == 2
                assert all(isinstance(c.score, float) for c in result)

                # Test compute_score
                score = await reranker.compute_score(query, chunks[0].content)
                assert isinstance(score, float)
                assert 0.0 <= score <= 1.0
            except Exception as e:
                # If the model fails due to resource limitations, that's acceptable for integration tests
                if "resource limitations" in str(e) or "unexpectedly stopped" in str(e):
                    pytest.skip(f"Ollama model failed due to resource limitations: {e}")
                else:
                    raise  # Re-raise unexpected errors


class TestUnitOllamaReranker:
    """Unit tests for OllamaReranker with mocked dependencies."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration with test models."""
        return {
            "reranker": {
                "registered_rerankers": {
                    "ollama_reranker": {"model_name": "bge-reranker-v2-m3", "api_base": "http://localhost:11434"}
                }
            }
        }

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_model_initialization(self, mock_config):
        """Test model initialization with valid config."""
        with patch("cogent_base.reranker.ollama_reranker.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.reranker.registered_rerankers = mock_config["reranker"]["registered_rerankers"]

            with patch("cogent_base.reranker.ollama_reranker.ollama") as mock_ollama:
                mock_ollama.__version__ = "0.1.0"

                reranker = OllamaReranker("ollama_reranker")
                assert reranker.model_key == "ollama_reranker"
                assert reranker.model_config == mock_config["reranker"]["registered_rerankers"]["ollama_reranker"]
                assert reranker.ollama_api_base == "http://localhost:11434"
                assert reranker.ollama_base_model_name == "bge-reranker-v2-m3"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_ollama_library_not_installed(self, mock_config):
        """Test error when Ollama library is not installed."""
        with patch("cogent_base.reranker.ollama_reranker.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.reranker.registered_rerankers = mock_config["reranker"]["registered_rerankers"]

            with patch("cogent_base.reranker.ollama_reranker.ollama", None):
                with pytest.raises(ImportError, match="Ollama library not installed"):
                    OllamaReranker("ollama_reranker")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_non_ollama_model_error(self, mock_config):
        """Test error when model is not configured as Ollama model."""
        mock_config["reranker"]["registered_rerankers"]["test_non_ollama"] = {"model_name": "gpt-4o-mini"}

        with patch("cogent_base.reranker.ollama_reranker.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.reranker.registered_rerankers = mock_config["reranker"]["registered_rerankers"]

            with patch("cogent_base.reranker.ollama_reranker.ollama") as mock_ollama:
                mock_ollama.__version__ = "0.1.0"

                with patch("cogent_base.reranker.ollama_reranker.initialize_ollama_model") as mock_init:
                    mock_init.return_value = (False, None, None)

                    with pytest.raises(ValueError, match="is not configured as an Ollama model"):
                        OllamaReranker("test_non_ollama")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_model_initialization_invalid_model(self, mock_config):
        """Test model initialization with invalid model key."""
        with patch("cogent_base.reranker.ollama_reranker.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.reranker.registered_rerankers = mock_config["reranker"]["registered_rerankers"]

            with patch("cogent_base.reranker.ollama_reranker.ollama") as mock_ollama:
                mock_ollama.__version__ = "0.1.0"

                with pytest.raises(ValueError, match="Reranker 'invalid_model' not found in registered_rerankers"):
                    OllamaReranker("invalid_model")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_ollama_reranker_rerank(self, mock_config):
        """Test Ollama reranker reranking functionality."""
        with patch("cogent_base.reranker.ollama_reranker.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.reranker.registered_rerankers = mock_config["reranker"]["registered_rerankers"]

            with patch("cogent_base.reranker.ollama_reranker.ollama") as mock_ollama:
                mock_ollama.__version__ = "0.1.0"

                mock_client = AsyncMock()
                mock_ollama.AsyncClient.return_value = mock_client
                mock_client.chat.return_value = {"message": {"content": "0.7"}}

                reranker = OllamaReranker("ollama_reranker")
                query = "What is machine learning?"
                chunks = [
                    ObjectChunk(
                        object_id="doc1",
                        content="Machine learning is a subset of AI.",
                        embedding=[0.1] * 768,
                        chunk_number=0,
                        score=0.0,
                    ),
                ]

                result = await reranker.rerank(query, chunks)

                assert len(result) == 1
                assert abs(result[0].score - 0.7) < 1e-6

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_ollama_reranker_compute_score(self, mock_config):
        """Test Ollama reranker compute score functionality."""
        with patch("cogent_base.reranker.ollama_reranker.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.reranker.registered_rerankers = mock_config["reranker"]["registered_rerankers"]

            with patch("cogent_base.reranker.ollama_reranker.ollama") as mock_ollama:
                mock_ollama.__version__ = "0.1.0"

                mock_client = AsyncMock()
                mock_ollama.AsyncClient.return_value = mock_client
                mock_client.chat.return_value = {"message": {"content": "0.5"}}

                reranker = OllamaReranker("ollama_reranker")
                query = "What is machine learning?"
                text = "Machine learning is a subset of AI."

                result = await reranker.compute_score(query, text)

                assert isinstance(result, float)
                assert abs(result - 0.5) < 1e-6

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_ollama_reranker_compute_score_multiple(self, mock_config):
        """Test Ollama reranker compute score with multiple texts."""
        with patch("cogent_base.reranker.ollama_reranker.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.reranker.registered_rerankers = mock_config["reranker"]["registered_rerankers"]

            with patch("cogent_base.reranker.ollama_reranker.ollama") as mock_ollama:
                mock_ollama.__version__ = "0.1.0"

                mock_client = AsyncMock()
                mock_ollama.AsyncClient.return_value = mock_client
                # Mock different scores for different texts
                mock_client.chat.side_effect = [
                    {"message": {"content": "0.8"}},
                    {"message": {"content": "0.6"}},
                    {"message": {"content": "0.9"}},
                ]

                reranker = OllamaReranker("ollama_reranker")
                query = "What is machine learning?"
                texts = [
                    "Machine learning is a subset of AI.",
                    "Python is a programming language.",
                    "Deep learning uses neural networks.",
                ]

                result = await reranker.compute_score(query, texts)

                assert isinstance(result, list)
                assert len(result) == 3
                assert result == [0.8, 0.6, 0.9]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_ollama_error_handling(self, mock_config):
        """Test error handling in Ollama reranker methods."""
        with patch("cogent_base.reranker.ollama_reranker.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.reranker.registered_rerankers = mock_config["reranker"]["registered_rerankers"]

            with patch("cogent_base.reranker.ollama_reranker.ollama") as mock_ollama:
                mock_ollama.__version__ = "0.1.0"

                mock_client = AsyncMock()
                mock_ollama.AsyncClient.return_value = mock_client
                mock_client.chat.side_effect = Exception("Ollama Error")

                reranker = OllamaReranker("ollama_reranker")
                query = "What is machine learning?"
                text = "Machine learning is a subset of AI."

                # Should handle error gracefully and return fallback score
                result = await reranker.compute_score(query, text)
                assert result == 0.5  # Fallback score

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_ollama_invalid_score_parsing(self, mock_config):
        """Test handling of invalid score responses from Ollama."""
        with patch("cogent_base.reranker.ollama_reranker.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.reranker.registered_rerankers = mock_config["reranker"]["registered_rerankers"]

            with patch("cogent_base.reranker.ollama_reranker.ollama") as mock_ollama:
                mock_ollama.__version__ = "0.1.0"

                mock_client = AsyncMock()
                mock_ollama.AsyncClient.return_value = mock_client
                mock_client.chat.return_value = {"message": {"content": "invalid_score"}}

                reranker = OllamaReranker("ollama_reranker")
                query = "What is machine learning?"
                text = "Machine learning is a subset of AI."

                result = await reranker.compute_score(query, text)
                assert result == 0.0  # Fallback for invalid parsing

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_ollama_empty_chunks(self, mock_config):
        """Test handling of empty chunk list with Ollama reranker."""
        with patch("cogent_base.reranker.ollama_reranker.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.reranker.registered_rerankers = mock_config["reranker"]["registered_rerankers"]

            with patch("cogent_base.reranker.ollama_reranker.ollama") as mock_ollama:
                mock_ollama.__version__ = "0.1.0"

                reranker = OllamaReranker("ollama_reranker")
                query = "What is machine learning?"
                chunks = []

                result = await reranker.rerank(query, chunks)
                assert result == []

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_ollama_score_bounds(self, mock_config):
        """Test that Ollama scores are properly bounded between 0 and 1."""
        with patch("cogent_base.reranker.ollama_reranker.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.reranker.registered_rerankers = mock_config["reranker"]["registered_rerankers"]

            with patch("cogent_base.reranker.ollama_reranker.ollama") as mock_ollama:
                mock_ollama.__version__ = "0.1.0"

                mock_client = AsyncMock()
                mock_ollama.AsyncClient.return_value = mock_client
                # Test scores outside bounds that should be clipped
                mock_client.chat.side_effect = [
                    {"message": {"content": "1.5"}},  # Should be clipped to 1.0
                    {"message": {"content": "-0.3"}},  # Should be clipped to 0.0
                ]

                reranker = OllamaReranker("ollama_reranker")
                query = "What is machine learning?"

                # Test high score clipping
                result1 = await reranker.compute_score(query, "text1")
                assert result1 == 1.0

                # Test low score clipping
                result2 = await reranker.compute_score(query, "text2")
                assert result2 == 0.0
