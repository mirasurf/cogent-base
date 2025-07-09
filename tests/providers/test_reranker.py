"""
Tests for reranker providers.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    import httpx
except ImportError:
    httpx = None

from cogent.base.models.chunk import ObjectChunk
from cogent.base.providers.reranker.cogent_reranker import CogentReranker
from cogent.base.providers.reranker.flag_reranker import FlagReranker
from cogent.base.providers.reranker.litellm_reranker import LiteLLMReranker


class TestFlagRerankerUnit:
    """Unit tests for FlagReranker implementation."""

    @pytest.fixture
    def flag_reranker(self):
        """Create a FlagReranker instance for testing."""
        with patch("FlagEmbedding.FlagAutoReranker.from_finetuned") as mock_from_finetuned:
            mock_reranker = MagicMock()

            # Mock compute_score to return different values for different calls
            def compute_score_side_effect(pairs, normalize=True):
                if len(pairs) == 1:
                    return [0.8]  # Single text returns list with one score
                return [0.8, 0.6, 0.9]  # Multiple texts return list of scores

            mock_reranker.compute_score.side_effect = compute_score_side_effect
            mock_from_finetuned.return_value = mock_reranker
            return FlagReranker()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_rerank(self, flag_reranker):
        """Test reranking functionality."""
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
                content="Python is a programming language.",
                embedding=[0.2] * 768,
                chunk_number=0,
                score=0.0,
            ),
            ObjectChunk(
                object_id="doc3",
                content="Deep learning uses neural networks.",
                embedding=[0.3] * 768,
                chunk_number=0,
                score=0.0,
            ),
        ]

        result = await flag_reranker.rerank(query, chunks)

        assert len(result) == 3
        assert result[0].score == 0.9  # Highest score should be first
        assert result[1].score == 0.8
        assert result[2].score == 0.6

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_compute_score_single(self, flag_reranker):
        """Test computing score for single text."""
        query = "What is machine learning?"
        text = "Machine learning is a subset of AI."

        result = await flag_reranker.compute_score(query, text)

        # FlagReranker returns a float for single text
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_compute_score_multiple(self, flag_reranker):
        """Test computing scores for multiple texts."""
        query = "What is machine learning?"
        texts = [
            "Machine learning is a subset of AI.",
            "Python is a programming language.",
            "Deep learning uses neural networks.",
        ]

        result = await flag_reranker.compute_score(query, texts)

        assert isinstance(result, list)
        assert len(result) == 3
        assert all(0.0 <= score <= 1.0 for score in result)


class TestLiteLLMRerankerUnit:
    """Unit tests for LiteLLMReranker implementation."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_rerank(self):
        """Test reranking functionality."""

        class DummyReranker:
            def __init__(self):
                self.registered_rerankers = {
                    "test_reranker": {"model_name": "gpt-4", "api_base": "http://localhost:11434"}
                }
                self.reranker_provider = "litellm"

        class DummyConfig:
            def __init__(self):
                self.reranker = DummyReranker()

        mock_config = DummyConfig()
        with patch("cogent.base.providers.reranker.litellm_reranker.get_cogent_config", return_value=mock_config):
            with patch("litellm.acompletion") as mock_litellm:
                mock_response = MagicMock()
                mock_response.choices = [MagicMock()]
                mock_response.choices[0].message.content = "0.8"
                mock_litellm.return_value = mock_response

                reranker = LiteLLMReranker("test_reranker")
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
                assert result[0].score == 0.8

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_compute_score(self):
        """Test computing scores."""

        class DummyReranker:
            def __init__(self):
                self.registered_rerankers = {
                    "test_reranker": {"model_name": "gpt-4", "api_base": "http://localhost:11434"}
                }
                self.reranker_provider = "litellm"

        class DummyConfig:
            def __init__(self):
                self.reranker = DummyReranker()

        mock_config = DummyConfig()
        with patch("cogent.base.providers.reranker.litellm_reranker.get_cogent_config", return_value=mock_config):
            with patch("litellm.acompletion") as mock_litellm:
                mock_response = MagicMock()
                mock_response.choices = [MagicMock()]
                mock_response.choices[0].message.content = "0.8"
                mock_litellm.return_value = mock_response

                reranker = LiteLLMReranker("test_reranker")
                query = "What is machine learning?"
                text = "Machine learning is a subset of AI."

                result = await reranker.compute_score(query, text)

                assert isinstance(result, float)
                assert result == 0.8


class TestCogentRerankerUnit:
    """Unit tests for CogentReranker implementation."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_rerank_flag_provider(self):
        """Test reranking with flag provider."""

        class DummyReranker:
            def __init__(self):
                self.registered_rerankers = {"test_reranker": {"model_name": "BAAI/bge-reranker-v2-gemma"}}
                self.reranker_provider = "flag"

        class DummyConfig:
            def __init__(self):
                self.reranker = DummyReranker()

        mock_config = DummyConfig()
        with patch("cogent.base.providers.reranker.cogent_reranker.get_cogent_config", return_value=mock_config):
            with patch("cogent.base.providers.reranker.flag_reranker.FlagReranker") as mock_flag_reranker:
                mock_reranker_instance = AsyncMock()
                mock_reranker_instance.rerank.return_value = [
                    ObjectChunk(
                        object_id="doc1", content="Test content", embedding=[0.1] * 768, chunk_number=0, score=0.9
                    )
                ]
                mock_flag_reranker.return_value = mock_reranker_instance

                reranker = CogentReranker("test_reranker")
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
                assert result[0].score == 0.9

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_compute_score_flag_provider(self):
        """Test computing scores with flag provider."""

        class DummyReranker:
            def __init__(self):
                self.registered_rerankers = {"test_reranker": {"model_name": "BAAI/bge-reranker-v2-gemma"}}
                self.reranker_provider = "flag"

        class DummyConfig:
            def __init__(self):
                self.reranker = DummyReranker()

        mock_config = DummyConfig()
        with patch("cogent.base.providers.reranker.cogent_reranker.get_cogent_config", return_value=mock_config):
            with patch("cogent.base.providers.reranker.flag_reranker.FlagReranker") as mock_flag_reranker:
                mock_reranker_instance = AsyncMock()
                mock_reranker_instance.compute_score.return_value = 0.8
                mock_flag_reranker.return_value = mock_reranker_instance

                reranker = CogentReranker("test_reranker")
                query = "What is machine learning?"
                text = "Machine learning is a subset of AI."

                result = await reranker.compute_score(query, text)

                assert result == 0.8

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_unsupported_provider(self):
        """Test error handling for unsupported provider."""

        class DummyReranker:
            def __init__(self):
                self.registered_rerankers = {"test_reranker": {"model_name": "BAAI/bge-reranker-v2-gemma"}}
                self.reranker_provider = "unsupported"

        class DummyConfig:
            def __init__(self):
                self.reranker = DummyReranker()

        mock_config = DummyConfig()
        with patch("cogent.base.providers.reranker.cogent_reranker.get_cogent_config", return_value=mock_config):
            with pytest.raises(ValueError, match="Provider 'unsupported' not supported"):
                CogentReranker("test_reranker")


class TestOllamaRerankerUnit:
    """Unit tests for Ollama reranker functionality."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_ollama_reranker_rerank(self):
        """Test Ollama reranker reranking functionality."""

        class DummyReranker:
            def __init__(self):
                self.registered_rerankers = {
                    "ollama_reranker": {"model_name": "ollama/bge-reranker-v2-m3", "api_base": "http://localhost:11434"}
                }
                self.reranker_provider = "ollama"

        class DummyConfig:
            def __init__(self):
                self.reranker = DummyReranker()

        mock_config = DummyConfig()
        with patch("cogent.base.providers.reranker.litellm_reranker.get_cogent_config", return_value=mock_config):
            with patch("cogent.base.providers.reranker.litellm_reranker.ollama") as mock_ollama:
                mock_client = MagicMock()
                mock_ollama.Client.return_value = mock_client
                mock_client.chat = AsyncMock()
                mock_client.chat.return_value.message.content = "0.7"

                reranker = LiteLLMReranker("ollama_reranker")
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
    async def test_ollama_reranker_compute_score(self):
        """Test Ollama reranker compute score functionality."""

        class DummyReranker:
            def __init__(self):
                self.registered_rerankers = {
                    "ollama_reranker": {"model_name": "ollama/bge-reranker-v2-m3", "api_base": "http://localhost:11434"}
                }
                self.reranker_provider = "ollama"

        class DummyConfig:
            def __init__(self):
                self.reranker = DummyReranker()

        mock_config = DummyConfig()
        with patch("cogent.base.providers.reranker.litellm_reranker.get_cogent_config", return_value=mock_config):
            with patch("cogent.base.providers.reranker.litellm_reranker.ollama") as mock_ollama:
                mock_client = MagicMock()
                mock_ollama.Client.return_value = mock_client
                mock_client.chat = AsyncMock()
                mock_client.chat.return_value.message.content = "0.5"

                reranker = LiteLLMReranker("ollama_reranker")
                query = "What is machine learning?"
                text = "Machine learning is a subset of AI."

                result = await reranker.compute_score(query, text)

                assert isinstance(result, float)
                assert abs(result - 0.5) < 1e-6


class TestIntegrationReranker:
    """Integration tests for reranker with external services."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ollama_reranker_integration(self):
        """Integration test for Ollama reranker with real server and ollama_reranker model."""
        # Skip if httpx is not available
        if httpx is None:
            pytest.skip("httpx library not available")

        # Skip if Ollama is not available
        try:
            pass
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

        from cogent.base.providers.reranker.litellm_reranker import LiteLLMReranker

        reranker = LiteLLMReranker("ollama_reranker")
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
