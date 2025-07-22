from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from cogent_base.completion.ollama_completion import OllamaCompletionModel
from cogent_base.models.completion import CompletionRequest, CompletionResponse


class PersonSchema(BaseModel):
    """Test schema for structured output testing."""

    name: str
    age: int
    occupation: str


class TestIntegrationOllamaCompletion:
    """Integration tests for OllamaCompletionModel with external services."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration with test models."""
        return {
            "llm": {
                "registered_models": {
                    "test_ollama": {"model_name": "llama3.2:3b", "api_base": "http://localhost:11434"},
                    "test_ollama_vision": {
                        "model_name": "qwen2.5vl:3b",
                        "api_base": "http://localhost:11434",
                        "vision": True,
                    },
                }
            }
        }

    @pytest.fixture
    def basic_request(self):
        """Basic completion request for testing."""
        return CompletionRequest(
            query="What is the capital of France?",
            context_chunks=["Paris is the capital of France."],
            max_tokens=100,
            temperature=0.1,
        )

    @pytest.fixture
    def structured_request(self):
        """Structured completion request for testing."""
        return CompletionRequest(
            query="Extract person information from the text",
            context_chunks=["John Smith is a 30-year-old software engineer."],
            max_tokens=100,
            temperature=0.1,
            schema=PersonSchema,
        )

    @pytest.fixture
    def vision_request(self):
        """Vision completion request for testing."""
        # Valid 1x1 red pixel PNG image in base64
        image_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
        return CompletionRequest(
            query="What do you see in this image?",
            context_chunks=[f"data:image/png;base64,{image_data}"],
            max_tokens=100,
            temperature=0.1,
        )

    @pytest.fixture
    def streaming_request(self):
        """Streaming completion request for testing."""
        return CompletionRequest(
            query="Write a short story about a robot.",
            context_chunks=[],
            max_tokens=200,
            temperature=0.7,
            stream_response=True,
        )

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ollama_basic_completion(self, mock_config, basic_request):
        """Test basic completion with Ollama model."""
        with patch("cogent_base.completion.ollama_completion.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]

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

            model = OllamaCompletionModel("test_ollama")
            response = await model.complete(basic_request)

            assert isinstance(response, CompletionResponse)
            assert isinstance(response.completion, str)
            assert len(response.completion) > 0
            assert "Paris" in response.completion or "France" in response.completion
            assert response.usage is not None
            assert "total_tokens" in response.usage

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ollama_structured_completion(self, mock_config, structured_request):
        """Test structured completion with Ollama model."""
        with patch("cogent_base.completion.ollama_completion.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]

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

            model = OllamaCompletionModel("test_ollama")
            response = await model.complete(structured_request)

            assert isinstance(response, CompletionResponse)
            assert isinstance(response.completion, PersonSchema)
            assert response.completion.name == "John Smith"
            assert response.completion.age == 30
            # Case-insensitive check for occupation
            assert response.completion.occupation.lower() == "software engineer"
            assert response.usage is not None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ollama_vision_completion(self, mock_config, vision_request):
        """Test vision completion with Ollama model."""
        with patch("cogent_base.completion.ollama_completion.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]

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

            model = OllamaCompletionModel("test_ollama_vision")

            # Test vision completion with error handling
            try:
                response = await model.complete(vision_request)

                assert isinstance(response, CompletionResponse)
                assert isinstance(response.completion, str)
                assert len(response.completion) > 0
                assert response.usage is not None

            except Exception as e:
                # If the model fails due to resource limitations or other issues,
                # we'll skip the test rather than fail it
                if "resource limitations" in str(e) or "unexpectedly stopped" in str(e):
                    pytest.skip(f"Vision model failed due to resource limitations: {e}")
                else:
                    # Re-raise other exceptions
                    raise

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ollama_streaming_completion(self, mock_config, streaming_request):
        """Test streaming completion with Ollama model."""
        with patch("cogent_base.completion.ollama_completion.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]

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

            model = OllamaCompletionModel("test_ollama")
            response_stream = await model.complete(streaming_request)

            # Collect all chunks from the async generator
            chunks = []
            async for chunk in response_stream:
                chunks.append(chunk)
                if len(chunks) >= 5:  # Limit to avoid long test runs
                    break

            assert len(chunks) > 0
            # Verify that each chunk is a string
            for chunk in chunks:
                assert isinstance(chunk, str)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ollama_invalid_model_error(self, mock_config):
        """Test error handling with invalid model configuration."""
        with patch("cogent_base.completion.ollama_completion.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]

            with pytest.raises(ValueError, match="Model 'invalid_model' not found in registered_models"):
                OllamaCompletionModel("invalid_model")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ollama_context_handling(self, mock_config):
        """Test context handling with Ollama model."""
        with patch("cogent_base.completion.ollama_completion.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]

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

            model = OllamaCompletionModel("test_ollama")

            # Test without context
            request_no_context = CompletionRequest(
                query="Tell me about artificial intelligence",
                context_chunks=[],
                max_tokens=50,
                temperature=0.1,
            )

            response = await model.complete(request_no_context)
            assert isinstance(response, CompletionResponse)
            assert len(response.completion) > 0


class TestUnitOllamaCompletion:
    """Unit tests for OllamaCompletionModel with mocked dependencies."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for unit tests."""
        return {
            "llm": {
                "registered_models": {
                    "test_ollama": {"model_name": "llama3.2:latest", "api_base": "http://localhost:11434"},
                }
            }
        }

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_model_initialization(self, mock_config):
        """Test model initialization with valid configuration."""
        with patch("cogent_base.completion.ollama_completion.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]

            with patch("cogent_base.completion.ollama_completion.ollama") as mock_ollama:
                mock_ollama.__version__ = "0.1.0"

                model = OllamaCompletionModel("test_ollama")
                assert model.model_key == "test_ollama"
                assert model.model_config == mock_config["llm"]["registered_models"]["test_ollama"]
                assert model.ollama_api_base == "http://localhost:11434"
                assert model.ollama_base_model_name == "llama3.2:latest"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_ollama_library_not_installed(self, mock_config):
        """Test error when Ollama library is not installed."""
        with patch("cogent_base.completion.ollama_completion.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]

            with patch("cogent_base.completion.ollama_completion.ollama", None):
                with pytest.raises(ImportError, match="Ollama library not installed"):
                    OllamaCompletionModel("test_ollama")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_non_ollama_model_error(self, mock_config):
        """Test error when model is not configured as Ollama model."""
        mock_config["llm"]["registered_models"]["test_non_ollama"] = {"model_name": "gpt-4"}

        with patch("cogent_base.completion.ollama_completion.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]

            with patch("cogent_base.completion.ollama_completion.ollama") as mock_ollama:
                mock_ollama.__version__ = "0.1.0"

                with patch("cogent_base.completion.ollama_completion.initialize_ollama_model") as mock_init:
                    mock_init.return_value = (False, None, None)

                    with pytest.raises(ValueError, match="is not configured as an Ollama model"):
                        OllamaCompletionModel("test_non_ollama")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mocked_ollama_completion(self, mock_config):
        """Test completion with mocked Ollama."""
        with patch("cogent_base.completion.ollama_completion.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]

            with patch("cogent_base.completion.ollama_completion.ollama") as mock_ollama:
                mock_ollama.__version__ = "0.1.0"

                # Create a proper mock client that can be awaited
                mock_client = AsyncMock()
                mock_client.chat = AsyncMock(
                    return_value={
                        "message": {"content": "Paris is the capital of France."},
                        "prompt_eval_count": 30,
                        "eval_count": 20,
                        "done_reason": "stop",
                    }
                )

                # Mock the AsyncClient constructor to return our mock client
                mock_ollama.AsyncClient = MagicMock(return_value=mock_client)

                model = OllamaCompletionModel("test_ollama")
                request = CompletionRequest(
                    query="What is the capital of France?",
                    context_chunks=["Paris is the capital of France."],
                    max_tokens=100,
                    temperature=0.1,
                )

                response = await model.complete(request)

                assert isinstance(response, CompletionResponse)
                assert "Paris" in response.completion
                assert response.usage is not None
                assert response.usage["total_tokens"] == 50  # 30 + 20
