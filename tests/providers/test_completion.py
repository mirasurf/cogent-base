import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from cogent.base.models.completion import CompletionRequest, CompletionResponse
from cogent.base.providers.completion.litellm_completion import LiteLLMCompletionModel


class PersonSchema(BaseModel):
    """Test schema for structured output testing."""

    name: str
    age: int
    occupation: str


class TestIntegrationLiteLLMCompletion:
    """Integration tests for LiteLLMCompletionModel with external services."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration with test models."""
        return {
            "llm": {
                "registered_models": {
                    "test_ollama": {"model_name": "llama3.2:3b", "api_base": "http://localhost:11434"},
                    "test_openai": {"model_name": "gpt-4o-mini"},
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
        image_data = "".join(
            [
                "iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAIAAACpL3sXAAAALUlEQVR42mNgGAWjYBSMglEwCkYxQK5BhEYMTAxEgHV",
                "EQwMVQGEIQC3KEIFkIMRAwAoTwOP96jCykAAAAASUVORK5C",
                "YII=",
            ]
        )
        return CompletionRequest(
            query="What do you see in this image?",
            context_chunks=[image_data],
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
        with patch("cogent.base.providers.completion.litellm_completion.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]

            # Skip if Ollama is not available
            try:
                pass
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

            model = LiteLLMCompletionModel("test_ollama")
            response = await model.complete(basic_request)

            assert isinstance(response, CompletionResponse)
            assert isinstance(response.completion, str)
            assert len(response.completion) > 0
            assert "Paris" in response.completion or "France" in response.completion
            assert response.usage is not None
            assert "total_tokens" in response.usage

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_openai_basic_completion(self, mock_config, basic_request):
        """Test basic completion with OpenAI model."""
        # Skip if OpenAI API key is not available
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OpenAI API key not available")

        with patch("cogent.base.providers.completion.litellm_completion.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]

            model = LiteLLMCompletionModel("test_openai")
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
        with patch("cogent.base.providers.completion.litellm_completion.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]

            # Skip if Ollama is not available
            try:
                pass
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

            model = LiteLLMCompletionModel("test_ollama")
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
    async def test_openai_structured_completion(self, mock_config, structured_request):
        """Test structured completion with OpenAI model."""
        # Skip if OpenAI API key is not available
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OpenAI API key not available")

        with patch("cogent.base.providers.completion.litellm_completion.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]

            model = LiteLLMCompletionModel("test_openai")
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
        with patch("cogent.base.providers.completion.litellm_completion.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]

            # Skip if Ollama is not available
            try:
                pass
            except ImportError:
                pytest.skip("Ollama library not available")

            # Check if Ollama service is running and vision model is available
            try:
                import httpx

                async with httpx.AsyncClient() as client:
                    response = await client.get("http://localhost:11434/api/tags", timeout=5.0)
                    if response.status_code != 200:
                        pytest.skip("Ollama service not running")

                    # Check if vision model is available
                    models = response.json().get("models", [])
                    vision_model_available = any("qwen2.5vl" in model.get("name", "") for model in models)
                    if not vision_model_available:
                        pytest.skip("Vision model not available in Ollama")
            except Exception:
                pytest.skip("Ollama service not accessible")

            model = LiteLLMCompletionModel("test_ollama_vision")

            # Handle potential model errors gracefully
            try:
                response = await model.complete(vision_request)

                assert isinstance(response, CompletionResponse)
                assert isinstance(response.completion, str)
                assert len(response.completion) > 0
                assert response.usage is not None
            except Exception as e:
                if "model runner has unexpectedly stopped" in str(e):
                    pytest.skip(f"Vision model error: {e}")
                else:
                    raise

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ollama_streaming_completion(self, mock_config, streaming_request):
        """Test streaming completion with Ollama model."""
        with patch("cogent.base.providers.completion.litellm_completion.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]

            # Skip if Ollama is not available
            try:
                pass
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

            model = LiteLLMCompletionModel("test_ollama")
            response_stream = await model.complete(streaming_request)

            assert hasattr(response_stream, "__aiter__")

            chunks = []
            async for chunk in response_stream:
                assert isinstance(chunk, str)
                chunks.append(chunk)

            assert len(chunks) > 0
            full_response = "".join(chunks)
            assert len(full_response) > 0
            assert "robot" in full_response.lower() or "story" in full_response.lower()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_openai_streaming_completion(self, mock_config, streaming_request):
        """Test streaming completion with OpenAI model."""
        # Skip if OpenAI API key is not available
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OpenAI API key not available")

        with patch("cogent.base.providers.completion.litellm_completion.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]

            model = LiteLLMCompletionModel("test_openai")
            response_stream = await model.complete(streaming_request)

            assert hasattr(response_stream, "__aiter__")

            chunks = []
            async for chunk in response_stream:
                assert isinstance(chunk, str)
                chunks.append(chunk)

            assert len(chunks) > 0
            full_response = "".join(chunks)
            assert len(full_response) > 0
            assert "robot" in full_response.lower() or "story" in full_response.lower()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_chat_history_completion(self, mock_config):
        """Test completion with chat history."""
        # Skip if OpenAI API key is not available
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OpenAI API key not available")

        from cogent.base.models.chat import ChatMessage

        request = CompletionRequest(
            query="What was my previous question?",
            context_chunks=[],
            max_tokens=100,
            temperature=0.1,
            chat_history=[
                ChatMessage(role="user", content="What is 2+2?"),
                ChatMessage(role="assistant", content="2+2 equals 4."),
            ],
        )

        with patch("cogent.base.providers.completion.litellm_completion.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]

            model = LiteLLMCompletionModel("test_openai")
            response = await model.complete(request)

            assert isinstance(response, CompletionResponse)
            assert isinstance(response.completion, str)
            assert len(response.completion) > 0
            assert "2+2" in response.completion or "previous" in response.completion.lower()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_custom_prompt_template(self, mock_config):
        """Test completion with custom prompt template."""
        # Skip if OpenAI API key is not available
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OpenAI API key not available")

        request = CompletionRequest(
            query="What is the weather like?",
            context_chunks=["The weather is sunny and warm."],
            max_tokens=100,
            temperature=0.1,
            prompt_template="Context: {context}\nUser Question: {question}\nPlease provide a detailed answer:",
        )

        with patch("cogent.base.providers.completion.litellm_completion.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]

            model = LiteLLMCompletionModel("test_openai")
            response = await model.complete(request)

            assert isinstance(response, CompletionResponse)
            assert isinstance(response.completion, str)
            assert len(response.completion) > 0
            assert "sunny" in response.completion.lower() or "warm" in response.completion.lower()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_error_handling_invalid_model(self):
        """Test error handling for invalid model key."""
        with pytest.raises(ValueError, match="Model 'invalid_model' not found"):
            LiteLLMCompletionModel("invalid_model")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_error_handling_network_failure(self, mock_config, basic_request):
        """Test error handling for network failures."""
        with patch("cogent.base.providers.completion.litellm_completion.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]

            # Test with invalid API base to simulate network failure
            mock_config["llm"]["registered_models"]["test_ollama"]["api_base"] = "http://invalid-host:9999"

            model = LiteLLMCompletionModel("test_ollama")

            with pytest.raises(Exception):
                await model.complete(basic_request)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_dynamic_schema_completion(self, mock_config):
        """Test completion with dynamic JSON schema."""
        # Skip if OpenAI API key is not available
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OpenAI API key not available")

        schema = {
            "type": "object",
            "properties": {"title": {"type": "string"}, "author": {"type": "string"}, "year": {"type": "integer"}},
            "required": ["title", "author", "year"],
        }

        request = CompletionRequest(
            query="Extract book information",
            context_chunks=["The Great Gatsby was written by F. Scott Fitzgerald in 1925."],
            max_tokens=100,
            temperature=0.1,
            schema=schema,
        )

        with patch("cogent.base.providers.completion.litellm_completion.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]

            model = LiteLLMCompletionModel("test_openai")
            response = await model.complete(request)

            assert isinstance(response, CompletionResponse)
            assert hasattr(response.completion, "title")
            assert hasattr(response.completion, "author")
            assert hasattr(response.completion, "year")
            assert response.completion.title == "The Great Gatsby"
            assert response.completion.author == "F. Scott Fitzgerald"
            assert response.completion.year == 1925


class TestLiteLLMCompletionUnit:
    """Unit tests for LiteLLMCompletionModel with mocked dependencies."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for unit tests."""
        return {
            "llm": {
                "registered_models": {
                    "test_model": {"model_name": "gpt-4o-mini"},
                    "test_ollama": {"model_name": "llama3.2:latest", "api_base": "http://localhost:11434"},
                }
            }
        }

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_model_initialization(self, mock_config):
        """Test model initialization with valid configuration."""
        with patch("cogent.base.providers.completion.litellm_completion.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]

            model = LiteLLMCompletionModel("test_model")
            assert model.model_key == "test_model"
            assert model.model_config == mock_config["llm"]["registered_models"]["test_model"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_ollama_detection(self, mock_config):
        """Test Ollama model detection."""
        with patch("cogent.base.providers.completion.litellm_completion.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]

            with patch("cogent.base.providers.completion.litellm_completion.ollama") as mock_ollama:
                mock_ollama.__version__ = "0.1.0"

                model = LiteLLMCompletionModel("test_ollama")
                assert model.is_ollama is True
                assert model.ollama_api_base == "http://localhost:11434"
                assert model.ollama_base_model_name == "llama3.2:latest"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_ollama_fallback_when_library_missing(self, mock_config):
        """Test fallback to LiteLLM when Ollama library is missing."""
        with patch("cogent.base.providers.completion.litellm_completion.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]

            with patch("cogent.base.providers.completion.litellm_completion.initialize_ollama_model") as mock_init:
                # Mock the utility function to return False (fallback to LiteLLM)
                mock_init.return_value = (False, None, None)

                model = LiteLLMCompletionModel("test_ollama")
                assert model.is_ollama is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_ollama_fallback_when_api_base_missing(self, mock_config):
        """Test fallback to LiteLLM when Ollama API base is missing."""
        mock_config["llm"]["registered_models"]["test_ollama"].pop("api_base")

        with patch("cogent.base.providers.completion.litellm_completion.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]

            with patch("cogent.base.providers.completion.litellm_completion.initialize_ollama_model") as mock_init:
                # Mock the utility function to return False (fallback to LiteLLM)
                mock_init.return_value = (False, None, None)

                model = LiteLLMCompletionModel("test_ollama")
                assert model.is_ollama is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mocked_litellm_completion(self, mock_config):
        """Test completion with mocked LiteLLM."""
        with patch("cogent.base.providers.completion.litellm_completion.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]

            with patch("cogent.base.providers.completion.litellm_completion.litellm") as mock_litellm:
                # Create proper mock objects with the expected structure
                mock_usage = MagicMock()
                mock_usage.prompt_tokens = 30
                mock_usage.completion_tokens = 20
                mock_usage.total_tokens = 50

                mock_message = MagicMock()
                mock_message.content = "Paris is the capital of France."

                mock_choice = MagicMock()
                mock_choice.message = mock_message
                mock_choice.finish_reason = "stop"

                mock_response = MagicMock()
                mock_response.choices = [mock_choice]
                mock_response.usage = mock_usage

                mock_litellm.acompletion = AsyncMock(return_value=mock_response)

                model = LiteLLMCompletionModel("test_model")
                request = CompletionRequest(
                    query="What is the capital of France?",
                    context_chunks=["Paris is the capital of France."],
                    max_tokens=100,
                    temperature=0.1,
                )

                response = await model.complete(request)

                assert isinstance(response, CompletionResponse)
                assert response.completion == "Paris is the capital of France."
                assert response.usage["total_tokens"] == 50
                assert response.usage["prompt_tokens"] == 30
                assert response.usage["completion_tokens"] == 20

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mocked_ollama_completion(self, mock_config):
        """Test completion with mocked Ollama."""
        with patch("cogent.base.providers.completion.litellm_completion.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]

            with patch("cogent.base.providers.completion.litellm_completion.ollama") as mock_ollama:
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

                model = LiteLLMCompletionModel("test_ollama")
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
