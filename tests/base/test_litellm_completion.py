import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from cogent.base.completion.litellm_completion import LiteLLMCompletionModel
from cogent.base.models.completion import CompletionRequest, CompletionResponse


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
                    "test_dashscope": {
                        "model_name": "dashscope/qwen3-30b-a3b",
                        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                        "provider": "dashscope",
                    },
                    "test_anthropic": {"model_name": "claude-3-haiku-20240307"},
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
            model_kwargs={"enable_thinking": False},
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
            model_kwargs={"enable_thinking": False},
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
            context_chunks=[f"data:image/png;base64,{image_data}"],
            max_tokens=100,
            temperature=0.1,
            model_kwargs={"enable_thinking": False},
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
            model_kwargs={"enable_thinking": False},
        )

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_dashscope_basic_completion(self, mock_config, basic_request):
        """Test basic completion with DashScope model."""
        # Skip if DashScope API key is not available
        if not os.getenv("DASHSCOPE_API_KEY"):
            pytest.skip("DashScope API key not available")

        with patch("cogent.base.completion.litellm_completion.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]

            model = LiteLLMCompletionModel("test_dashscope")
            response = await model.complete(basic_request)

            assert isinstance(response, CompletionResponse)
            assert isinstance(response.completion, str)
            assert len(response.completion) > 0
            assert "Paris" in response.completion or "France" in response.completion
            assert response.usage is not None
            assert "total_tokens" in response.usage

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_dashscope_structured_completion(self, mock_config, structured_request):
        """Test structured completion with DashScope model."""
        # Skip if DashScope API key is not available
        if not os.getenv("DASHSCOPE_API_KEY"):
            pytest.skip("DashScope API key not available")

        with patch("cogent.base.completion.litellm_completion.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]

            model = LiteLLMCompletionModel("test_dashscope")
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
    async def test_dashscope_vision_completion(self, mock_config, vision_request):
        """Test vision completion with DashScope model."""
        # Skip if DashScope API key is not available
        if not os.getenv("DASHSCOPE_API_KEY"):
            pytest.skip("DashScope API key not available")

        with patch("cogent.base.completion.litellm_completion.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]

            model = LiteLLMCompletionModel("test_dashscope")
            response = await model.complete(vision_request)

            assert isinstance(response, CompletionResponse)
            assert isinstance(response.completion, str)
            assert len(response.completion) > 0
            assert response.usage is not None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_dashscope_streaming_completion(self, mock_config, streaming_request):
        """Test streaming completion with DashScope model."""
        # Skip if DashScope API key is not available
        if not os.getenv("DASHSCOPE_API_KEY"):
            pytest.skip("DashScope API key not available")

        with patch("cogent.base.completion.litellm_completion.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]

            model = LiteLLMCompletionModel("test_dashscope")
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
    async def test_anthropic_basic_completion(self, mock_config, basic_request):
        """Test basic completion with Anthropic model."""
        # Skip if Anthropic API key is not available
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("Anthropic API key not available")

        with patch("cogent.base.completion.litellm_completion.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]

            model = LiteLLMCompletionModel("test_anthropic")
            response = await model.complete(basic_request)

            assert isinstance(response, CompletionResponse)
            assert isinstance(response.completion, str)
            assert len(response.completion) > 0
            assert "Paris" in response.completion or "France" in response.completion
            assert response.usage is not None
            assert "total_tokens" in response.usage

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_dashscope_structured_completion_with_json_schema(self, mock_config):
        """Test structured completion with JSON schema instead of Pydantic model (DashScope)."""
        # Skip if DashScope API key is not available
        if not os.getenv("DASHSCOPE_API_KEY"):
            pytest.skip("DashScope API key not available")

        with patch("cogent.base.completion.litellm_completion.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]

            json_schema = {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "occupation": {"type": "string"},
                },
            }

            request = CompletionRequest(
                query="Extract person information from the text",
                context_chunks=["John Smith is a 30-year-old software engineer."],
                max_tokens=100,
                temperature=0.1,
                schema=json_schema,
                model_kwargs={"enable_thinking": False},
            )

            model = LiteLLMCompletionModel("test_dashscope")
            response = await model.complete(request)

            assert isinstance(response, CompletionResponse)
            # The response should be a pydantic model created dynamically
            assert hasattr(response.completion, "name")
            assert hasattr(response.completion, "age")
            assert hasattr(response.completion, "occupation")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_invalid_model_error(self, mock_config):
        """Test error handling with invalid model configuration."""
        with patch("cogent.base.completion.litellm_completion.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]

            with pytest.raises(ValueError, match="not found in registered_models"):
                LiteLLMCompletionModel("invalid_model")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_dashscope_context_handling(self, mock_config):
        """Test context handling with DashScope model."""
        # Skip if DashScope API key is not available
        if not os.getenv("DASHSCOPE_API_KEY"):
            pytest.skip("DashScope API key not available")

        with patch("cogent.base.completion.litellm_completion.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]

            model = LiteLLMCompletionModel("test_dashscope")

            # Test without context
            request_no_context = CompletionRequest(
                query="Tell me about artificial intelligence",
                context_chunks=[],
                max_tokens=50,
                temperature=0.1,
                model_kwargs={"enable_thinking": False},
            )

            response = await model.complete(request_no_context)
            assert isinstance(response, CompletionResponse)
            assert len(response.completion) > 0


class TestUnitLiteLLMCompletion:
    """Unit tests for LiteLLMCompletionModel with mocked dependencies."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for unit tests."""
        return {
            "llm": {
                "registered_models": {
                    "test_model": {"model_name": "gpt-4o-mini"},
                }
            }
        }

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_model_initialization(self, mock_config):
        """Test model initialization with valid configuration."""
        with patch("cogent.base.completion.litellm_completion.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]

            model = LiteLLMCompletionModel("test_model")
            assert model.model_key == "test_model"
            assert model.model_config == mock_config["llm"]["registered_models"]["test_model"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_mocked_litellm_completion(self, mock_config):
        """Test completion with mocked LiteLLM."""
        with patch("cogent.base.completion.litellm_completion.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.llm.registered_models = mock_config["llm"]["registered_models"]

            with patch("cogent.base.completion.litellm_completion.litellm") as mock_litellm:
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
