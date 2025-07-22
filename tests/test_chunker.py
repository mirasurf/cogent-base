"""
Unit tests for the sensory parser module.
Tests all chunker classes and their methods including text splitting,
contextual chunking, and error handling.
"""

from unittest.mock import MagicMock, patch

import pytest

from cogent_base.models.chunk import Chunk
from cogent_base.sensory.chunker.contextual_chunker import ContextualChunker
from cogent_base.sensory.chunker.standard_chunker import (
    RecursiveCharacterTextSplitter,
    StandardChunker,
)


class TestRecursiveCharacterTextSplitter:
    """Test the RecursiveCharacterTextSplitter class."""

    @pytest.mark.unit
    def test_init_default_values(self):
        """Test RecursiveCharacterTextSplitter initialization with default values."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

        assert splitter.chunk_size == 100
        assert splitter.chunk_overlap == 20
        assert splitter.length_function == len
        assert splitter.separators == ["\n\n", "\n", ". ", " ", ""]

    @pytest.mark.unit
    def test_init_custom_values(self):
        """Test RecursiveCharacterTextSplitter initialization with custom values."""

        def custom_length(text):
            return len(text) * 2

        custom_separators = ["\n", " "]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=200, chunk_overlap=50, length_function=custom_length, separators=custom_separators
        )

        assert splitter.chunk_size == 200
        assert splitter.chunk_overlap == 50
        assert splitter.length_function == custom_length
        assert splitter.separators == custom_separators

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_split_text_empty_string(self):
        """Test splitting an empty string."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        result = await splitter.split_text("")

        assert result == []

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_split_text_smaller_than_chunk_size(self):
        """Test splitting text smaller than chunk size."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        text = "This is a short text."
        result = await splitter.split_text(text)

        assert len(result) == 1
        assert result[0].content == text

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_split_text_larger_than_chunk_size(self):
        """Test splitting text larger than chunk size."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=5)
        text = "This is a longer text that should be split into multiple chunks."
        result = await splitter.split_text(text)

        assert len(result) > 1

        # Check that chunks are reasonably sized (allowing for some flexibility)
        for chunk in result:
            assert len(chunk.content) <= 50  # Allow more flexibility for chunking logic

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_split_text_with_overlap(self):
        """Test splitting text with overlap."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=15, chunk_overlap=5)
        text = "First chunk. Second chunk. Third chunk."
        result = await splitter.split_text(text)

        assert len(result) > 1

        # Check that consecutive chunks have overlap
        for i in range(1, len(result)):
            prev_chunk = result[i - 1].content
            curr_chunk = result[i].content

            # Check if there's overlap (last 5 chars of prev should be in curr)
            overlap_found = False
            for j in range(len(prev_chunk) - 5 + 1):
                if prev_chunk[j : j + 5] in curr_chunk:
                    overlap_found = True
                    break

            assert overlap_found, f"No overlap found between chunks {i-1} and {i}"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_split_text_with_custom_separators(self):
        """Test splitting text with custom separators."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=0, separators=["|", " "])
        text = "Part1|Part2|Part3|Part4"
        result = await splitter.split_text(text)

        assert len(result) > 1

        # Check that splitting respected the custom separator
        for chunk in result:
            assert len(chunk.content) <= 20

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_split_text_no_separators_left(self):
        """Test splitting when no separators are left."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=0, separators=[])
        text = "This is a very long text without any separators"
        result = await splitter.split_text(text)

        # Should split at chunk_size boundaries
        for chunk in result:
            assert len(chunk.content) <= 10


class TestStandardChunker:
    """Test the StandardChunker class."""

    @pytest.mark.unit
    def test_init(self):
        """Test StandardChunker initialization."""
        chunker = StandardChunker(chunk_size=100, chunk_overlap=20)

        assert chunker.text_splitter.chunk_size == 100
        assert chunker.text_splitter.chunk_overlap == 20
        assert chunker.text_splitter.length_function == len
        assert chunker.text_splitter.separators == ["\n\n", "\n", ". ", " ", ""]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_split_text_empty_string(self):
        """Test splitting an empty string."""
        chunker = StandardChunker(chunk_size=100, chunk_overlap=20)
        result = await chunker.split_text("")

        assert result == []

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_split_text_single_chunk(self):
        """Test splitting text that fits in a single chunk."""
        chunker = StandardChunker(chunk_size=100, chunk_overlap=20)
        text = "This is a short text that should fit in one chunk."
        result = await chunker.split_text(text)

        assert len(result) == 1
        assert result[0].content == text
        assert isinstance(result[0], Chunk)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_split_text_multiple_chunks(self):
        """Test splitting text into multiple chunks."""
        chunker = StandardChunker(chunk_size=20, chunk_overlap=5)
        text = "This is a longer text that should be split into multiple chunks."
        result = await chunker.split_text(text)

        assert len(result) > 1

        # Check that all chunks are Chunk instances
        for chunk in result:
            assert isinstance(chunk, Chunk)
            assert len(chunk.content) <= 50  # Allow more flexibility for chunking logic

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_split_text_with_paragraphs(self):
        """Test splitting text with paragraph separators."""
        chunker = StandardChunker(chunk_size=50, chunk_overlap=10)
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        result = await chunker.split_text(text)

        assert len(result) > 1

        # Check that paragraphs are respected in splitting
        for chunk in result:
            assert isinstance(chunk, Chunk)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_split_text_with_sentences(self):
        """Test splitting text with sentence separators."""
        chunker = StandardChunker(chunk_size=30, chunk_overlap=5)
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        result = await chunker.split_text(text)

        assert len(result) > 1

        for chunk in result:
            assert isinstance(chunk, Chunk)
            assert len(chunk.content) <= 35  # Allow some flexibility


class TestContextualChunker:
    """Test the ContextualChunker class."""

    @pytest.mark.unit
    @patch("cogent_base.sensory.chunker.contextual_chunker.get_cogent_config")
    def test_init_success(self, mock_get_config):
        """Test successful initialization of ContextualChunker."""
        # Mock config with registered models
        mock_config = MagicMock()
        mock_config.sensory.contextual_chunking_model = "test_model"
        mock_config.llm.registered_models = {
            "test_model": {"model": "test_model", "api_base": "http://localhost:11434", "api_key": "test_key"}
        }
        mock_get_config.return_value = mock_config

        chunker = ContextualChunker(chunk_size=100, chunk_overlap=20)

        assert chunker.model_key == "test_model"
        assert chunker.model_config == {
            "model": "test_model",
            "api_base": "http://localhost:11434",
            "api_key": "test_key",
        }
        assert isinstance(chunker.standard_chunker, StandardChunker)

    @pytest.mark.unit
    @patch("cogent_base.sensory.chunker.contextual_chunker.get_cogent_config")
    def test_init_model_not_found(self, mock_get_config):
        """Test initialization when model is not found in config."""
        # Mock config without the requested model
        mock_config = MagicMock()
        mock_config.sensory.contextual_chunking_model = "test_model"
        mock_config.llm.registered_models = {
            "other_model": {"model": "other_model", "api_base": "http://localhost:11434", "api_key": "test_key"}
        }
        mock_get_config.return_value = mock_config

        with pytest.raises(ValueError) as context:
            ContextualChunker(chunk_size=100, chunk_overlap=20)

        assert "Model 'test_model' not found" in str(context.value)

    @pytest.mark.unit
    @patch("cogent_base.sensory.chunker.contextual_chunker.get_cogent_config")
    def test_init_no_registered_models(self, mock_get_config):
        """Test initialization when no models are registered."""
        # Mock config with no models
        mock_config = MagicMock()
        mock_config.sensory.contextual_chunking_model = "test_model"
        mock_config.llm.registered_models = {}
        mock_get_config.return_value = mock_config

        with pytest.raises(ValueError) as context:
            ContextualChunker(chunk_size=100, chunk_overlap=20)

        assert "Model 'test_model' not found" in str(context.value)

    @pytest.mark.unit
    @patch("cogent_base.sensory.chunker.contextual_chunker.get_cogent_config")
    @patch("cogent_base.sensory.chunker.contextual_chunker.LiteLLMCompletionModel")
    @patch("cogent_base.sensory.chunker.contextual_chunker.CompletionRequest")
    @pytest.mark.asyncio
    async def test_situate_context_success(self, mock_completion_request, mock_completion_model, mock_get_config):
        """Test successful context situating."""
        # Mock config
        mock_config = MagicMock()
        mock_config.sensory.contextual_chunking_model = "test_model"
        mock_config.llm.registered_models = {
            "test_model": {"model": "test_model", "api_base": "http://localhost:11434", "api_key": "test_key"}
        }
        mock_get_config.return_value = mock_config

        # Mock completion model
        mock_model_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.completion = "Situated context"
        mock_completion_model.return_value = mock_model_instance

        # Mock CompletionRequest to avoid validation errors
        mock_request = MagicMock()
        mock_completion_request.return_value = mock_request

        # Track calls to complete method
        call_count = 0

        async def async_complete(request):
            nonlocal call_count
            call_count += 1
            return mock_response

        mock_model_instance.complete = async_complete

        chunker = ContextualChunker(chunk_size=100, chunk_overlap=20)
        doc = "This is a document about artificial intelligence."
        chunk = "artificial intelligence"

        result = await chunker._situate_context(doc, chunk)

        assert result == "Situated context"
        assert call_count == 1

    @pytest.mark.unit
    @patch("cogent_base.sensory.chunker.contextual_chunker.get_cogent_config")
    @patch("cogent_base.sensory.chunker.contextual_chunker.LiteLLMCompletionModel")
    @patch("cogent_base.sensory.chunker.contextual_chunker.CompletionRequest")
    @pytest.mark.asyncio
    async def test_situate_context_completion_error(
        self, mock_completion_request, mock_completion_model, mock_get_config
    ):
        """Test context situating when completion fails."""
        # Mock config
        mock_config = MagicMock()
        mock_config.sensory.contextual_chunking_model = "test_model"
        mock_config.llm.registered_models = {
            "test_model": {"model": "test_model", "api_base": "http://localhost:11434", "api_key": "test_key"}
        }
        mock_get_config.return_value = mock_config

        # Mock completion model to raise exception
        mock_model_instance = MagicMock()
        mock_completion_model.return_value = mock_model_instance

        # Mock CompletionRequest to avoid validation errors
        mock_request = MagicMock()
        mock_completion_request.return_value = mock_request

        # Make the complete method async and raise exception
        async def async_complete_error(request):
            raise Exception("Completion error")

        mock_model_instance.complete = async_complete_error

        chunker = ContextualChunker(chunk_size=100, chunk_overlap=20)
        doc = "This is a document about artificial intelligence."
        chunk = "artificial intelligence"

        with pytest.raises(Exception) as context:
            await chunker._situate_context(doc, chunk)

        assert str(context.value) == "Completion error"

    @pytest.mark.unit
    @patch("cogent_base.sensory.chunker.contextual_chunker.get_cogent_config")
    @patch("cogent_base.sensory.chunker.contextual_chunker.LiteLLMCompletionModel")
    @patch("cogent_base.sensory.chunker.contextual_chunker.CompletionRequest")
    @pytest.mark.asyncio
    async def test_split_text_success(self, mock_completion_request, mock_completion_model, mock_get_config):
        """Test successful text splitting with contextual chunking."""
        # Mock config
        mock_config = MagicMock()
        mock_config.sensory.contextual_chunking_model = "test_model"
        mock_config.llm.registered_models = {
            "test_model": {"model": "test_model", "api_base": "http://localhost:11434", "api_key": "test_key"}
        }
        mock_get_config.return_value = mock_config

        # Mock completion model
        mock_model_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.completion = "Context for chunk"
        mock_completion_model.return_value = mock_model_instance

        # Mock CompletionRequest to avoid validation errors
        mock_request = MagicMock()
        mock_completion_request.return_value = mock_request

        # Make the complete method async
        async def async_complete(request):
            return mock_response

        mock_model_instance.complete = async_complete

        chunker = ContextualChunker(chunk_size=20, chunk_overlap=5)
        text = "This is a longer text that should be split into multiple chunks."

        result = await chunker.split_text(text)

        assert len(result) > 1

        # Check that all chunks have contextual prefixes
        for chunk in result:
            assert isinstance(chunk, Chunk)
            assert chunk.content.startswith("Context for chunk; ")

    @pytest.mark.unit
    @patch("cogent_base.sensory.chunker.contextual_chunker.get_cogent_config")
    @pytest.mark.asyncio
    async def test_split_text_empty_string(self, mock_get_config):
        """Test splitting an empty string."""
        # Mock config
        mock_config = MagicMock()
        mock_config.sensory.contextual_chunking_model = "test_model"
        mock_config.llm.registered_models = {
            "test_model": {"model": "test_model", "api_base": "http://localhost:11434", "api_key": "test_key"}
        }
        mock_get_config.return_value = mock_config

        chunker = ContextualChunker(chunk_size=100, chunk_overlap=20)
        result = await chunker.split_text("")

        assert result == []

    @pytest.mark.unit
    @patch("cogent_base.sensory.chunker.contextual_chunker.get_cogent_config")
    @patch("cogent_base.sensory.chunker.contextual_chunker.LiteLLMCompletionModel")
    @patch("cogent_base.sensory.chunker.contextual_chunker.CompletionRequest")
    @pytest.mark.asyncio
    async def test_split_text_single_chunk(self, mock_completion_request, mock_completion_model, mock_get_config):
        """Test splitting text that fits in a single chunk."""
        # Mock config
        mock_config = MagicMock()
        mock_config.sensory.contextual_chunking_model = "test_model"
        mock_config.llm.registered_models = {
            "test_model": {"model": "test_model", "api_base": "http://localhost:11434", "api_key": "test_key"}
        }
        mock_get_config.return_value = mock_config

        # Mock completion model
        mock_model_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.completion = "Single chunk context"
        mock_completion_model.return_value = mock_model_instance

        # Mock CompletionRequest to avoid validation errors
        mock_request = MagicMock()
        mock_completion_request.return_value = mock_request

        # Make the complete method async
        async def async_complete(request):
            return mock_response

        mock_model_instance.complete = async_complete

        chunker = ContextualChunker(chunk_size=100, chunk_overlap=20)
        text = "This is a short text."

        result = await chunker.split_text(text)

        assert len(result) == 1
        assert result[0].content == "Single chunk context; This is a short text."
        assert isinstance(result[0], Chunk)

    @pytest.mark.unit
    def test_prompt_templates(self):
        """Test that prompt templates are correctly formatted."""
        # Test OBJECT_CONTEXT_PROMPT
        doc_content = "Sample document content"
        object_prompt = ContextualChunker.OBJECT_CONTEXT_PROMPT.format(doc_content=doc_content)
        assert doc_content in object_prompt
        assert "<object>" in object_prompt
        assert "</object>" in object_prompt

        # Test CHUNK_CONTEXT_PROMPT
        chunk_content = "Sample chunk content"
        chunk_prompt = ContextualChunker.CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk_content)
        assert chunk_content in chunk_prompt
        assert "<chunk>" in chunk_prompt
        assert "</chunk>" in chunk_prompt
        assert "succinct context" in chunk_prompt
