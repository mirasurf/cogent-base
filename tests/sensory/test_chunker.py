"""
Unit tests for the sensory parser module.
Tests all chunker classes and their methods including text splitting,
contextual chunking, and error handling.
"""

import unittest
from unittest.mock import MagicMock, patch

import pytest

from cogentbase.models.chunk import Chunk
from cogentbase.sensory.chunker.contextual_chunker import ContextualChunker
from cogentbase.sensory.chunker.standard_chunker import (
    RecursiveCharacterTextSplitter,
    StandardChunker,
)


class TestRecursiveCharacterTextSplitter(unittest.TestCase):
    """Test the RecursiveCharacterTextSplitter class."""

    @pytest.mark.unit
    def test_init_default_values(self):
        """Test RecursiveCharacterTextSplitter initialization with default values."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

        self.assertEqual(splitter.chunk_size, 100)
        self.assertEqual(splitter.chunk_overlap, 20)
        self.assertEqual(splitter.length_function, len)
        self.assertEqual(splitter.separators, ["\n\n", "\n", ". ", " ", ""])

    @pytest.mark.unit
    def test_init_custom_values(self):
        """Test RecursiveCharacterTextSplitter initialization with custom values."""

        def custom_length(text):
            return len(text) * 2

        custom_separators = ["\n", " "]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=200, chunk_overlap=50, length_function=custom_length, separators=custom_separators
        )

        self.assertEqual(splitter.chunk_size, 200)
        self.assertEqual(splitter.chunk_overlap, 50)
        self.assertEqual(splitter.length_function, custom_length)
        self.assertEqual(splitter.separators, custom_separators)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_split_text_empty_string(self):
        """Test splitting an empty string."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        result = await splitter.split_text("")

        self.assertEqual(result, [])

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_split_text_smaller_than_chunk_size(self):
        """Test splitting text smaller than chunk size."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        text = "This is a short text."
        result = await splitter.split_text(text)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].content, text)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_split_text_larger_than_chunk_size(self):
        """Test splitting text larger than chunk size."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=5)
        text = "This is a longer text that should be split into multiple chunks."
        result = await splitter.split_text(text)

        self.assertGreater(len(result), 1)

        # Check that chunks are reasonably sized (allowing for some flexibility)
        for chunk in result:
            self.assertLessEqual(len(chunk.content), 25)  # Allow some flexibility

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_split_text_with_overlap(self):
        """Test splitting text with overlap."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=15, chunk_overlap=5)
        text = "First chunk. Second chunk. Third chunk."
        result = await splitter.split_text(text)

        self.assertGreater(len(result), 1)

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

            self.assertTrue(overlap_found, f"No overlap found between chunks {i-1} and {i}")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_split_text_with_custom_separators(self):
        """Test splitting text with custom separators."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=0, separators=["|", " "])
        text = "Part1|Part2|Part3|Part4"
        result = await splitter.split_text(text)

        self.assertGreater(len(result), 1)

        # Check that splitting respected the custom separator
        for chunk in result:
            self.assertLessEqual(len(chunk.content), 20)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_split_text_no_separators_left(self):
        """Test splitting when no separators are left."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=0, separators=[])
        text = "This is a very long text without any separators"
        result = await splitter.split_text(text)

        # Should split at chunk_size boundaries
        for chunk in result:
            self.assertLessEqual(len(chunk.content), 10)


class TestStandardChunker(unittest.TestCase):
    """Test the StandardChunker class."""

    @pytest.mark.unit
    def test_init(self):
        """Test StandardChunker initialization."""
        chunker = StandardChunker(chunk_size=100, chunk_overlap=20)

        self.assertEqual(chunker.text_splitter.chunk_size, 100)
        self.assertEqual(chunker.text_splitter.chunk_overlap, 20)
        self.assertEqual(chunker.text_splitter.length_function, len)
        self.assertEqual(chunker.text_splitter.separators, ["\n\n", "\n", ". ", " ", ""])

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_split_text_empty_string(self):
        """Test splitting an empty string."""
        chunker = StandardChunker(chunk_size=100, chunk_overlap=20)
        result = await chunker.split_text("")

        self.assertEqual(result, [])

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_split_text_single_chunk(self):
        """Test splitting text that fits in a single chunk."""
        chunker = StandardChunker(chunk_size=100, chunk_overlap=20)
        text = "This is a short text that should fit in one chunk."
        result = await chunker.split_text(text)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].content, text)
        self.assertIsInstance(result[0], Chunk)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_split_text_multiple_chunks(self):
        """Test splitting text into multiple chunks."""
        chunker = StandardChunker(chunk_size=20, chunk_overlap=5)
        text = "This is a longer text that should be split into multiple chunks."
        result = await chunker.split_text(text)

        self.assertGreater(len(result), 1)

        # Check that all chunks are Chunk instances
        for chunk in result:
            self.assertIsInstance(chunk, Chunk)
            self.assertLessEqual(len(chunk.content), 25)  # Allow some flexibility

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_split_text_with_paragraphs(self):
        """Test splitting text with paragraph separators."""
        chunker = StandardChunker(chunk_size=50, chunk_overlap=10)
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        result = await chunker.split_text(text)

        self.assertGreater(len(result), 1)

        # Check that paragraphs are respected in splitting
        for chunk in result:
            self.assertIsInstance(chunk, Chunk)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_split_text_with_sentences(self):
        """Test splitting text with sentence separators."""
        chunker = StandardChunker(chunk_size=30, chunk_overlap=5)
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        result = await chunker.split_text(text)

        self.assertGreater(len(result), 1)

        for chunk in result:
            self.assertIsInstance(chunk, Chunk)
            self.assertLessEqual(len(chunk.content), 35)  # Allow some flexibility


class TestContextualChunker(unittest.TestCase):
    """Test the ContextualChunker class."""

    @pytest.mark.unit
    @patch("cogentbase.sensory.chunker.contextual_chunker.get_cogent_config")
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

        self.assertEqual(chunker.model_key, "test_model")
        self.assertEqual(
            chunker.model_config, {"model": "test_model", "api_base": "http://localhost:11434", "api_key": "test_key"}
        )
        self.assertIsInstance(chunker.standard_chunker, StandardChunker)

    @pytest.mark.unit
    @patch("cogentbase.sensory.chunker.contextual_chunker.get_cogent_config")
    def test_init_model_not_found(self, mock_get_config):
        """Test initialization when model is not found in config."""
        # Mock config without the requested model
        mock_config = MagicMock()
        mock_config.sensory.contextual_chunking_model = "test_model"
        mock_config.llm.registered_models = {
            "other_model": {"model": "other_model", "api_base": "http://localhost:11434", "api_key": "test_key"}
        }
        mock_get_config.return_value = mock_config

        with self.assertRaises(ValueError) as context:
            ContextualChunker(chunk_size=100, chunk_overlap=20)

        self.assertIn("Model 'test_model' not found", str(context.exception))

    @pytest.mark.unit
    @patch("cogentbase.sensory.chunker.contextual_chunker.get_cogent_config")
    def test_init_no_registered_models(self, mock_get_config):
        """Test initialization when no models are registered."""
        # Mock config with no models
        mock_config = MagicMock()
        mock_config.sensory.contextual_chunking_model = "test_model"
        mock_config.llm.registered_models = {}
        mock_get_config.return_value = mock_config

        with self.assertRaises(ValueError) as context:
            ContextualChunker(chunk_size=100, chunk_overlap=20)

        self.assertIn("Model 'test_model' not found", str(context.exception))

    @pytest.mark.unit
    @patch("cogentbase.sensory.chunker.contextual_chunker.get_cogent_config")
    @patch("cogentbase.sensory.chunker.contextual_chunker.LiteLLMCompletionModel")
    @pytest.mark.asyncio
    async def test_situate_context_success(self, mock_completion_model, mock_get_config):
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
        mock_model_instance.complete.return_value = mock_response
        mock_completion_model.return_value = mock_model_instance

        chunker = ContextualChunker(chunk_size=100, chunk_overlap=20)
        doc = "This is a document about artificial intelligence."
        chunk = "artificial intelligence"

        result = await chunker._situate_context(doc, chunk)

        self.assertEqual(result, "Situated context")
        mock_model_instance.complete.assert_called_once()

    @pytest.mark.unit
    @patch("cogentbase.sensory.chunker.contextual_chunker.get_cogent_config")
    @patch("cogentbase.sensory.chunker.contextual_chunker.LiteLLMCompletionModel")
    @pytest.mark.asyncio
    async def test_situate_context_completion_error(self, mock_completion_model, mock_get_config):
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
        mock_model_instance.complete.side_effect = Exception("Completion error")
        mock_completion_model.return_value = mock_model_instance

        chunker = ContextualChunker(chunk_size=100, chunk_overlap=20)
        doc = "This is a document about artificial intelligence."
        chunk = "artificial intelligence"

        with self.assertRaises(Exception) as context:
            await chunker._situate_context(doc, chunk)

        self.assertEqual(str(context.exception), "Completion error")

    @pytest.mark.unit
    @patch("cogentbase.sensory.chunker.contextual_chunker.get_cogent_config")
    @patch("cogentbase.sensory.chunker.contextual_chunker.LiteLLMCompletionModel")
    @pytest.mark.asyncio
    async def test_split_text_success(self, mock_completion_model, mock_get_config):
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
        mock_model_instance.complete.return_value = mock_response
        mock_completion_model.return_value = mock_model_instance

        chunker = ContextualChunker(chunk_size=20, chunk_overlap=5)
        text = "This is a longer text that should be split into multiple chunks."

        result = await chunker.split_text(text)

        self.assertGreater(len(result), 1)

        # Check that all chunks have contextual prefixes
        for chunk in result:
            self.assertIsInstance(chunk, Chunk)
            self.assertTrue(chunk.content.startswith("Context for chunk; "))

    @pytest.mark.unit
    @patch("cogentbase.sensory.chunker.contextual_chunker.get_cogent_config")
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

        self.assertEqual(result, [])

    @pytest.mark.unit
    @patch("cogentbase.sensory.chunker.contextual_chunker.get_cogent_config")
    @patch("cogentbase.sensory.chunker.contextual_chunker.LiteLLMCompletionModel")
    @pytest.mark.asyncio
    async def test_split_text_single_chunk(self, mock_completion_model, mock_get_config):
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
        mock_model_instance.complete.return_value = mock_response
        mock_completion_model.return_value = mock_model_instance

        chunker = ContextualChunker(chunk_size=100, chunk_overlap=20)
        text = "This is a short text."

        result = await chunker.split_text(text)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].content, "Single chunk context; This is a short text.")
        self.assertIsInstance(result[0], Chunk)

    @pytest.mark.unit
    def test_prompt_templates(self):
        """Test that prompt templates are correctly formatted."""
        # Test OBJECT_CONTEXT_PROMPT
        doc_content = "Sample document content"
        object_prompt = ContextualChunker.OBJECT_CONTEXT_PROMPT.format(doc_content=doc_content)
        self.assertIn(doc_content, object_prompt)
        self.assertIn("<object>", object_prompt)
        self.assertIn("</object>", object_prompt)

        # Test CHUNK_CONTEXT_PROMPT
        chunk_content = "Sample chunk content"
        chunk_prompt = ContextualChunker.CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk_content)
        self.assertIn(chunk_content, chunk_prompt)
        self.assertIn("<chunk>", chunk_prompt)
        self.assertIn("</chunk>", chunk_prompt)
        self.assertIn("succinct context", chunk_prompt)


if __name__ == "__main__":
    unittest.main()
