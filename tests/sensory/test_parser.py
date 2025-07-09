"""
Unit tests for CogentParser class.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from cogent.base.models.chunk import Chunk
from cogent.base.sensory.parser.base_parser import ParsedElement
from cogent.base.sensory.parser.cogent_parser import CogentParser


class TestCogentParser:
    """Test cases for CogentParser class."""

    @pytest.mark.unit
    @pytest.fixture
    def parser(self):
        """Create a basic CogentParser instance for testing."""
        return CogentParser(
            chunk_size=1000,
            chunk_overlap=200,
            use_unstructured_api=False,
            assemblyai_api_key="test_key",
        )

    @pytest.mark.unit
    @pytest.fixture
    def contextual_parser(self):
        """Create a CogentParser instance with contextual chunking."""
        return CogentParser(
            chunk_size=1000,
            chunk_overlap=200,
            use_contextual_chunking=True,
            assemblyai_api_key="test_key",
        )

    @pytest.mark.unit
    @pytest.fixture
    def sample_text_file(self):
        """Create sample text file content."""
        return b"This is a sample text file content for testing."

    @pytest.mark.unit
    @pytest.fixture
    def sample_video_file(self):
        """Create sample video file content (mock)."""
        # Create a mock video file with proper MIME type detection
        return b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"

    @pytest.mark.unit
    @pytest.fixture
    def sample_docx_file(self):
        """Create sample DOCX file content."""
        return b"PK\x03\x04\x14\x00\x00\x00\x08\x00"

    @pytest.mark.unit
    def test_init_standard_chunking(self):
        """Test parser initialization with standard chunking."""
        parser = CogentParser(chunk_size=500, chunk_overlap=100)
        assert parser.use_unstructured_api is False
        assert parser._unstructured_api_key is None
        assert parser._assemblyai_api_key is None
        assert parser.frame_sample_rate == 1
        assert hasattr(parser.chunker, "split_text")

    @pytest.mark.unit
    def test_init_contextual_chunking(self):
        """Test parser initialization with contextual chunking."""
        parser = CogentParser(
            chunk_size=500,
            chunk_overlap=100,
            use_contextual_chunking=True,
        )
        assert hasattr(parser.chunker, "split_text")

    @pytest.mark.unit
    def test_init_with_api_keys(self):
        """Test parser initialization with API keys."""
        parser = CogentParser(
            unstructured_api_key="test_unstructured_key",
            assemblyai_api_key="test_assemblyai_key",
        )
        assert parser._unstructured_api_key == "test_unstructured_key"
        assert parser._assemblyai_api_key == "test_assemblyai_key"

    @pytest.mark.unit
    @patch("cogent.base.sensory.parser.cogent_parser.filetype")
    def test_is_video_file_true(self, mock_filetype, parser):
        """Test video file detection for actual video files."""
        mock_kind = Mock()
        mock_kind.mime = "video/mp4"
        mock_filetype.guess.return_value = mock_kind

        result = parser._is_video_file(b"fake_video_content", "test.mp4")
        assert result is True

    @pytest.mark.unit
    @patch("cogent.base.sensory.parser.cogent_parser.filetype")
    def test_is_video_file_false(self, mock_filetype, parser):
        """Test video file detection for non-video files."""
        mock_kind = Mock()
        mock_kind.mime = "text/plain"
        mock_filetype.guess.return_value = mock_kind

        result = parser._is_video_file(b"fake_text_content", "test.txt")
        assert result is False

    @pytest.mark.unit
    @patch("cogent.base.sensory.parser.cogent_parser.filetype")
    def test_is_video_file_none_kind(self, mock_filetype, parser):
        """Test video file detection when filetype.guess returns None."""
        mock_filetype.guess.return_value = None

        result = parser._is_video_file(b"fake_content", "test.unknown")
        assert result is False

    @pytest.mark.unit
    @patch("cogent.base.sensory.parser.cogent_parser.filetype")
    def test_is_video_file_exception(self, mock_filetype, parser):
        """Test video file detection with exception handling."""
        mock_filetype.guess.side_effect = ValueError("Test error")

        result = parser._is_video_file(b"fake_content", "test.mp4")
        assert result is False

    @pytest.mark.unit
    @patch("cogent.base.sensory.parser.cogent_parser.VideoParser")
    @patch("cogent.base.sensory.parser.cogent_parser.get_cogent_config")
    @patch("cogent.base.sensory.parser.cogent_parser.tempfile")
    @patch("cogent.base.sensory.parser.cogent_parser.os")
    async def test_parse_video_success(self, mock_os, mock_tempfile, mock_get_config, mock_video_parser, parser):
        """Test successful video parsing."""
        # Mock temporary file
        mock_temp_file = Mock()
        mock_temp_file.name = "/tmp/test_video.mp4"
        mock_tempfile.NamedTemporaryFile.return_value = mock_temp_file

        # Mock config
        mock_config = {"parser": {"vision": {"frame_sample_rate": 30}}}
        mock_get_config.return_value = mock_config

        # Mock video parser results
        mock_results = Mock()
        mock_results.metadata = {"duration": 10.0, "fps": 30.0}
        mock_results.frame_descriptions.time_to_content = {1.0: "Frame 1", 2.0: "Frame 2"}
        mock_results.transcript.time_to_content = {0.5: "Hello", 1.5: "World"}

        mock_parser_instance = AsyncMock()
        mock_parser_instance.process_video.return_value = mock_results
        mock_video_parser.return_value = mock_parser_instance

        # Mock file existence check
        mock_os.path.exists.return_value = True

        result_metadata, result_elements = await parser._parse_video(b"fake_video_content")

        # Verify results
        assert len(result_elements) == 1
        assert "Frame Descriptions:" in result_elements[0].text
        assert "Transcript:" in result_elements[0].text
        assert result_metadata["video_metadata"] == mock_results.metadata
        assert result_metadata["frame_timestamps"] == [1.0, 2.0]
        assert result_metadata["transcript_timestamps"] == [0.5, 1.5]

        # Verify cleanup
        mock_os.unlink.assert_called_once_with("/tmp/test_video.mp4")

    @pytest.mark.unit
    async def test_parse_video_no_api_key(self, parser):
        """Test video parsing without API key."""
        parser._assemblyai_api_key = None

        with pytest.raises(ValueError, match="AssemblyAI API key is required for video parsing"):
            await parser._parse_video(b"fake_video_content")

    @pytest.mark.unit
    @patch("cogent.base.sensory.parser.cogent_parser.partition")
    async def test_parse_object_success(self, mock_partition, parser):
        """Test successful object parsing."""
        # Mock unstructured partition results
        mock_part1 = Mock()
        mock_part1.category = "Title"
        mock_part1.text = "Sample Title"
        mock_part1.metadata = Mock()
        mock_part1.metadata.languages = ["en"]
        mock_part1.metadata.page_number = 1
        mock_part1.metadata.coordinates = None

        mock_part2 = Mock()
        mock_part2.category = "Text"
        mock_part2.text = "Sample content"
        mock_part2.metadata = Mock()
        mock_part2.metadata.languages = ["en"]
        mock_part2.metadata.page_number = 1
        mock_part2.metadata.coordinates = Mock()
        mock_part2.metadata.coordinates.points = ((0, 0), (100, 100))
        mock_part2.metadata.text_as_html = "<p>Sample content</p>"

        mock_partition.return_value = [mock_part1, mock_part2]

        metadata, elements = await parser._parse_object(b"fake_content", "test.docx")

        # Verify results
        assert len(elements) == 2
        assert elements[0].category == "Title"
        assert elements[0].text == "Sample Title"
        assert elements[0].langs == ["en"]
        assert elements[0].page_number == 1

        assert elements[1].category == "Text"
        assert elements[1].text == "Sample content"
        assert elements[1].box == ((0, 0), (100, 100))
        assert elements[1].text_html == "<p>Sample content</p>"

        # Verify partition was called with correct parameters
        mock_partition.assert_called_once()
        call_args = mock_partition.call_args
        assert call_args[1]["strategy"] == "fast"  # DOCX should use fast strategy
        assert call_args[1]["api_key"] is None  # use_unstructured_api is False

    @pytest.mark.unit
    @patch("cogent.base.sensory.parser.cogent_parser.partition")
    async def test_parse_object_txt_file(self, mock_partition, parser):
        """Test object parsing for text files."""
        mock_partition.return_value = []

        await parser._parse_object(b"fake_content", "test.txt")

        call_args = mock_partition.call_args
        assert call_args[1]["strategy"] == "fast"
        assert call_args[1]["content_type"] == "text/plain"

    @pytest.mark.unit
    @patch("cogent.base.sensory.parser.cogent_parser.partition")
    async def test_parse_object_json_file(self, mock_partition, parser):
        """Test object parsing for JSON files."""
        mock_partition.return_value = []

        await parser._parse_object(b'{"key": "value"}', "test.json")

        call_args = mock_partition.call_args
        assert call_args[1]["strategy"] == "fast"
        assert call_args[1]["content_type"] == "application/json"

    @pytest.mark.unit
    @patch("cogent.base.sensory.parser.cogent_parser.partition")
    async def test_parse_object_with_unstructured_api(self, mock_partition):
        """Test object parsing with unstructured API enabled."""
        parser = CogentParser(use_unstructured_api=True, unstructured_api_key="test_key")
        mock_partition.return_value = []

        await parser._parse_object(b"fake_content", "test.pdf")

        call_args = mock_partition.call_args
        assert call_args[1]["api_key"] == "test_key"

    @pytest.mark.unit
    @patch("cogent.base.sensory.parser.cogent_parser.filetype")
    async def test_parse_file_to_text_video(self, mock_filetype, parser):
        """Test parse_file_to_text for video files."""
        # Mock video detection
        mock_kind = Mock()
        mock_kind.mime = "video/mp4"
        mock_filetype.guess.return_value = mock_kind

        # Mock video parsing
        with patch.object(parser, "_parse_video") as mock_parse_video:
            mock_parse_video.return_value = ({}, [ParsedElement(text="video content")])

            metadata, elements = await parser.parse_file_to_text(b"fake_video", "test.mp4")

            mock_parse_video.assert_called_once_with(b"fake_video")
            assert len(elements) == 1
            assert elements[0].text == "video content"

    @pytest.mark.unit
    @patch("cogent.base.sensory.parser.cogent_parser.filetype")
    async def test_parse_file_to_text_object(self, mock_filetype, parser):
        """Test parse_file_to_text for non-video files."""
        # Mock non-video detection
        mock_kind = Mock()
        mock_kind.mime = "text/plain"
        mock_filetype.guess.return_value = mock_kind

        # Mock object parsing
        with patch.object(parser, "_parse_object") as mock_parse_object:
            mock_parse_object.return_value = ({}, [ParsedElement(text="text content")])

            metadata, elements = await parser.parse_file_to_text(b"fake_text", "test.txt")

            mock_parse_object.assert_called_once_with(b"fake_text", "test.txt")
            assert len(elements) == 1
            assert elements[0].text == "text content"

    @pytest.mark.unit
    async def test_split_text_standard_chunking(self, parser):
        """Test text splitting with standard chunking."""
        text = "This is a test text that should be split into chunks. " * 10

        chunks = await parser.split_text(text)

        assert isinstance(chunks, list)
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert len(chunks) > 0

    @pytest.mark.unit
    @patch("cogent.base.sensory.chunker.contextual_chunker.ContextualChunker")
    async def test_split_text_contextual_chunking(self, mock_contextual_chunker_class, contextual_parser):
        """Test text splitting with contextual chunking."""
        # Mock the contextual chunker to avoid ChatMessage validation issues
        mock_chunker = AsyncMock()
        mock_chunker.split_text.return_value = [
            Chunk(content="Contextualized chunk 1", metadata={}),
            Chunk(content="Contextualized chunk 2", metadata={}),
        ]
        mock_contextual_chunker_class.return_value = mock_chunker

        # Create a new parser with mocked contextual chunker
        parser = CogentParser(
            chunk_size=1000,
            chunk_overlap=200,
            use_contextual_chunking=True,
            assemblyai_api_key="test_key",
        )
        parser.chunker = mock_chunker

        text = "This is a test text that should be split into chunks. " * 10

        chunks = await parser.split_text(text)

        assert isinstance(chunks, list)
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert len(chunks) == 2
        assert chunks[0].content == "Contextualized chunk 1"
        assert chunks[1].content == "Contextualized chunk 2"

    @pytest.mark.unit
    async def test_split_text_empty(self, parser):
        """Test text splitting with empty text."""
        chunks = await parser.split_text("")

        assert isinstance(chunks, list)
        assert len(chunks) == 0

    @pytest.mark.unit
    async def test_split_text_short(self, parser):
        """Test text splitting with text shorter than chunk size."""
        text = "Short text"

        chunks = await parser.split_text(text)

        assert isinstance(chunks, list)
        assert len(chunks) == 1
        assert chunks[0].content == text

    @pytest.mark.unit
    @patch("cogent.base.sensory.parser.cogent_parser.VideoParser")
    @patch("cogent.base.sensory.parser.cogent_parser.get_cogent_config")
    @patch("cogent.base.sensory.parser.cogent_parser.tempfile")
    @patch("cogent.base.sensory.parser.cogent_parser.os")
    async def test_parse_video_cleanup_on_exception(
        self, mock_os, mock_tempfile, mock_get_config, mock_video_parser, parser
    ):
        """Test that temporary files are cleaned up even when exceptions occur."""
        # Mock temporary file
        mock_temp_file = Mock()
        mock_temp_file.name = "/tmp/test_video.mp4"
        mock_tempfile.NamedTemporaryFile.return_value = mock_temp_file

        # Mock config
        mock_get_config.return_value = {"parser": {"vision": {"frame_sample_rate": 30}}}

        # Mock video parser to raise exception
        mock_parser_instance = AsyncMock()
        mock_parser_instance.process_video.side_effect = Exception("Test error")
        mock_video_parser.return_value = mock_parser_instance

        # Mock file existence check
        mock_os.path.exists.return_value = True

        with pytest.raises(Exception, match="Test error"):
            await parser._parse_video(b"fake_video_content")

        # Verify cleanup still happened
        mock_os.unlink.assert_called_once_with("/tmp/test_video.mp4")

    @pytest.mark.unit
    @patch("cogent.base.sensory.parser.cogent_parser.VideoParser")
    @patch("cogent.base.sensory.parser.cogent_parser.get_cogent_config")
    @patch("cogent.base.sensory.parser.cogent_parser.tempfile")
    @patch("cogent.base.sensory.parser.cogent_parser.os")
    async def test_parse_video_cleanup_file_not_exists(
        self, mock_os, mock_tempfile, mock_get_config, mock_video_parser, parser
    ):
        """Test cleanup when temporary file doesn't exist."""
        # Mock temporary file
        mock_temp_file = Mock()
        mock_temp_file.name = "/tmp/test_video.mp4"
        mock_tempfile.NamedTemporaryFile.return_value = mock_temp_file

        # Mock config
        mock_get_config.return_value = {"parser": {"vision": {"frame_sample_rate": 30}}}

        # Mock video parser results
        mock_results = Mock()
        mock_results.metadata = {"duration": 10.0}
        mock_results.frame_descriptions.time_to_content = {}
        mock_results.transcript.time_to_content = {}

        mock_parser_instance = AsyncMock()
        mock_parser_instance.process_video.return_value = mock_results
        mock_video_parser.return_value = mock_parser_instance

        # Mock file doesn't exist
        mock_os.path.exists.return_value = False

        await parser._parse_video(b"fake_video_content")

        # Verify cleanup was not called since file doesn't exist
        mock_os.unlink.assert_not_called()

    @pytest.mark.unit
    @patch("cogent.base.sensory.parser.cogent_parser.VideoParser")
    @patch("cogent.base.sensory.parser.cogent_parser.get_cogent_config")
    @patch("cogent.base.sensory.parser.cogent_parser.tempfile")
    @patch("cogent.base.sensory.parser.cogent_parser.os")
    async def test_parse_video_cleanup_os_error(
        self, mock_os, mock_tempfile, mock_get_config, mock_video_parser, parser
    ):
        """Test cleanup when os.unlink raises an error."""
        # Mock temporary file
        mock_temp_file = Mock()
        mock_temp_file.name = "/tmp/test_video.mp4"
        mock_tempfile.NamedTemporaryFile.return_value = mock_temp_file

        # Mock config
        mock_get_config.return_value = {"parser": {"vision": {"frame_sample_rate": 30}}}

        # Mock video parser results
        mock_results = Mock()
        mock_results.metadata = {"duration": 10.0}
        mock_results.frame_descriptions.time_to_content = {}
        mock_results.transcript.time_to_content = {}

        mock_parser_instance = AsyncMock()
        mock_parser_instance.process_video.return_value = mock_results
        mock_video_parser.return_value = mock_parser_instance

        # Mock file exists but unlink fails
        mock_os.path.exists.return_value = True
        mock_os.unlink.side_effect = OSError("Permission denied")

        # Should not raise exception, just log warning
        await parser._parse_video(b"fake_video_content")

        # Verify cleanup was attempted
        mock_os.unlink.assert_called_once_with("/tmp/test_video.mp4")

    @pytest.mark.unit
    @patch("cogent.base.sensory.parser.cogent_parser.partition")
    async def test_parse_object_empty_file(self, mock_partition, parser):
        """Test object parsing with empty file."""
        mock_partition.return_value = []

        metadata, elements = await parser._parse_object(b"", "empty.txt")

        assert len(elements) == 0
        assert metadata == {}

    @pytest.mark.unit
    @patch("cogent.base.sensory.parser.cogent_parser.partition")
    async def test_parse_object_metadata_none(self, mock_partition, parser):
        """Test object parsing when metadata is None."""
        mock_part = Mock()
        mock_part.category = "Text"
        mock_part.text = "Sample content"
        mock_part.metadata = None

        mock_partition.return_value = [mock_part]

        metadata, elements = await parser._parse_object(b"fake_content", "test.txt")

        assert len(elements) == 1
        assert elements[0].category == "Text"
        assert elements[0].text == "Sample content"
        # Should have default values for metadata fields
        assert elements[0].langs == []
        assert elements[0].page_number == 0

    @pytest.mark.unit
    @patch("cogent.base.sensory.parser.cogent_parser.partition")
    async def test_parse_object_missing_text_as_html(self, mock_partition, parser):
        """Test object parsing when text_as_html is not available."""
        mock_part = Mock()
        mock_part.category = "Text"
        mock_part.text = "Sample content"
        mock_part.metadata = Mock()
        mock_part.metadata.languages = ["en"]
        mock_part.metadata.page_number = 1
        mock_part.metadata.coordinates = None
        # Remove text_as_html from dir() result
        mock_part.metadata.__dir__ = lambda self: ["languages", "page_number", "coordinates"]

        mock_partition.return_value = [mock_part]

        metadata, elements = await parser._parse_object(b"fake_content", "test.txt")

        assert len(elements) == 1
        assert elements[0].text_html == ""  # Should have default value

    @pytest.mark.unit
    @patch("cogent.base.sensory.parser.cogent_parser.filetype")
    async def test_parse_file_to_text_unknown_file_type(self, mock_filetype, parser):
        """Test parse_file_to_text for unknown file types."""
        # Mock filetype.guess to return None (unknown type)
        mock_filetype.guess.return_value = None

        # Mock object parsing
        with patch.object(parser, "_parse_object") as mock_parse_object:
            mock_parse_object.return_value = ({}, [ParsedElement(text="unknown content")])

            metadata, elements = await parser.parse_file_to_text(b"unknown_content", "test.unknown")

            mock_parse_object.assert_called_once_with(b"unknown_content", "test.unknown")
            assert len(elements) == 1
            assert elements[0].text == "unknown content"

    @pytest.mark.unit
    async def test_split_text_very_long_text(self, parser):
        """Test text splitting with very long text."""
        # Create a very long text that will definitely be split
        text = (
            "This is a very long sentence that will be repeated many times to create a text that exceeds the "
            "chunk size limit. " * 100
        )

        chunks = await parser.split_text(text)

        assert isinstance(chunks, list)
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert len(chunks) > 1  # Should be split into multiple chunks

    @pytest.mark.unit
    async def test_split_text_with_special_characters(self, parser):
        """Test text splitting with special characters and unicode."""
        text = "Special chars: ñáéíóú 🚀 🌟 💻 \n\n New paragraph with emojis! 🎉"

        chunks = await parser.split_text(text)

        assert isinstance(chunks, list)
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert len(chunks) > 0
        # Verify that special characters are preserved
        combined_text = " ".join(chunk.content for chunk in chunks)
        assert "ñáéíóú" in combined_text
        assert "🚀 🌟 💻" in combined_text
