"""
Unit tests for cogent.providers.utils module.
"""

from unittest.mock import MagicMock, patch

import pytest

from cogentbase.providers.utils import initialize_ollama_model, is_ollama_model


class TestIsOllamaModel:
    """Test the is_ollama_model function."""

    @pytest.mark.unit
    def test_ollama_in_model_name(self):
        """Test detection when 'ollama' is in model name."""
        assert is_ollama_model("ollama/llama3.2:latest", "test_model") is True
        assert is_ollama_model("OLLAMA/llama3.2:latest", "test_model") is True
        assert is_ollama_model("my-ollama-model", "test_model") is True

    @pytest.mark.unit
    def test_ollama_in_model_key(self):
        """Test detection when 'ollama' is in model key."""
        assert is_ollama_model("llama3.2:latest", "test_ollama") is True
        assert is_ollama_model("llama3.2:latest", "OLLAMA_MODEL") is True
        assert is_ollama_model("llama3.2:latest", "my-ollama-key") is True

    @pytest.mark.unit
    def test_ollama_in_both(self):
        """Test detection when 'ollama' is in both model name and key."""
        assert is_ollama_model("ollama/llama3.2:latest", "test_ollama") is True

    @pytest.mark.unit
    def test_not_ollama_model(self):
        """Test detection when model is not an Ollama model."""
        assert is_ollama_model("gpt-4o-mini", "test_model") is False
        assert is_ollama_model("claude-3-sonnet", "openai_model") is False
        assert is_ollama_model("", "") is False

    @pytest.mark.unit
    def test_case_insensitive(self):
        """Test that detection is case insensitive."""
        assert is_ollama_model("OLLAMA/llama3.2:latest", "test_model") is True
        assert is_ollama_model("ollama/llama3.2:latest", "TEST_OLLAMA") is True
        assert is_ollama_model("OLLAMA/llama3.2:latest", "TEST_OLLAMA") is True


class TestInitializeOllamaModel:
    """Test the initialize_ollama_model function."""

    @pytest.mark.unit
    def test_non_ollama_model(self):
        """Test initialization with non-Ollama model."""
        model_key = "test_model"
        model_config = {"model_name": "gpt-4o-mini"}

        result = initialize_ollama_model(model_key, model_config)

        assert result == (False, None, None)

    @pytest.mark.unit
    def test_ollama_model_with_library_available(self):
        """Test initialization with Ollama model when library is available."""
        model_key = "test_ollama"
        model_config = {"model_name": "ollama/llama3.2:latest", "api_base": "http://localhost:11434"}

        with patch("builtins.__import__") as mock_import:
            # Mock successful import
            mock_ollama = MagicMock()
            mock_ollama.__version__ = "0.1.0"
            mock_import.return_value = mock_ollama

            result = initialize_ollama_model(model_key, model_config)

            assert result == (True, "http://localhost:11434", "ollama/llama3.2:latest")

    @pytest.mark.unit
    def test_ollama_model_library_missing(self):
        """Test initialization with Ollama model when library is missing."""
        model_key = "test_ollama"
        model_config = {"model_name": "ollama/llama3.2:latest", "api_base": "http://localhost:11434"}

        with patch("builtins.__import__", side_effect=ImportError("No module named 'ollama'")):
            with patch("cogentbase.providers.utils.logger") as mock_logger:
                result = initialize_ollama_model(model_key, model_config)

                assert result == (False, None, None)
                mock_logger.warning.assert_called_once_with(
                    "Ollama model selected, but 'ollama' library not installed. Falling back to LiteLLM."
                )

    @pytest.mark.unit
    def test_ollama_model_missing_api_base(self):
        """Test initialization with Ollama model when api_base is missing."""
        model_key = "test_ollama"
        model_config = {"model_name": "ollama/llama3.2:latest"}

        with patch("builtins.__import__") as mock_import:
            # Mock successful import
            mock_ollama = MagicMock()
            mock_ollama.__version__ = "0.1.0"
            mock_import.return_value = mock_ollama

            with patch("cogentbase.providers.utils.logger") as mock_logger:
                result = initialize_ollama_model(model_key, model_config)

                assert result == (False, None, None)
                mock_logger.warning.assert_called_once_with(
                    f"Ollama model {model_key} selected for direct use, "
                    "but 'api_base' is missing in config. Falling back to LiteLLM."
                )

    @pytest.mark.unit
    def test_ollama_model_missing_model_name(self):
        """Test initialization with Ollama model when model_name is missing."""
        model_key = "test_ollama"
        model_config = {"api_base": "http://localhost:11434"}

        with patch("builtins.__import__") as mock_import:
            # Mock successful import
            mock_ollama = MagicMock()
            mock_ollama.__version__ = "0.1.0"
            mock_import.return_value = mock_ollama

            with patch("cogentbase.providers.utils.logger") as mock_logger:
                result = initialize_ollama_model(model_key, model_config)

                assert result == (False, None, None)
                mock_logger.warning.assert_called_once_with(
                    f"Could not parse base model name from Ollama model "
                    f"{model_config.get('model_name', '')}. Falling back to LiteLLM."
                )

    @pytest.mark.unit
    def test_ollama_model_empty_model_name(self):
        """Test initialization with Ollama model when model_name is empty."""
        model_key = "test_ollama"
        model_config = {"model_name": "", "api_base": "http://localhost:11434"}

        with patch("builtins.__import__") as mock_import:
            # Mock successful import
            mock_ollama = MagicMock()
            mock_ollama.__version__ = "0.1.0"
            mock_import.return_value = mock_ollama

            with patch("cogentbase.providers.utils.logger") as mock_logger:
                result = initialize_ollama_model(model_key, model_config)

                assert result == (False, None, None)
                mock_logger.warning.assert_called_once_with(
                    f"Could not parse base model name from Ollama model "
                    f"{model_config['model_name']}. Falling back to LiteLLM."
                )

    @pytest.mark.unit
    def test_ollama_model_none_model_name(self):
        """Test initialization with Ollama model when model_name is None."""
        model_key = "test_ollama"
        model_config = {"model_name": None, "api_base": "http://localhost:11434"}

        with patch("builtins.__import__") as mock_import:
            # Mock successful import
            mock_ollama = MagicMock()
            mock_ollama.__version__ = "0.1.0"
            mock_import.return_value = mock_ollama

            with patch("cogentbase.providers.utils.logger") as mock_logger:
                result = initialize_ollama_model(model_key, model_config)

                assert result == (False, None, None)
                mock_logger.warning.assert_called_once_with(
                    f"Could not parse base model name from Ollama model "
                    f"{model_config['model_name']}. Falling back to LiteLLM."
                )

    @pytest.mark.unit
    def test_ollama_model_complete_config(self):
        """Test initialization with complete Ollama model configuration."""
        model_key = "test_ollama"
        model_config = {
            "model_name": "ollama/llama3.2:latest",
            "api_base": "http://localhost:11434",
            "temperature": 0.7,
            "max_tokens": 1000,
        }

        with patch("builtins.__import__") as mock_import:
            # Mock successful import
            mock_ollama = MagicMock()
            mock_ollama.__version__ = "0.1.0"
            mock_import.return_value = mock_ollama

            result = initialize_ollama_model(model_key, model_config)

            assert result == (True, "http://localhost:11434", "ollama/llama3.2:latest")

    @pytest.mark.unit
    def test_ollama_model_case_insensitive_detection(self):
        """Test that Ollama detection is case insensitive."""
        model_key = "TEST_OLLAMA"
        model_config = {"model_name": "OLLAMA/llama3.2:latest", "api_base": "http://localhost:11434"}

        with patch("builtins.__import__") as mock_import:
            # Mock successful import
            mock_ollama = MagicMock()
            mock_ollama.__version__ = "0.1.0"
            mock_import.return_value = mock_ollama

            result = initialize_ollama_model(model_key, model_config)

            assert result == (True, "http://localhost:11434", "OLLAMA/llama3.2:latest")

    @pytest.mark.unit
    def test_ollama_model_import_error_simulation(self):
        """Test initialization when ollama import raises ImportError."""
        model_key = "test_ollama"
        model_config = {"model_name": "ollama/llama3.2:latest", "api_base": "http://localhost:11434"}

        with patch("builtins.__import__", side_effect=ImportError("No module named 'ollama'")):
            with patch("cogentbase.providers.utils.logger") as mock_logger:
                result = initialize_ollama_model(model_key, model_config)

                assert result == (False, None, None)
                mock_logger.warning.assert_called_once_with(
                    "Ollama model selected, but 'ollama' library not installed. Falling back to LiteLLM."
                )
