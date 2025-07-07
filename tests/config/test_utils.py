"""
Tests for configuration utilities.
"""

import tempfile
import unittest
from pathlib import Path

import pytest

from cogentbase.config import (
    load_merged_toml_configs,
    load_toml_config,
)


class TestLoadTomlConfig(unittest.TestCase):
    """Test the load_toml_config function."""

    @pytest.mark.unit
    def test_load_valid_toml(self):
        """Test loading a valid TOML file."""
        toml_content = """
        [registered_models]
        openai_gpt4 = { model_name = "gpt-4" }

        [completion]
        model = "openai_gpt4"
        default_max_tokens = 1000

        [embedding]
        model = "text-embedding-3-small"
        dimensions = 1536
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            temp_path = Path(f.name)

        try:
            result = load_toml_config(temp_path)
            self.assertIn("registered_models", result)
            self.assertIn("completion", result)
            self.assertIn("embedding", result)
            self.assertEqual(result["registered_models"]["openai_gpt4"]["model_name"], "gpt-4")
        finally:
            temp_path.unlink()

    @pytest.mark.unit
    def test_load_nonexistent_file(self):
        """Test loading a non-existent TOML file."""
        result = load_toml_config(Path("/nonexistent/file.toml"))
        self.assertEqual(result, {})

    @pytest.mark.unit
    def test_load_invalid_toml(self):
        """Test loading an invalid TOML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("invalid toml content [")
            temp_path = Path(f.name)

        try:
            result = load_toml_config(temp_path)
            self.assertEqual(result, {})
        finally:
            temp_path.unlink()


class TestLoadMergedTomlConfigs(unittest.TestCase):
    """Test the load_merged_toml_configs function."""

    @pytest.mark.unit
    def test_merge_multiple_toml_files(self):
        """Test merging multiple TOML files."""
        base_toml = """
        [sensory.parser]
        chunk_size = 6000
        chunk_overlap = 300

        [graph]
        model = "ollama_qwen_vision"
        enable_entity_resolution = true
        """

        providers_toml = """
        [registered_models]
        openai_gpt4 = { model_name = "gpt-4" }

        [completion]
        model = "openai_gpt4"
        default_max_tokens = 1000

        [embedding]
        model = "text-embedding-3-small"
        dimensions = 1536
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f1:
            f1.write(base_toml)
            base_path = Path(f1.name)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f2:
            f2.write(providers_toml)
            providers_path = Path(f2.name)

        try:
            result = load_merged_toml_configs([base_path, providers_path])

            # Check that both files are merged
            self.assertIn("sensory", result)
            self.assertIn("graph", result)
            self.assertIn("registered_models", result)
            self.assertIn("completion", result)
            self.assertIn("embedding", result)

            # Check specific values
            self.assertEqual(result["sensory"]["parser"]["chunk_size"], 6000)
            self.assertEqual(result["graph"]["model"], "ollama_qwen_vision")
            self.assertEqual(result["registered_models"]["openai_gpt4"]["model_name"], "gpt-4")
            self.assertEqual(result["completion"]["model"], "openai_gpt4")
            self.assertEqual(result["embedding"]["dimensions"], 1536)
        finally:
            base_path.unlink()
            providers_path.unlink()

    @pytest.mark.unit
    def test_merge_with_overlapping_keys(self):
        """Test merging TOML files with overlapping keys."""
        file1_content = """
        [section]
        key1 = "value1"
        key2 = "value2"
        """

        file2_content = """
        [section]
        key2 = "overwritten"
        key3 = "value3"
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f1:
            f1.write(file1_content)
            path1 = Path(f1.name)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f2:
            f2.write(file2_content)
            path2 = Path(f2.name)

        try:
            result = load_merged_toml_configs([path1, path2])

            # Later file should overwrite earlier file's values
            self.assertEqual(result["section"]["key1"], "value1")
            self.assertEqual(result["section"]["key2"], "overwritten")
            self.assertEqual(result["section"]["key3"], "value3")
        finally:
            path1.unlink()
            path2.unlink()

    @pytest.mark.unit
    def test_merge_with_nonexistent_files(self):
        """Test merging with some non-existent files."""
        valid_toml = """
        [test]
        key = "value"
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(valid_toml)
            valid_path = Path(f.name)

        try:
            nonexistent_path = Path("/nonexistent/file.toml")
            result = load_merged_toml_configs([valid_path, nonexistent_path])

            # Should still load the valid file
            self.assertIn("test", result)
            self.assertEqual(result["test"]["key"], "value")
        finally:
            valid_path.unlink()


class TestSafeConversion(unittest.TestCase):
    """Test safe conversion functions."""

    @pytest.mark.unit
    def test_safe_int(self):
        """Test _safe_int function."""
        from cogentbase.config import _safe_int

        self.assertEqual(_safe_int(42, 0), 42)
        self.assertEqual(_safe_int("42", 0), 42)
        self.assertEqual(_safe_int("invalid", 10), 10)
        self.assertEqual(_safe_int(None, 5), 5)

    @pytest.mark.unit
    def test_safe_bool(self):
        """Test _safe_bool function."""
        from cogentbase.config import _safe_bool

        self.assertTrue(_safe_bool(True, False))
        self.assertFalse(_safe_bool(False, True))
        self.assertTrue(_safe_bool("true", False))
        self.assertTrue(_safe_bool("1", False))
        self.assertTrue(_safe_bool("yes", False))
        self.assertTrue(_safe_bool("on", False))
        self.assertFalse(_safe_bool("false", True))
        self.assertFalse(_safe_bool("invalid", True))
        self.assertFalse(_safe_bool(None, False))
