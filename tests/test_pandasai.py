"""
Copyright (c) 2025 Mirasurf
Tests for PandasAI integration module.
"""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from cogent_base.smart_frame import (
    CogentPandasAI,
    DashScopePandasAI,
    PandasAIOnDashScope,
    create_pandasai_agent,
    setup_pandasai_llm,
)


class TestPandasAIOnDashScope:
    """Test the PandasAIOnDashScope class."""

    @pytest.mark.unit
    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        with patch.dict("os.environ", {"DASHSCOPE_API_KEY": "test-key"}):
            llm = PandasAIOnDashScope(api_token="test-key")
            assert llm.model == "qwen-plus"
            assert llm._is_chat_model is True

    @pytest.mark.unit
    def test_init_with_custom_model(self):
        """Test initialization with custom model."""
        with patch.dict("os.environ", {"DASHSCOPE_API_KEY": "test-key"}):
            llm = PandasAIOnDashScope(api_token="test-key", model="qwen-turbo")
            assert llm.model == "qwen-turbo"


class TestDashScopePandasAI:
    """Test the DashScopePandasAI class."""

    @pytest.mark.unit
    def test_init(self):
        """Test initialization."""
        model = DashScopePandasAI("test_model")
        assert model.model_key == "test_model"
        assert model.llm_instance is None

    @pytest.mark.unit
    def test_is_available_false(self):
        """Test is_available when LLM is not set up."""
        model = DashScopePandasAI("test_model")
        assert model.is_available() is False

    @pytest.mark.unit
    def test_is_available_true(self):
        """Test is_available when LLM is set up."""
        model = DashScopePandasAI("test_model")
        model.llm_instance = Mock()
        assert model.is_available() is True

    @patch("cogent_base.smart_frame.dashscope_pandasai.get_cogent_config")
    @patch.dict("os.environ", {"DASHSCOPE_API_KEY": "test-key"})
    @pytest.mark.unit
    def test_setup_llm_success(self, mock_get_config):
        """Test successful LLM setup."""
        # Mock config
        mock_config = Mock()
        mock_config.llm.registered_models = {
            "test_model": {"model_name": "qwen-plus", "api_base": "https://test.api.com"}
        }
        mock_get_config.return_value = mock_config

        model = DashScopePandasAI("test_model")
        result = model.setup_llm()

        assert result is True
        assert model.llm_instance is not None

    @patch("cogent_base.smart_frame.dashscope_pandasai.get_cogent_config")
    @pytest.mark.unit
    def test_setup_llm_no_api_key(self, mock_get_config):
        """Test LLM setup without API key."""
        mock_config = Mock()
        mock_get_config.return_value = mock_config

        model = DashScopePandasAI("test_model")
        result = model.setup_llm()

        assert result is False
        assert model.llm_instance is None


class TestCogentPandasAI:
    """Test the CogentPandasAI class."""

    @patch("cogent_base.smart_frame.cogent_pandasai.get_cogent_config")
    @pytest.mark.unit
    def test_init_success(self, mock_get_config):
        """Test successful initialization."""
        # Mock config
        mock_config = Mock()
        mock_config.llm.registered_models = {
            "test_model": {"model_name": "qwen-plus", "api_base": "https://test.api.com"}
        }
        mock_get_config.return_value = mock_config

        # Mock factory
        with patch("cogent_base.smart_frame.cogent_pandasai.PandasAIFactory.create_model") as mock_factory:
            mock_impl = Mock()
            mock_factory.return_value = mock_impl

            pandasai = CogentPandasAI("test_model", "dashscope")

            assert pandasai.model_key == "test_model"
            assert pandasai.provider == "dashscope"
            assert pandasai.pandasai_impl == mock_impl

    @patch("cogent_base.smart_frame.cogent_pandasai.get_cogent_config")
    @pytest.mark.unit
    def test_init_model_not_found(self, mock_get_config):
        """Test initialization with non-existent model."""
        mock_config = Mock()
        mock_config.llm.registered_models = {}
        mock_get_config.return_value = mock_config

        with pytest.raises(ValueError, match="Model 'test_model' not found"):
            CogentPandasAI("test_model", "dashscope")

    @pytest.mark.unit
    def test_analyze_data(self):
        """Test data analysis functionality."""
        # Create test data
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

        # Mock implementation
        mock_impl = Mock()
        mock_agent = Mock()
        mock_agent.chat.return_value = "Analysis result"
        mock_impl.create_agent.return_value = mock_agent

        pandasai = CogentPandasAI.__new__(CogentPandasAI)
        pandasai.pandasai_impl = mock_impl

        result = pandasai.analyze_data(df, "What is the sum of column A?")

        assert result == "Analysis result"
        mock_impl.create_agent.assert_called_once()
        mock_agent.chat.assert_called_once_with("What is the sum of column A?")

    @pytest.mark.integration
    def test_generate_chart(self):
        """Test chart generation functionality."""
        # Create test data
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

        # Mock implementation
        mock_impl = Mock()
        mock_agent = Mock()
        mock_agent.chat.return_value = "Chart result"
        mock_impl.create_agent.return_value = mock_agent

        pandasai = CogentPandasAI.__new__(CogentPandasAI)
        pandasai.pandasai_impl = mock_impl

        result = pandasai.generate_chart(df, "Create a bar chart of column A")

        assert result == "Chart result"
        mock_impl.create_agent.assert_called_once()
        mock_agent.chat.assert_called_once_with("Create a chart: Create a bar chart of column A")


class TestIntegrationPandasAI:
    """Integration tests for PandasAI functionality."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
                "age": [25, 30, 35, 28, 32],
                "salary": [50000, 60000, 70000, 55000, 65000],
                "department": ["Engineering", "Marketing", "Engineering", "HR", "Marketing"],
            }
        )

    @pytest.fixture
    def mock_dashscope_config(self):
        """Mock DashScope configuration."""
        return {
            "test_model": {"model_name": "qwen-plus", "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1"}
        }

    @pytest.mark.integration
    def test_dashscope_pandasai_full_workflow(self, sample_dataframe, mock_dashscope_config):
        """Test complete workflow with DashScopePandasAI."""
        with patch("cogent_base.smart_frame.dashscope_pandasai.get_cogent_config") as mock_get_config:
            mock_config = Mock()
            mock_config.llm.registered_models = mock_dashscope_config
            mock_get_config.return_value = mock_config

            with patch.dict("os.environ", {"DASHSCOPE_API_KEY": "test-key"}):
                # Test initialization
                model = DashScopePandasAI("test_model")
                assert model.model_key == "test_model"
                assert not model.is_available()

                # Test LLM setup
                result = model.setup_llm()
                assert result is True
                assert model.is_available()
                assert model.llm_instance is not None

                # Test agent creation
                agent = model.create_agent(sample_dataframe)
                assert agent is not None

    @pytest.mark.integration
    def test_cogent_pandasai_full_workflow(self, sample_dataframe, mock_dashscope_config):
        """Test complete workflow with CogentPandasAI."""
        with patch("cogent_base.smart_frame.cogent_pandasai.get_cogent_config") as mock_get_config:
            mock_config = Mock()
            mock_config.llm.registered_models = mock_dashscope_config
            mock_get_config.return_value = mock_config

            with patch("cogent_base.smart_frame.cogent_pandasai.PandasAIFactory.create_model") as mock_factory:
                # Create a real DashScopePandasAI instance
                mock_impl = DashScopePandasAI("test_model")
                mock_impl.llm_instance = Mock()  # Mock the LLM instance
                mock_factory.return_value = mock_impl

                # Test initialization
                pandasai = CogentPandasAI("test_model", "dashscope")
                assert pandasai.model_key == "test_model"
                assert pandasai.provider == "dashscope"
                assert pandasai.pandasai_impl is not None

                # Test setup
                setup_result = pandasai.setup_llm()
                assert setup_result is True

                # Test availability
                assert pandasai.is_available() is True

    @pytest.mark.integration
    def test_data_analysis_integration(self, sample_dataframe):
        """Test data analysis with real DataFrame."""
        with patch("cogent_base.smart_frame.cogent_pandasai.get_cogent_config") as mock_get_config:
            mock_config = Mock()
            mock_config.llm.registered_models = {
                "test_model": {"model_name": "qwen-plus", "api_base": "https://test.api.com"}
            }
            mock_get_config.return_value = mock_config

            with patch("cogent_base.smart_frame.cogent_pandasai.PandasAIFactory.create_model") as mock_factory:
                # Create mock implementation that returns a real agent
                mock_impl = Mock()
                mock_agent = Mock()
                mock_agent.chat.return_value = "The average age is 30 years"
                mock_impl.create_agent.return_value = mock_agent
                mock_impl.setup_llm.return_value = True
                mock_impl.is_available.return_value = True
                mock_factory.return_value = mock_impl

                pandasai = CogentPandasAI("test_model", "dashscope")
                pandasai.setup_llm()

                # Test data analysis
                result = pandasai.analyze_data(sample_dataframe, "What is the average age?")
                assert result == "The average age is 30 years"
                assert mock_agent.chat.called

    @pytest.mark.integration
    def test_chart_generation_integration(self, sample_dataframe):
        """Test chart generation with real DataFrame."""
        with patch("cogent_base.smart_frame.cogent_pandasai.get_cogent_config") as mock_get_config:
            mock_config = Mock()
            mock_config.llm.registered_models = {
                "test_model": {"model_name": "qwen-plus", "api_base": "https://test.api.com"}
            }
            mock_get_config.return_value = mock_config

            with patch("cogent_base.smart_frame.cogent_pandasai.PandasAIFactory.create_model") as mock_factory:
                # Create mock implementation that returns a real agent
                mock_impl = Mock()
                mock_agent = Mock()
                mock_agent.chat.return_value = "Chart created successfully"
                mock_impl.create_agent.return_value = mock_agent
                mock_impl.setup_llm.return_value = True
                mock_impl.is_available.return_value = True
                mock_factory.return_value = mock_impl

                pandasai = CogentPandasAI("test_model", "dashscope")
                pandasai.setup_llm()

                # Test chart generation
                result = pandasai.generate_chart(sample_dataframe, "Create a bar chart of salaries by department")
                assert result == "Chart created successfully"
                assert mock_agent.chat.called

    @pytest.mark.integration
    def test_setup_functions_integration(self, sample_dataframe, mock_dashscope_config):
        """Test setup functions with real configuration."""
        with patch("cogent_base.smart_frame.dashscope_pandasai.get_cogent_config") as mock_get_config:
            mock_config = Mock()
            mock_config.llm.registered_models = mock_dashscope_config
            mock_get_config.return_value = mock_config

            with patch.dict("os.environ", {"DASHSCOPE_API_KEY": "test-key"}):
                # Test LLM setup
                llm = setup_pandasai_llm("test_model")
                assert llm is not None
                assert isinstance(llm, PandasAIOnDashScope)
                assert llm.model == "qwen-plus"

                # Test agent creation
                agent = create_pandasai_agent(sample_dataframe, llm)
                assert agent is not None

    @pytest.mark.integration
    def test_error_handling_integration(self):
        """Test error handling in integration scenarios."""
        # Test with invalid model key
        with patch("cogent_base.smart_frame.cogent_pandasai.get_cogent_config") as mock_get_config:
            mock_config = Mock()
            mock_config.llm.registered_models = {}
            mock_get_config.return_value = mock_config

            with pytest.raises(ValueError, match="Model 'invalid_model' not found"):
                CogentPandasAI("invalid_model", "dashscope")

        # Test with missing API key
        with patch("cogent_base.smart_frame.dashscope_pandasai.get_cogent_config") as mock_get_config:
            mock_config = Mock()
            mock_config.llm.registered_models = {
                "test_model": {"model_name": "qwen-plus", "api_base": "https://test.api.com"}
            }
            mock_get_config.return_value = mock_config

            # Remove DASHSCOPE_API_KEY from environment to test missing API key
            with patch.dict("os.environ", {}, clear=True):
                result = setup_pandasai_llm("test_model")
                assert result is None

    @pytest.mark.integration
    def test_dataframe_operations_integration(self):
        """Test various DataFrame operations with PandasAI."""
        # Create a more complex DataFrame
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=10, freq="D"),
                "sales": [100, 150, 200, 175, 225, 300, 250, 275, 350, 400],
                "region": ["North", "South", "North", "South", "North", "South", "North", "South", "North", "South"],
                "product": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
            }
        )

        with patch("cogent_base.smart_frame.cogent_pandasai.get_cogent_config") as mock_get_config:
            mock_config = Mock()
            mock_config.llm.registered_models = {
                "test_model": {"model_name": "qwen-plus", "api_base": "https://test.api.com"}
            }
            mock_get_config.return_value = mock_config

            with patch("cogent_base.smart_frame.cogent_pandasai.PandasAIFactory.create_model") as mock_factory:
                mock_impl = Mock()
                mock_agent = Mock()
                mock_agent.chat.return_value = "Analysis completed"
                mock_impl.create_agent.return_value = mock_agent
                mock_impl.setup_llm.return_value = True
                mock_impl.is_available.return_value = True
                mock_factory.return_value = mock_impl

                pandasai = CogentPandasAI("test_model", "dashscope")
                pandasai.setup_llm()

                # Test various analysis queries
                queries = [
                    "What is the total sales?",
                    "Show me sales by region",
                    "What is the trend in sales over time?",
                    "Which product has higher sales?",
                ]

                for query in queries:
                    result = pandasai.analyze_data(df, query)
                    assert result == "Analysis completed"
                    assert mock_agent.chat.called


class TestSetupFunctions:
    """Test the setup functions."""

    @patch("cogent_base.smart_frame.dashscope_pandasai.get_cogent_config")
    @patch.dict("os.environ", {"DASHSCOPE_API_KEY": "test-key"})
    @pytest.mark.unit
    def test_setup_pandasai_llm_success(self, mock_get_config):
        """Test successful PandasAI LLM setup."""
        mock_config = Mock()
        mock_config.llm.registered_models = {
            "test_model": {"model_name": "qwen-plus", "api_base": "https://test.api.com"}
        }
        mock_get_config.return_value = mock_config

        result = setup_pandasai_llm("test_model")

        assert result is not None
        assert isinstance(result, PandasAIOnDashScope)

    @patch("cogent_base.smart_frame.dashscope_pandasai.get_cogent_config")
    @pytest.mark.unit
    def test_setup_pandasai_llm_no_api_key(self, mock_get_config):
        """Test PandasAI LLM setup without API key."""
        mock_config = Mock()
        mock_get_config.return_value = mock_config

        result = setup_pandasai_llm("test_model")

        assert result is None

    @patch("cogent_base.smart_frame.dashscope_pandasai.get_cogent_config")
    @patch.dict("os.environ", {"DASHSCOPE_API_KEY": "test-key"})
    @pytest.mark.unit
    def test_create_pandasai_agent_success(self, mock_get_config):
        """Test successful PandasAI agent creation."""
        # Create test data
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

        # Mock LLM
        mock_llm = Mock()

        # Mock pandasai config and Agent
        with patch("cogent_base.smart_frame.dashscope_pandasai.pai.config.set") as mock_config_set:
            with patch("cogent_base.smart_frame.dashscope_pandasai.Agent") as mock_agent_class:
                mock_agent = Mock()
                mock_agent_class.return_value = mock_agent

                result = create_pandasai_agent(df, mock_llm)

                assert result == mock_agent
                mock_config_set.assert_called_once()
                mock_agent_class.assert_called_once()

    @pytest.mark.unit
    def test_create_pandasai_agent_failure(self):
        """Test PandasAI agent creation failure."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        mock_llm = Mock()

        # Mock import failure
        with patch("cogent_base.smart_frame.dashscope_pandasai.pai", new=None):
            result = create_pandasai_agent(df, mock_llm)

            assert result is None
