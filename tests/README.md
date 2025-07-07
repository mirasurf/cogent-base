# Cogent AI Agent System - Testing Guide

This directory contains comprehensive tests for the Cogent AI Agent System, including both unit tests and integration tests.

## Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Pytest configuration and common fixtures
├── test_cogent_config.py    # Configuration tests
├── providers/
│   └── test_completion.py   # LiteLLM completion model tests
└── README.md               # This file
```

## Test Categories

### Unit Tests
- **Purpose**: Test individual components in isolation
- **Dependencies**: None (all external dependencies are mocked)
- **Speed**: Fast
- **Markers**: `@pytest.mark.unit`

### Integration Tests
- **Purpose**: Test components with real external services
- **Dependencies**: Ollama service and/or OpenAI API key
- **Speed**: Slower (depends on external service response times)
- **Markers**: `@pytest.mark.integration`

## Running Tests

### Prerequisites

1. Install development dependencies:
   ```bash
   make install-dev
   ```

2. For integration tests, ensure external services are available:
   - **Ollama**: Install and start Ollama service
   - **OpenAI**: Set `OPENAI_API_KEY` environment variable

### Test Commands

#### Using Makefile (Recommended)
```bash
# Run all tests
make test

# Run unit tests only
make test-unit

# Run integration tests only
make test-integration

# Run integration tests with service checks
make test-integration-full

# Run tests with coverage
make test-coverage

# Run tests in watch mode
make test-watch
```

#### Using pytest directly
```bash
# Run all tests
pytest tests/ -v

# Run unit tests only
pytest tests/ -v -m "not integration"

# Run integration tests only
pytest tests/ -v -m "integration"

# Run specific test file
pytest tests/providers/test_completion.py -v

# Run specific test
pytest tests/providers/test_completion.py::TestIntegrationLiteLLMCompletion::test_ollama_basic_completion -v
```

## Integration Test Setup

### Ollama Setup

1. **Install Ollama**:
   ```bash
   # macOS
   brew install ollama
   
   # Linux
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

2. **Start Ollama service**:
   ```bash
   ollama serve
   ```

3. **Pull required models**:
   ```bash
   ollama pull llama3.2:latest
   ollama pull qwen2.5vl:latest
   ```

4. **Verify service is running**:
   ```bash
   curl http://localhost:11434/api/tags
   ```

### OpenAI Setup

1. **Get API key**: Sign up at [OpenAI](https://platform.openai.com/)

2. **Set environment variable**:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

3. **Verify API key**:
   ```bash
   echo $OPENAI_API_KEY
   ```

## Test Coverage

The integration tests cover:

### LiteLLMCompletionModel Tests
- ✅ Basic text completion (Ollama & OpenAI)
- ✅ Structured output completion (Ollama & OpenAI)
- ✅ Vision/multimodal completion (Ollama)
- ✅ Streaming completion (Ollama & OpenAI)
- ✅ Chat history integration
- ✅ Custom prompt templates
- ✅ Dynamic JSON schema parsing
- ✅ Error handling and fallbacks

### Test Scenarios
- **Service Availability**: Tests skip gracefully when services are unavailable
- **Model Detection**: Proper detection of Ollama vs other models
- **Fallback Behavior**: Graceful fallback when Ollama library is missing
- **Network Errors**: Proper error handling for network failures
- **Invalid Configurations**: Error handling for invalid model configurations

## Test Configuration

### Pytest Configuration (`pytest.ini`)
- Automatic test discovery
- Custom markers for test categorization
- Async test support
- Verbose output by default

### Common Fixtures (`conftest.py`)
- `test_config`: Test model configurations
- `mock_config`: Mocked global configuration
- `check_ollama_service`: Ollama service availability check
- `check_openai_key`: OpenAI API key availability check

## Troubleshooting

### Common Issues

1. **Ollama service not accessible**:
   ```bash
   # Check if Ollama is running
   curl http://localhost:11434/api/tags
   
   # Start Ollama if not running
   ollama serve
   ```

2. **Missing models**:
   ```bash
   # List available models
   ollama list
   
   # Pull missing models
   ollama pull llama3.2:latest
   ```

3. **OpenAI API key issues**:
   ```bash
   # Check if key is set
   echo $OPENAI_API_KEY
   
   # Set key if missing
   export OPENAI_API_KEY="your-key-here"
   ```

4. **Test dependencies missing**:
   ```bash
   # Install development dependencies
   make install-dev
   ```

### Debugging Tests

1. **Run with verbose output**:
   ```bash
   pytest tests/ -v -s
   ```

2. **Run specific failing test**:
   ```bash
   pytest tests/providers/test_completion.py::TestIntegrationLiteLLMCompletion::test_ollama_basic_completion -v -s
   ```

3. **Run with coverage**:
   ```bash
   make test-coverage
   ```

## Contributing

When adding new tests:

1. **Use appropriate markers**:
   - `@pytest.mark.unit` for unit tests
   - `@pytest.mark.integration` for integration tests
   - `@pytest.mark.slow` for slow-running tests

2. **Follow naming conventions**:
   - Test classes: `Test<ComponentName>`
   - Test methods: `test_<scenario>_<expected_behavior>`

3. **Add proper error handling**:
   - Skip tests when dependencies are unavailable
   - Provide helpful error messages
   - Use appropriate assertions

4. **Update this README** when adding new test categories or requirements 