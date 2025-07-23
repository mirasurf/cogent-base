# Development Guide

This guide covers development setup, testing, code quality, and deployment procedures for cogent-base.

## Development Setup

### Prerequisites
- Python 3.11+
- Poetry (for dependency management)

### Initial Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mirasurf/cogent-base.git
   cd cogent-base
   ```

2. **Install development dependencies**:
   ```bash
   make install-dev
   ```

3. **Verify setup**:
   ```bash
   make check-env
   ```

## Development Workflow

### Running Tests

#### Unit Tests
```bash
# Run unit tests only
make test-unit

# Using pytest directly
pytest tests/ -v -m "not integration"
```

#### Integration Tests
```bash
# Run integration tests only
make test-integration

# Run integration tests with service checks
make test-integration-full

# Using pytest directly
pytest tests/ -v -m "integration"
```

#### All Tests
```bash
# Run all tests
make test

# Run tests with coverage
make test-coverage

# Run tests in watch mode
make test-watch
```

### Code Quality

#### Formatting
```bash
# Format code with black and isort
make format

# Check formatting without making changes
make format-check
```

#### Linting
```bash
# Run all linting checks
make lint

# Quality checks (format + lint)
make quality
```

#### Development Checks
```bash
# Quick development check (quality + unit tests)
make dev-check

# Full development check (all checks + all tests + build)
make full-check
```

## Configuration System

### Configuration Loading Order

Cogent's configuration system follows a **layered precedence**:

1. **Class Defaults**: Python default values defined in config classes
2. **Package TOML**: Built-in `base.toml` shipped with the package
3. **User Runtime TOML**: Optional `base.toml` in the current working directory

**Precedence:** _User TOML_ > _Package TOML_ > _Class Defaults_

### Core Configuration Classes

- `LLMConfig`: Language model configuration
- `VectorStoreConfig`: Vector database configuration  
- `RerankerConfig`: Reranking model configuration
- `SensoryConfig`: Document processing configuration

### Configuration Methods

- `register_config(name, config)`: Register a new submodule configuration
- `get_config(name)`: Get a submodule configuration by name
- `get_all_configs()`: Get all registered submodule configurations
- `llm`, `vector_store`, `reranker`, `sensory`: Convenience properties

## Testing

### Test Structure

```
tests/
├── conftest.py              # Pytest configuration and common fixtures
├── test_cogent_config.py    # Configuration tests
├── providers/
│   └── test_completion.py   # LiteLLM completion model tests
└── README.md               # Testing guide
```

### Test Categories

- **Unit Tests**: Test individual components in isolation (mocked dependencies)
- **Integration Tests**: Test components with real external services

### Integration Test Setup

#### Ollama Setup
```bash
# Install Ollama
brew install ollama  # macOS
# or
curl -fsSL https://ollama.ai/install.sh | sh  # Linux

# Start service
ollama serve

# Pull required models
ollama pull llama3.2:latest
ollama pull qwen2.5vl:latest
```

#### OpenAI Setup
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Building and Publishing

### Building
```bash
# Build package
make build

# Clean build artifacts
make clean
```

### Publishing

#### Prerequisites
```bash
# Install twine if not already available
pip install twine

# Set PyPI credentials (one of these methods):
export TWINE_USERNAME="your-username"
export TWINE_PASSWORD="your-password"
# or configure ~/.pypirc
```

#### Publish to TestPyPI
```bash
make publish-test
```

#### Publish to PyPI
```bash
make publish
```

### Release Process
```bash
# Full release workflow
make release  # Equivalent to: clean build
```

## CI/CD

### CI Pipeline Commands
```bash
# Run CI pipeline (quality checks + tests)
make ci

# CI quality checks only
make ci-quality

# CI tests only  
make ci-test
```

## Documentation

### Building Documentation
```bash
# Build documentation
make docs

# Build and serve locally
make docs-serve  # Available at http://localhost:8000

# Clean documentation build artifacts
make docs-clean
```

## Development Commands Reference

| Command | Description |
|---------|-------------|
| `make help` | Show available commands |
| `make install` | Install dependencies |
| `make install-dev` | Install development dependencies |
| `make setup` | Setup development environment |
| `make test` | Run all tests |
| `make test-unit` | Run unit tests only |
| `make test-integration` | Run integration tests only |
| `make test-coverage` | Run tests with coverage |
| `make format` | Format code |
| `make format-check` | Check code formatting |
| `make lint` | Run linting checks |
| `make quality` | Run quality checks |
| `make dev-check` | Quick development check |
| `make full-check` | Full development check |
| `make build` | Build package |
| `make clean` | Clean build artifacts |
| `make publish` | Publish to PyPI |
| `make publish-test` | Publish to TestPyPI |
| `make ci` | Run CI pipeline |

## Troubleshooting

### Common Issues

1. **Poetry not found**:
   ```bash
   pip install poetry
   ```

2. **Python version issues**:
   ```bash
   pyenv install 3.11
   pyenv local 3.11
   ```

3. **Test dependencies missing**:
   ```bash
   make install-dev
   ```

4. **Integration tests failing**:
   - Check if external services (Ollama, OpenAI) are properly configured
   - Verify API keys and service availability
   - Review test logs for specific error messages

## Contributing

1. Follow the development workflow outlined above
2. Ensure all tests pass: `make full-check`
3. Update documentation as needed
4. Submit pull requests with clear descriptions

For detailed testing information, see [tests/README.md](tests/README.md).