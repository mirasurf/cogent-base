# Cogent Base

[![PyPI version](https://img.shields.io/pypi/v/cogent-base)](https://pypi.python.org/pypi/cogent-base)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/mirasurf/cogent-base/actions/workflows/ci.yml/badge.svg)](https://github.com/mirasurf/cogent-base/actions/workflows/ci.yml)

A shared Python module for agentic cognitive computing frameworks, providing extensible configuration management, logging utilities, and core components.

## Features

- **Extensible Configuration System**: Register custom configurations with TOML support
- **Flexible Logging**: Basic logging utilities that can be overridden by downstream libraries
- **Provider Abstraction**: Unified interfaces for LLM, embedding, reranking, and vector store providers
- **Sensory Processing**: Document parsing and text chunking capabilities
- **Modular Design**: Clean separation of concerns with extensible architecture

## Installation

**Requirements**: Python 3.11+

```bash
pip install cogent-base
```

For development:

```bash
git clone https://github.com/mirasurf/cogent-base.git
cd cogent-base
make install-dev
```

## Quick Start

### Basic Configuration

```python
from cogentbase.config import get_cogent_config, BaseConfig, toml_config

# Get the global configuration
config = get_cogent_config()

# Access built-in configurations
llm_config = config.llm
vector_store_config = config.vector_store
```

### Extending Configuration

Create custom configuration classes and extend `CogentBaseConfig`:

```python
from cogentbase.config import CogentBaseConfig, BaseConfig, toml_config

@toml_config("agent")
class AgentConfig(BaseConfig):
    name: str = "default_agent"
    max_conversations: int = 10
    timeout: int = 30
    enable_memory: bool = True

class MyCogentConfig(CogentBaseConfig):
    def _load_default_configs(self):
        # Load parent configs
        super()._load_default_configs()
        # Add custom configs
        self.register_config("agent", AgentConfig())

# Use your extended config
config = MyCogentConfig()
agent_config = config.get_config("agent")
```

## Configuration System

### Configuration Loading Order

Cogent's configuration system follows a **layered precedence**:

1. **Class Defaults**: Python default values defined in config classes
2. **Package TOML**: Built-in `base.toml` shipped with the package
3. **User Runtime TOML**: Optional `base.toml` in the current working directory

**Precedence:** _User TOML_ > _Package TOML_ > _Class Defaults_

#### Example

```toml
# Package base.toml (shipped with cogent-base)
[completion]
model = "ollama_qwen_vision"

# User base.toml (in project directory) - overrides package default
[completion]
model = "my_custom_model"
```

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

## Provider System

### LLM Providers

```python
from cogentbase.providers.completion import LiteLLMCompletionModel

model = LiteLLMCompletionModel("gpt-4")
response = await model.complete(request)
```

### Embedding Providers

```python
from cogentbase.providers.embedding import LiteLLMEmbeddingModel

model = LiteLLMEmbeddingModel("text-embedding-ada-002")
embeddings = await model.embed_texts(["Hello", "World"])
```

### Vector Store Providers

```python
from cogentbase.providers.vector_store import WeaviateVectorStore

store = WeaviateVectorStore()
await store.insert(vectors, metadata)
results = await store.search(query_vector, limit=10)
```

## Sensory Processing

### Document Parsing

```python
from cogentbase.sensory.parser import CogentParser

parser = CogentParser()
metadata, elements = await parser.parse_file_to_text(file_content, filename)
```

### Text Chunking

```python
from cogentbase.sensory.chunker import StandardChunker

chunker = StandardChunker(chunk_size=1000, overlap=200)
chunks = await chunker.split_text(long_text)
```

## Development

### Running Tests

```bash
# Unit tests
make test-unit

# Integration tests  
make test-integration

# With coverage
make test-coverage
```

### Code Quality

```bash
# Format code
make format

# Check quality
make quality

# Lint only
make lint
```

### Building

```bash
# Build package
make build

# Clean build artifacts
make clean
```

## License

MIT
