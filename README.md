# Cogent Base

[![PyPI version](https://badge.fury.io/py/cogent-base.svg)](https://badge.fury.io/py/cogent-base)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A shared Python module for agentic cognitive computing frameworks, providing extensible configuration management, logging utilities, and core components.

## Features

- **Extensible Configuration System**: Register custom configurations with TOML support
- **Flexible Logging**: Basic logging utilities that can be overridden by downstream libraries
- **Provider Abstraction**: Unified interfaces for LLM, embedding, reranking, and vector store providers
- **Sensory Processing**: Document parsing and text chunking capabilities
- **Modular Design**: Clean separation of concerns with extensible architecture

## Installation

**Requirements**: Python 3.10+

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
from cogent.base.config import get_cogent_config, BaseConfig, toml_config

# Get the global configuration
config = get_cogent_config()

# Access built-in configurations
llm_config = config.llm
vector_store_config = config.vector_store
```

### Custom Configuration

Create a custom configuration class:

```python
from cogent.base.config import BaseConfig, toml_config

@toml_config("agent")
class AgentConfig(BaseConfig):
    name: str = "default_agent"
    max_conversations: int = 10
    timeout: int = 30
    enable_memory: bool = True

# Register with the global config
config = get_cogent_config()
agent_config = AgentConfig()
config.register_config("agent", agent_config)
```

### TOML Configuration

Create a TOML file for your custom configuration:

```toml
# config/agent.toml
[agent]
name = "my_custom_agent"
max_conversations = 20
timeout = 60
enable_memory = false
```

The configuration will be automatically loaded and merged with your registered config.

### Logging

Cogent-base provides basic logging utilities that can be extended:

```python
from cogent.base.logger import get_logger, setup_logger_with_handlers

# Basic logger
logger = get_logger("my_module")

# Advanced logger with file handlers
logger = setup_logger_with_handlers(
    name="my_module",
    level="DEBUG",
    log_dir=Path("./logs"),
    enable_file_logging=True,
    enable_error_file=True
)
```

## Configuration System

### Core Configuration Classes

- `LLMConfig`: Language model configuration
- `VectorStoreConfig`: Vector database configuration  
- `RerankerConfig`: Reranking model configuration
- `SensoryConfig`: Document processing configuration

### Extending Configuration

1. **Create a custom config class**:
   ```python
   @toml_config("my_section")
   class MyCustomConfig(BaseConfig):
       setting1: str = "default"
       setting2: int = 100
   ```

2. **Register with global config**:
   ```python
   config = get_cogent_config()
   my_config = MyCustomConfig()
   config.register_config("my_module", my_config)
   ```

3. **Create TOML file** (optional):
   ```toml
   [my_section]
   setting1 = "custom_value"
   setting2 = 200
   ```

4. **Access in your code**:
   ```python
   config = get_cogent_config()
   my_config = config.get_config("my_module")
   # or use convenience property
   my_config = config.my_module
   ```

### Configuration Loading Order

1. Default values from config classes
2. TOML files (merged in order)
3. Environment variables
4. Runtime overrides

## Provider System

### LLM Providers

```python
from cogent.base.providers.completion import LiteLLMCompletionModel

model = LiteLLMCompletionModel("gpt-4")
response = await model.complete(request)
```

### Embedding Providers

```python
from cogent.base.providers.embedding import LiteLLMEmbeddingModel

model = LiteLLMEmbeddingModel("text-embedding-ada-002")
embeddings = await model.embed_texts(["Hello", "World"])
```

### Vector Store Providers

```python
from cogent.base.providers.vector_store import WeaviateVectorStore

store = WeaviateVectorStore()
await store.insert(vectors, metadata)
results = await store.search(query_vector, limit=10)
```

## Sensory Processing

### Document Parsing

```python
from cogent.base.sensory.parser import CogentParser

parser = CogentParser()
metadata, elements = await parser.parse_file_to_text(file_content, filename)
```

### Text Chunking

```python
from cogent.base.sensory.chunker import StandardChunker

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
