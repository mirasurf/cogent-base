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

To use advanced vendor's features, install with

```bash
pip install cogent-base[extensions]
```

to enable features:

- vector stores of weaviate
- FlagEmbedding
- video parser with assemblyai
- smart frame with pandasai
- smart voice with ailyun lingjie NLS

For development:

```bash
git clone https://github.com/mirasurf/cogent-base.git
cd cogent-base
make install-dev
```

## Quick Start

```python
from cogent_base.config import get_cogent_config

# Get the global configuration
config = get_cogent_config()

# Access built-in configurations
llm_config = config.llm
vector_store_config = config.vector_store
```

## Documentation

- **[Examples](examples/)** - Practical usage examples for all features
- **[Development Guide](DEVELOPMENT.md)** - Setup, testing, and deployment procedures
- **[Testing Guide](tests/README.md)** - Comprehensive testing documentation

## Core Components

### Configuration System
- Layered configuration loading (Class Defaults → Package TOML → User TOML)
- Custom configuration class registration
- TOML-based configuration files

### Provider Interfaces
- **LLM Providers**: LiteLLM-based completion models
- **Embedding Providers**: Text embedding generation
- **Vector Stores**: Weaviate integration for vector operations
- **Reranking**: Document reranking capabilities

### Sensory Processing
- **Document Parsing**: Multi-format document text extraction
- **Text Chunking**: Configurable text splitting with overlap

### Smart Processing
- **Smart DataFrame**: dataframe talks to LLM with pandasai
- **Smart Voice**: voice-to-text transcription based on Aliyun Lingji AI.

## License

MIT
