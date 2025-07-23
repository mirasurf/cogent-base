# Cogent Base Examples

This directory contains practical examples demonstrating how to use various features of cogent-base.

## Example Categories

### Basic Configuration
- [`basic_usage.py`](basic-config/basic_usage.py) - Get and use global configuration

### Extending Configuration  
- [`custom_config.py`](extending-config/custom_config.py) - Create custom configuration classes
- [`config_with_toml.py`](extending-config/config_with_toml.py) - TOML configuration overrides

### Providers
- [`llm_provider.py`](providers/llm_provider.py) - LLM completion provider usage
- [`embedding_provider.py`](providers/embedding_provider.py) - Text embedding provider usage
- [`vector_store_provider.py`](providers/vector_store_provider.py) - Vector store operations

### Sensory Processing
- [`document_parsing.py`](sensory/document_parsing.py) - Parse documents with CogentParser
- [`text_chunking.py`](sensory/text_chunking.py) - Split text into chunks

## Running Examples

1. Install cogent-base:
   ```bash
   pip install cogent-base
   ```

2. Run any example:
   ```bash
   python examples/basic-config/basic_usage.py
   ```

## Prerequisites

Some examples may require additional setup:
- **LLM Providers**: OpenAI API key or Ollama service
- **Vector Store**: Weaviate instance running
- **Configuration**: Proper TOML files for custom configurations