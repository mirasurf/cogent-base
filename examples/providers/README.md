# Provider Examples

Examples demonstrating usage of various provider interfaces in cogent-base.

## Files

- `llm_provider.py` - LLM completion model usage
- `embedding_provider.py` - Text embedding model usage  
- `vector_store_provider.py` - Vector store operations

## Prerequisites

- **LLM Provider**: OpenAI API key or Ollama service running
- **Vector Store**: Weaviate instance (for vector store example)

## Setup

1. For OpenAI:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```

2. For Ollama:
   ```bash
   ollama serve
   ollama pull llama3.2:latest
   ```

3. For Weaviate:
   ```bash
   docker run -p 8080:8080 semitechnologies/weaviate:latest
   ```

## Usage

```bash
python llm_provider.py
python embedding_provider.py  
python vector_store_provider.py
```