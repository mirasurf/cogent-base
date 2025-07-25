"""
Basic Configuration Usage Example

This example demonstrates how to get and use the global configuration
in cogent-base.
"""

from cogent_base.config import get_cogent_config

# Get the global configuration
config = get_cogent_config()

# Access built-in configurations
llm_config = config.llm
vector_store_config = config.vector_store

print(f"Completion Model: {llm_config.completion_model}")
print(f"Embedding Model: {llm_config.embedding_model}")
print(f"Vector Store Type: {vector_store_config.provider}")
