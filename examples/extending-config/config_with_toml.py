"""
TOML Configuration Override Example

This example demonstrates the configuration loading order:
Class Defaults < Package TOML < User TOML
"""

# Create a sample base.toml file for demonstration
sample_toml = """
# User base.toml (in project directory) - overrides package default
[completion]
model = "my_custom_model"

[agent]
name = "production_agent"
max_conversations = 50
timeout = 60
"""

print("Sample base.toml configuration:")
print(sample_toml)

# In practice, you would have this file in your project directory
# and the configuration system would automatically load it
from cogent_base.config import get_cogent_config

config = get_cogent_config()
print(f"Loaded completion model: {config.llm.model}")