#!/usr/bin/env python3
"""
Comprehensive Configuration Example

This example demonstrates:
1. Basic config directory functionality
2. TOML-based configuration extension
3. Class-based configuration extension
4. Configuration loading order and precedence
"""

import os
from pathlib import Path

from cogent_base.config import BaseConfig, CogentBaseConfig, toml_config
from cogent_base.config.core import get_cogent_config, init_cogent_config, set_cogent_config_dir

# =============================================================================
# 1. CUSTOM CONFIGURATION CLASSES
# =============================================================================


@toml_config("agent")
class AgentConfig(BaseConfig):
    """Custom agent configuration that can be loaded from TOML."""

    name: str = "default_agent"
    max_conversations: int = 10
    timeout: int = 30
    enable_memory: bool = True
    memory_size: int = 1000


@toml_config("monitoring")
class MonitoringConfig(BaseConfig):
    """Custom monitoring configuration."""

    enable_logging: bool = True
    log_level: str = "INFO"
    metrics_enabled: bool = False
    alert_threshold: float = 0.8


@toml_config("security")
class SecurityConfig(BaseConfig):
    """Custom security configuration."""

    enable_encryption: bool = True
    encryption_key: str = ""
    max_retries: int = 3
    session_timeout: int = 3600


# =============================================================================
# 2. EXTENDED CONFIGURATION CLASS
# =============================================================================


class MyCogentConfig(CogentBaseConfig):
    """Extended configuration class with custom configs."""

    def _load_default_configs(self) -> None:
        """Load both parent and custom configurations."""
        # Load parent configs (llm, vector_store, reranker, sensory)
        super()._load_default_configs()

        # Add custom configs
        self.register_config("agent", AgentConfig())
        self.register_config("monitoring", MonitoringConfig())
        self.register_config("security", SecurityConfig())

    # Convenience properties for custom configs
    @property
    def agent(self) -> AgentConfig:
        """Get agent configuration."""
        return self.get_config("agent")

    @property
    def monitoring(self) -> MonitoringConfig:
        """Get monitoring configuration."""
        return self.get_config("monitoring")

    @property
    def security(self) -> SecurityConfig:
        """Get security configuration."""
        return self.get_config("security")


# =============================================================================
# 3. CONFIGURATION UTILITY FUNCTIONS
# =============================================================================


def create_sample_toml_config() -> str:
    """Create a sample TOML configuration file content."""
    return """# Custom Cogent Configuration
# This file demonstrates configuration loading order and custom extensions

# =============================================================================
# REGISTERED MODELS
# =============================================================================
[registered_models.openai_gpt4]
model_name = "gpt-4"
api_key = "your-openai-api-key"

[registered_models.anthropic_claude]
model_name = "claude-3-sonnet-20240229"
api_key = "your-anthropic-api-key"

# =============================================================================
# COMPLETION SETTINGS (overrides class defaults)
# =============================================================================
[completion]
provider = "openai"
model = "openai_gpt4"
default_max_tokens = 2000
default_temperature = 0.1

# =============================================================================
# VECTOR STORE SETTINGS
# =============================================================================
[vector_store]
provider = "weaviate"
collection_name = "my_project"
embedding_model_dims = 1536

# =============================================================================
# SENSORY SETTINGS
# =============================================================================
[sensory.parser]
chunk_size = 8000
chunk_overlap = 400
use_contextual_chunking = true
contextual_chunking_model = "openai_gpt4"

# =============================================================================
# CUSTOM AGENT CONFIGURATION
# =============================================================================
[agent]
name = "production_agent"
max_conversations = 50
timeout = 60
enable_memory = true
memory_size = 2000

# =============================================================================
# CUSTOM MONITORING CONFIGURATION
# =============================================================================
[monitoring]
enable_logging = true
log_level = "DEBUG"
metrics_enabled = true
alert_threshold = 0.9

# =============================================================================
# CUSTOM SECURITY CONFIGURATION
# =============================================================================
[security]
enable_encryption = true
encryption_key = "your-encryption-key"
max_retries = 5
session_timeout = 7200
"""


def create_minimal_toml_config() -> str:
    """Create a minimal TOML configuration for testing."""
    return """[completion]
provider = "anthropic"
model = "anthropic_claude"
default_temperature = 0.2

[agent]
name = "test_agent"
max_conversations = 5
"""


# =============================================================================
# 4. MAIN DEMONSTRATION FUNCTION
# =============================================================================


def demonstrate_config_loading() -> None:
    """Demonstrate the complete configuration loading process."""

    print("=== Comprehensive Configuration Example ===\n")

    # Step 1: Show default configuration
    print("1. DEFAULT CONFIGURATION (class defaults):")
    default_config = get_cogent_config()
    print(f"   LLM Provider: {default_config.llm.completion_provider}")
    print(f"   LLM Model: {default_config.llm.completion_model}")
    print(f"   Vector Store: {default_config.vector_store.provider}")
    print(f"   Chunk Size: {default_config.sensory.chunk_size}")
    print()

    # Step 2: Create custom config directory with TOML
    print("2. CUSTOM CONFIGURATION WITH TOML:")
    custom_dir = Path("./custom_config_demo")
    custom_dir.mkdir(exist_ok=True)

    # Create comprehensive TOML config
    toml_content = create_sample_toml_config()
    with open(custom_dir / ".cogent.toml", "w") as f:
        f.write(toml_content)

    print("   Created .cogent.toml with custom settings")
    print("   - LLM: OpenAI GPT-4")
    print("   - Vector Store: Weaviate")
    print("   - Custom Agent: production_agent")
    print("   - Custom Monitoring: DEBUG logging")
    print()

    # Step 3: Use extended config class with custom directory
    print("3. EXTENDED CONFIGURATION CLASS:")
    extended_config = MyCogentConfig(config_dir=custom_dir)

    print("   Core Settings:")
    print(f"     LLM Provider: {extended_config.llm.completion_provider}")
    print(f"     LLM Model: {extended_config.llm.completion_model}")
    print(f"     Vector Store: {extended_config.vector_store.provider}")
    print(f"     Chunk Size: {extended_config.sensory.chunk_size}")

    print("   Custom Settings:")
    print(f"     Agent Name: {extended_config.agent.name}")
    print(f"     Max Conversations: {extended_config.agent.max_conversations}")
    print(f"     Memory Enabled: {extended_config.agent.enable_memory}")
    print(f"     Log Level: {extended_config.monitoring.log_level}")
    print(f"     Encryption: {extended_config.security.enable_encryption}")
    print()

    # Step 4: Demonstrate config directory switching
    print("4. CONFIGURATION DIRECTORY SWITCHING:")

    # Create another config directory
    another_dir = Path("./another_config_demo")
    another_dir.mkdir(exist_ok=True)

    minimal_toml = create_minimal_toml_config()
    with open(another_dir / ".cogent.toml", "w") as f:
        f.write(minimal_toml)

    # Switch to new directory
    set_cogent_config_dir(another_dir)
    switched_config = get_cogent_config()

    print("   Switched to minimal config:")
    print(f"     LLM Provider: {switched_config.llm.completion_provider}")
    print(f"     LLM Model: {switched_config.llm.completion_model}")
    print(f"     Agent Name: {switched_config.agent.name if hasattr(switched_config, 'agent') else 'N/A'}")
    print()

    # Step 5: Show configuration precedence
    print("5. CONFIGURATION PRECEDENCE:")
    print("   Order (highest to lowest priority):")
    print("   1. User TOML (.cogent.toml in config directory)")
    print("   2. Environment variables (COGENT_CONFIG_DIR)")
    print("   3. Package defaults (base.toml)")
    print("   4. Class defaults (Python code)")
    print()

    # Step 6: Demonstrate environment variable usage
    print("6. ENVIRONMENT VARIABLE CONFIGURATION:")
    env_dir = Path("./env_config_demo")
    env_dir.mkdir(exist_ok=True)

    env_toml = """[completion]
provider = "ollama"
model = "llama2"

[agent]
name = "env_agent"
"""

    with open(env_dir / ".cogent.toml", "w") as f:
        f.write(env_toml)

    # Set environment variable
    original_env = os.environ.get("COGENT_CONFIG_DIR")
    os.environ["COGENT_CONFIG_DIR"] = str(env_dir.absolute())

    # Initialize config (should use environment variable)
    env_config = init_cogent_config()

    print(f"   Environment COGENT_CONFIG_DIR: {os.environ['COGENT_CONFIG_DIR']}")
    print(f"   LLM Provider: {env_config.llm.completion_provider}")
    print(f"   LLM Model: {env_config.llm.completion_model}")
    print(f"   Agent Name: {env_config.agent.name if hasattr(env_config, 'agent') else 'N/A'}")
    print()

    # Restore environment
    if original_env:
        os.environ["COGENT_CONFIG_DIR"] = original_env
    else:
        os.environ.pop("COGENT_CONFIG_DIR", None)

    # Cleanup
    print("7. CLEANUP:")
    for dir_path in [custom_dir, another_dir, env_dir]:
        config_file = dir_path / ".cogent.toml"
        if config_file.exists():
            config_file.unlink()
        if dir_path.exists():
            dir_path.rmdir()
    print("   Removed temporary configuration directories")


def main():
    """Run the comprehensive configuration demonstration."""
    try:
        demonstrate_config_loading()
    except Exception as e:
        print(f"Error during demonstration: {e}")
        print("This might be due to missing dependencies or configuration issues.")


if __name__ == "__main__":
    main()
