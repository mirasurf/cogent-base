"""
Custom Configuration Extension Example

This example shows how to create custom configuration classes 
and extend CogentBaseConfig.
"""

from cogent_base.config import CogentBaseConfig, BaseConfig, toml_config

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

print(f"Agent Name: {agent_config.name}")
print(f"Max Conversations: {agent_config.max_conversations}")
print(f"Memory Enabled: {agent_config.enable_memory}")