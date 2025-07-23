# Cogent Configuration Examples

This directory contains examples demonstrating how to use and extend the Cogent configuration system.

## Files

- `custom_config.py` - Comprehensive configuration example showing TOML-based and class-based extension

## Usage

```bash
# Run the comprehensive configuration example
poetry run python examples/custom_config.py
```

## What the Example Demonstrates

The `custom_config.py` example shows:

### 1. Basic Configuration Directory Functionality
- Setting custom config directories
- Switching between different configuration directories
- Environment variable-based configuration

### 2. TOML-Based Configuration Extension
- Creating custom `.cogent.toml` files
- Overriding default settings
- Configuration loading order and precedence

### 3. Class-Based Configuration Extension
- Creating custom configuration classes with `@toml_config` decorator
- Extending `CogentBaseConfig` with custom configs
- Adding convenience properties for custom configurations

### 4. Configuration Precedence
The example demonstrates the configuration loading order:
1. **User TOML** (.cogent.toml in config directory) - Highest priority
2. **Environment variables** (COGENT_CONFIG_DIR)
3. **Package defaults** (base.toml)
4. **Class defaults** (Python code) - Lowest priority

## Key Features Demonstrated

- **Custom Config Classes**: `AgentConfig`, `MonitoringConfig`, `SecurityConfig`
- **Extended Config Class**: `MyCogentConfig` that includes custom configs
- **Dynamic Directory Switching**: Using `set_cogent_config_dir()` and `init_cogent_config()`
- **Environment Variable Support**: Using `COGENT_CONFIG_DIR` environment variable
- **TOML Integration**: Loading configuration from TOML files with custom sections

## Configuration Loading Order

The example shows how different configuration sources are merged:

1. **Class Defaults**: Python default values defined in config classes
2. **Package TOML**: Built-in configuration shipped with the package
3. **User Runtime TOML**: Optional `.cogent.toml` in the current working directory or specified config directory
4. **Environment Variables**: `COGENT_CONFIG_DIR` for specifying config directory

**Precedence:** User TOML > Environment Variables > Package TOML > Class Defaults