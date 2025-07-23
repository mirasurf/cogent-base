# Extending Configuration Examples

Examples showing how to extend and customize the configuration system.

## Files

- `custom_config.py` - Create custom configuration classes and extend CogentBaseConfig
- `config_with_toml.py` - Demonstrates TOML configuration loading order

## Usage

```bash
python custom_config.py
python config_with_toml.py
```

## Configuration Loading Order

1. **Class Defaults**: Python default values defined in config classes
2. **Package TOML**: Built-in `base.toml` shipped with the package  
3. **User Runtime TOML**: Optional `base.toml` in the current working directory

**Precedence:** User TOML > Package TOML > Class Defaults