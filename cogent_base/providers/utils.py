import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def is_ollama_model(model_name: str, model_key: str) -> bool:
    """Check if a model is an Ollama model"""
    model_name = model_name or ""
    model_key = model_key or ""
    return "ollama" in model_name.lower() or "ollama" in model_key.lower()


def initialize_ollama_model(model_key: str, model_config: Dict) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Initialize Ollama model configuration and validate requirements.

    Args:
        model_key: The key of the model in the configuration
        model_config: The model configuration dictionary

    Returns:
        Tuple of (is_ollama, ollama_api_base, ollama_base_model_name)
        - is_ollama: Whether to use direct Ollama client
        - ollama_api_base: The API base URL for Ollama (None if not available)
        - ollama_base_model_name: The base model name for Ollama (None if not available)
    """
    # Check if it's an Ollama model for potential direct usage
    is_ollama = is_ollama_model(model_config.get("model_name", ""), model_key)
    ollama_api_base = None
    ollama_base_model_name = None

    if is_ollama:
        try:
            import ollama  # noqa: F401
        except ImportError:
            logger.warning("Ollama model selected, but 'ollama' library not installed. Falling back to LiteLLM.")
            return False, None, None  # Fallback to LiteLLM if library missing

        ollama_api_base = model_config.get("api_base")
        if not ollama_api_base:
            logger.warning(
                f"Ollama model {model_key} selected for direct use, "
                "but 'api_base' is missing in config. Falling back to LiteLLM."
            )
            return False, None, None  # Fallback if api_base is missing

        ollama_base_model_name = model_config.get("model_name", None)
        if not ollama_base_model_name:
            logger.warning(
                f"Could not parse base model name from Ollama model "
                f"{model_config.get('model_name', '')}. Falling back to LiteLLM."
            )
            return False, None, None  # Fallback if name parsing fails

    return is_ollama, ollama_api_base, ollama_base_model_name
