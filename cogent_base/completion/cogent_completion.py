"""
Copyright (c) 2025 Mirasurf
"""

import logging
from typing import AsyncGenerator, Union

from cogent_base.completion.base_completion import BaseCompletionModel
from cogent_base.config import get_cogent_config
from cogent_base.config.consts import COGENT_RERANKER_COMPLETION_LITTELM, COGENT_RERANKER_COMPLETION_OLLAMA
from cogent_base.models.completion import CompletionRequest, CompletionResponse

logger = logging.getLogger(__name__)


class CogentCompletionModel(BaseCompletionModel):
    """
    Cogent completion model that provides a unified interface for different completion providers.
    Uses registered models from the config file and routes to appropriate implementations.
    """

    def __init__(self, model_key: str) -> None:
        """
        Initialize Cogent completion model with a model key from registered_models.

        Args:
            model_key: The key of the model in the registered_models config
        """
        settings = get_cogent_config()
        self.model_key = model_key
        self.completion_impl: BaseCompletionModel = None

        # Get the model configuration from registered_models
        if not hasattr(settings.llm, "registered_models") or model_key not in settings.llm.registered_models:
            raise ValueError(f"Model '{model_key}' not found in registered_models configuration")

        self.model_config = settings.llm.registered_models[model_key]
        self.provider = settings.llm.completion_provider

        # Initialize the appropriate completion implementation
        if self.provider == COGENT_RERANKER_COMPLETION_LITTELM:
            from cogent_base.completion.litellm_completion import LiteLLMCompletionModel

            self.completion_impl = LiteLLMCompletionModel(model_key)
        elif self.provider == COGENT_RERANKER_COMPLETION_OLLAMA:
            from cogent_base.completion.ollama_completion import OllamaCompletionModel

            self.completion_impl = OllamaCompletionModel(model_key)
        else:
            raise ValueError(f"Completion provider '{self.provider}' not supported")

        logger.info(f"Initialized Cogent completion model with model_key={model_key}, config={self.model_config}")

    async def complete(self, request: CompletionRequest) -> Union[CompletionResponse, AsyncGenerator[str, None]]:
        """
        Generate completion using the appropriate provider implementation.

        Args:
            request: CompletionRequest object containing query, context, and parameters

        Returns:
            CompletionResponse object with the generated text and usage statistics or
            AsyncGenerator for streaming responses
        """
        return await self.completion_impl.complete(request)
