"""
Copyright (c) 2025 Mirasurf
Copyright (c) 2023-2025 morphik/morphik-core
Original code from https://github.com/morphik/morphik-core
"""

from cogentbase.providers.embedding.base_embedding import BaseEmbeddingModel
from cogentbase.providers.embedding.litellm_embedding import LiteLLMEmbeddingModel

__all__ = ["BaseEmbeddingModel", "LiteLLMEmbeddingModel"]
