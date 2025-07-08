"""
Copyright (c) 2025 Mirasurf
Copyright (c) 2023-2025 morphik/morphik-core
Original code from https://github.com/morphik/morphik-core
"""

from abc import ABC, abstractmethod

from cogent_base.models.completion import CompletionRequest, CompletionResponse


class BaseCompletionModel(ABC):
    """Base class for completion models"""

    @abstractmethod
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate completion from query and context"""
