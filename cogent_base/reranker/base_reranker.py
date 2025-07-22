"""
Copyright (c) 2025 Mirasurf
Copyright (c) 2023-2025 morphik/morphik-core
Original code from https://github.com/morphik/morphik-core
"""

from abc import ABC, abstractmethod
from typing import List, Union

from cogent_base.models.chunk import ObjectChunk


class BaseReranker(ABC):
    """Base class for reranking search results"""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        chunks: List[ObjectChunk],
    ) -> List[ObjectChunk]:
        """Rerank chunks based on their relevance to the query"""

    @abstractmethod
    async def compute_score(
        self,
        query: str,
        text: Union[str, List[str]],
    ) -> Union[float, List[float]]:
        """Compute relevance scores between query and text"""
