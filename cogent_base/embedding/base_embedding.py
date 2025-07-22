"""
Copyright (c) 2025 Mirasurf
Copyright (c) 2023-2025 morphik/morphik-core
Original code from https://github.com/morphik/morphik-core
"""

from abc import ABC, abstractmethod
from typing import List, Union

from cogent_base.models.chunk import Chunk


class BaseEmbeddingModel(ABC):
    @abstractmethod
    async def embed_for_chunks(self, chunks: Union[Chunk, List[Chunk]]) -> List[List[float]]:
        """Generate embeddings for input text"""

    @abstractmethod
    async def embed_for_query(self, text: str) -> List[float]:
        """Generate embeddings for input text"""
