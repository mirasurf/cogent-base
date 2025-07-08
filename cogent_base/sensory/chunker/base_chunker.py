from abc import ABC, abstractmethod
from typing import List

from cogent_base.models.chunk import Chunk


class BaseChunker(ABC):
    """Base class for text chunking strategies"""

    @abstractmethod
    async def split_text(self, text: str) -> List[Chunk]:
        """Split text into chunks"""
