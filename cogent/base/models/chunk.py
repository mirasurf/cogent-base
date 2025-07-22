from typing import Any, Dict, List

import numpy as np
from pydantic import BaseModel, Field

Embedding = List[float] | List[List[float]] | np.ndarray


class ObjectChunk(BaseModel):
    """Represents a chunk stored in VectorStore"""

    object_id: str  # external_id of parent object
    content: str
    embedding: Embedding
    chunk_number: int
    # chunk-specific metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0
    model_config = {"arbitrary_types_allowed": True}


class Chunk(BaseModel):
    """Represents a chunk containing content and metadata"""

    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    def to_object_chunk(self, object_id: str, chunk_number: int, embedding: Embedding) -> ObjectChunk:
        return ObjectChunk(
            object_id=object_id,
            content=self.content,
            embedding=embedding,
            chunk_number=chunk_number,
            metadata=self.metadata,
        )
