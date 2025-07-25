"""
Copyright (c) 2025 Mirasurf
Copyright (c) 2023-2025 mem0ai/mem0
Original code from https://github.com/mem0ai/mem0
Licensed under the Apache License, Version 2.0
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .models import OutputData


class BaseVectorStore(ABC):
    @abstractmethod
    def create_col(self, vector_size: int, distance: str = "cosine") -> None:
        """Create a new collection."""

    @abstractmethod
    def insert(
        self,
        vectors: List[List[float]],
        payloads: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """Insert vectors into a collection."""

    @abstractmethod
    def search(
        self, query: str, vectors: List[float], limit: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[OutputData]:
        """Search for similar vectors."""

    @abstractmethod
    def delete(self, vector_id: str) -> None:
        """Delete a vector by ID."""

    @abstractmethod
    def update(
        self, vector_id: str, vector: Optional[List[float]] = None, payload: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update a vector and its payload."""

    @abstractmethod
    def get(self, vector_id: str) -> Optional[OutputData]:
        """Retrieve a vector by ID."""

    @abstractmethod
    def list_cols(self) -> List[str]:
        """List all collections."""

    @abstractmethod
    def delete_col(self) -> None:
        """Delete a collection."""

    @abstractmethod
    def col_info(self) -> Dict[str, Any]:
        """Get information about a collection."""

    @abstractmethod
    def list(self, filters: Optional[Dict[str, Any]] = None, limit: Optional[int] = None) -> List[OutputData]:
        """List all memories."""

    @abstractmethod
    def reset(self) -> None:
        """Reset by delete the collection and recreate it."""
