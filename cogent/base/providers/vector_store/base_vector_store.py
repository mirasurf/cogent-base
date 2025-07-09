"""
Copyright (c) 2025 Mirasurf
Copyright (c) 2023-2025 mem0ai/mem0
Original code from https://github.com/mem0ai/mem0
Licensed under the Apache License, Version 2.0
"""

from abc import ABC, abstractmethod


class VectorStoreBase(ABC):
    @abstractmethod
    def create_col(self, name, vector_size, distance):
        """Create a new collection."""

    @abstractmethod
    def insert(self, vectors, payloads=None, ids=None):
        """Insert vectors into a collection."""

    @abstractmethod
    def search(self, query, vectors, limit=5, filters=None):
        """Search for similar vectors."""

    @abstractmethod
    def delete(self, vector_id):
        """Delete a vector by ID."""

    @abstractmethod
    def update(self, vector_id, vector=None, payload=None):
        """Update a vector and its payload."""

    @abstractmethod
    def get(self, vector_id):
        """Retrieve a vector by ID."""

    @abstractmethod
    def list_cols(self):
        """List all collections."""

    @abstractmethod
    def delete_col(self):
        """Delete a collection."""

    @abstractmethod
    def col_info(self):
        """Get information about a collection."""

    @abstractmethod
    def list(self, filters=None, limit=None):
        """List all memories."""

    @abstractmethod
    def reset(self):
        """Reset by delete the collection and recreate it."""
