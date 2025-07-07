import logging
from typing import Optional

from cogentbase.config import get_cogent_config
from cogentbase.providers.vector_store.base_vector_store import VectorStoreBase

logger = logging.getLogger(__name__)


class CogentVectorStore(VectorStoreBase):
    def __init__(self, store_key: str):
        """
        Initialize Cogent vector store with a model key from registered_vector_stores.

        Args:
            store_key: The key of the store in the registered_vector_stores config
        """
        settings = get_cogent_config()
        self.store_key = store_key
        self.store_impl: Optional[VectorStoreBase] = None

        # Get the model configuration from registered_models
        if (
            not hasattr(settings.vector_store, "registered_vector_stores")
            or store_key not in settings.vector_store.registered_vector_stores
        ):
            raise ValueError(f"Store '{store_key}' not found in registered_vector_stores configuration")

        self.store_config = settings.vector_store.registered_vector_stores[store_key]
        self.provider = settings.vector_store.provider

        # Set these attributes before using them in the provider initialization
        self.collection_name = settings.vector_store.collection_name
        self.embedding_model_dims = settings.vector_store.embedding_model_dims

        if self.provider == "pgvector":
            from cogentbase.providers.vector_store.pgvector_vector_store import (
                PGVector,
            )

            self.store_impl = PGVector(
                dbname=self.store_config["dbname"],
                user=self.store_config["user"],
                password=self.store_config["password"],
                host=self.store_config["host"],
                port=self.store_config["port"],
                diskann=self.store_config["diskann"],
                hnsw=self.store_config["hnsw"],
            )
        elif self.provider == "weaviate":
            from cogentbase.providers.vector_store.weaviate_vector_store import (
                Weaviate,
            )

            self.store_impl = Weaviate(
                collection_name=self.collection_name,
                embedding_model_dims=self.embedding_model_dims,
                cluster_url=self.store_config["cluster_url"],
                auth_client_secret=self.store_config["auth_client_secret"],
                additional_headers=self.store_config["additional_headers"],
            )
        else:
            raise ValueError(f"Provider '{self.provider}' not supported")

        logger.info(f"Initialized Cogent vector store with store_key={store_key}, config={self.store_config}")

    def create_col(self, embedding_model_dims):
        self.store_impl.create_col(embedding_model_dims)

    def insert(self, vectors, payloads=None, ids=None):
        self.store_impl.insert(vectors, payloads, ids)

    def search(self, query, vectors, limit=5, filters=None):
        return self.store_impl.search(query, vectors, limit, filters)

    def delete(self, vector_id):
        self.store_impl.delete(vector_id)

    def update(self, vector_id, vector=None, payload=None):
        self.store_impl.update(vector_id, vector, payload)

    def get(self, vector_id):
        return self.store_impl.get(vector_id)

    def list_cols(self):
        return self.store_impl.list_cols()

    def delete_col(self):
        self.store_impl.delete_col()

    def col_info(self):
        return self.store_impl.col_info()

    def list(self, filters=None, limit=None):
        return self.store_impl.list(filters, limit)

    def reset(self):
        self.store_impl.reset()
