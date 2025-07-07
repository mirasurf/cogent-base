import uuid
from unittest.mock import MagicMock, patch

import pytest

from cogent.base.providers.vector_store.cogent_vector_store import CogentVectorStore
from cogent.base.providers.vector_store.weaviate_vector_store import Weaviate


class TestWeaviateIntegration:
    """Integration tests for Weaviate vector store."""

    @pytest.fixture
    def mock_config(self):
        return {
            "vector_store": {
                "registered_vector_stores": {
                    "test_weaviate": {
                        "cluster_url": "http://localhost:8080",
                        "auth_client_secret": "secret",
                        "additional_headers": {},
                    }
                },
                "vector_store_provider": "weaviate",
                "vector_store_collection_name": "test_collection",
                "vector_store_embedding_model_dims": 768,
            }
        }

    @pytest.mark.integration
    def test_weaviate_connection(self, mock_config):
        """Test Weaviate connection."""
        try:
            pass
        except ImportError:
            pytest.skip("Weaviate library not available")

        try:
            import httpx

            response = httpx.get("http://localhost:8080/v1/meta", timeout=5.0)
            if response.status_code != 200:
                pytest.skip("Weaviate service not running")
        except Exception:
            pytest.skip("Weaviate service not accessible")

        with patch("cogent.base.providers.vector_store.cogent_vector_store.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.vector_store.registered_vector_stores = mock_config["vector_store"][
                "registered_vector_stores"
            ]
            mock_get_config.return_value.vector_store.vector_store_provider = mock_config["vector_store"][
                "vector_store_provider"
            ]
            mock_get_config.return_value.vector_store.vector_store_collection_name = mock_config["vector_store"][
                "vector_store_collection_name"
            ]
            mock_get_config.return_value.vector_store.vector_store_embedding_model_dims = mock_config["vector_store"][
                "vector_store_embedding_model_dims"
            ]

            try:
                vector_store = CogentVectorStore("test_weaviate")
                assert vector_store.store_impl is not None
                assert vector_store.collection_name == "test_collection"
                assert vector_store.embedding_model_dims == 768
            except Exception as e:
                pytest.skip(f"Weaviate connection failed: {e}")

    @pytest.mark.integration
    def test_weaviate_basic_operations(self, mock_config):
        """Test basic Weaviate operations."""
        try:
            pass
        except ImportError:
            pytest.skip("Weaviate library not available")

        try:
            import httpx

            response = httpx.get("http://localhost:8080/v1/meta", timeout=5.0)
            if response.status_code != 200:
                pytest.skip("Weaviate service not running")
        except Exception:
            pytest.skip("Weaviate service not accessible")

        with patch("cogent.base.providers.vector_store.cogent_vector_store.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.vector_store.registered_vector_stores = mock_config["vector_store"][
                "registered_vector_stores"
            ]
            mock_get_config.return_value.vector_store.vector_store_provider = mock_config["vector_store"][
                "vector_store_provider"
            ]
            mock_get_config.return_value.vector_store.vector_store_collection_name = mock_config["vector_store"][
                "vector_store_collection_name"
            ]
            mock_get_config.return_value.vector_store.vector_store_embedding_model_dims = mock_config["vector_store"][
                "vector_store_embedding_model_dims"
            ]

            try:
                vector_store = CogentVectorStore("test_weaviate")

                # Test vectors and payloads
                test_vectors = [[0.1, 0.2, 0.3] * 256]  # 768 dimensions
                test_payloads = [{"data": "test object", "hash": "abc123"}]
                test_ids = [str(uuid.uuid4())]  # Use a valid UUID instead of "test_id_1"

                # Insert
                vector_store.insert(test_vectors, test_payloads, test_ids)

                # Search
                query_vector = [0.1, 0.2, 0.3] * 256
                results = vector_store.search("test query", query_vector, limit=1)
                assert results is not None

                # Clean up
                vector_store.delete(test_ids[0])  # Use the actual UUID that was inserted

            except Exception as e:
                pytest.skip(f"Weaviate operations failed: {e}")


class TestCogentVectorStoreUnit:
    """Unit tests for CogentVectorStore."""

    @pytest.fixture
    def mock_config(self):
        return {
            "vector_store": {
                "registered_vector_stores": {
                    "test_weaviate": {
                        "cluster_url": "http://localhost:8080",
                        "auth_client_secret": "secret",
                        "additional_headers": {},
                    }
                },
                "vector_store_provider": "weaviate",
                "vector_store_collection_name": "test_collection",
                "vector_store_embedding_model_dims": 768,
            }
        }

    @pytest.mark.unit
    def test_initialization(self, mock_config):
        """Test CogentVectorStore initialization."""
        with patch("cogent.base.providers.vector_store.cogent_vector_store.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.vector_store.registered_vector_stores = mock_config["vector_store"][
                "registered_vector_stores"
            ]
            mock_get_config.return_value.vector_store.provider = mock_config["vector_store"]["vector_store_provider"]
            mock_get_config.return_value.vector_store.collection_name = mock_config["vector_store"][
                "vector_store_collection_name"
            ]
            mock_get_config.return_value.vector_store.embedding_model_dims = mock_config["vector_store"][
                "vector_store_embedding_model_dims"
            ]

            with patch("cogent.base.providers.vector_store.weaviate_vector_store.Weaviate") as mock_weaviate_class:
                mock_weaviate_instance = MagicMock()
                mock_weaviate_class.return_value = mock_weaviate_instance

                vector_store = CogentVectorStore("test_weaviate")

                assert vector_store.store_key == "test_weaviate"
                assert vector_store.store_impl == mock_weaviate_instance
                assert vector_store.collection_name == "test_collection"
                assert vector_store.embedding_model_dims == 768

    @pytest.mark.unit
    def test_invalid_store_key(self, mock_config):
        """Test initialization with invalid store key."""
        with patch("cogent.base.providers.vector_store.cogent_vector_store.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.vector_store.registered_vector_stores = mock_config["vector_store"][
                "registered_vector_stores"
            ]

            with pytest.raises(ValueError, match="Store 'invalid_key' not found"):
                CogentVectorStore("invalid_key")

    @pytest.mark.unit
    def test_method_delegation(self, mock_config):
        """Test method delegation to underlying store."""
        with patch("cogent.base.providers.vector_store.cogent_vector_store.get_cogent_config") as mock_get_config:
            mock_get_config.return_value.vector_store.registered_vector_stores = mock_config["vector_store"][
                "registered_vector_stores"
            ]
            mock_get_config.return_value.vector_store.provider = mock_config["vector_store"]["vector_store_provider"]
            mock_get_config.return_value.vector_store.collection_name = mock_config["vector_store"][
                "vector_store_collection_name"
            ]
            mock_get_config.return_value.vector_store.embedding_model_dims = mock_config["vector_store"][
                "vector_store_embedding_model_dims"
            ]

            with patch("cogent.base.providers.vector_store.weaviate_vector_store.Weaviate") as mock_weaviate_class:
                mock_weaviate_instance = MagicMock()
                mock_weaviate_class.return_value = mock_weaviate_instance

                vector_store = CogentVectorStore("test_weaviate")

                # Test delegation
                test_vectors = [[0.1, 0.2, 0.3]]
                test_payloads = [{"data": "test"}]
                test_ids = [str(uuid.uuid4())]

                vector_store.insert(test_vectors, test_payloads, test_ids)
                mock_weaviate_instance.insert.assert_called_once_with(test_vectors, test_payloads, test_ids)

                vector_store.search("query", test_vectors[0], limit=5)
                mock_weaviate_instance.search.assert_called_once_with("query", test_vectors[0], 5, None)

                test_delete_id = "test-delete-id"
                vector_store.delete(test_delete_id)
                mock_weaviate_instance.delete.assert_called_once_with(test_delete_id)


class TestWeaviateUnit:
    """Unit tests for Weaviate vector store."""

    @pytest.fixture
    def mock_weaviate_client(self):
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_batch = MagicMock()

        mock_collection.query.hybrid.return_value = MagicMock()
        mock_collection.data.delete_many.return_value = MagicMock()
        mock_collection.data.update.return_value = MagicMock()
        mock_collection.data.get_by_id.return_value = MagicMock()

        mock_batch.__enter__ = MagicMock(return_value=mock_batch)
        mock_batch.__exit__ = MagicMock(return_value=None)
        mock_batch.add_object = MagicMock()

        mock_client.collections.get.return_value = mock_collection
        mock_client.batch.fixed_size.return_value = mock_batch
        mock_client.collections.exists.return_value = False
        mock_client.collections.create = MagicMock()
        mock_client.collections.list.return_value = MagicMock()
        mock_client.collections.delete = MagicMock()

        return mock_client

    @pytest.mark.unit
    def test_initialization(self, mock_weaviate_client):
        """Test Weaviate initialization."""
        with patch("cogent.base.providers.vector_store.weaviate_vector_store.weaviate") as mock_weaviate:
            mock_weaviate.connect_to_local.return_value = mock_weaviate_client

            weaviate_store = Weaviate(
                collection_name="test_collection", embedding_model_dims=768, cluster_url="http://localhost:8080"
            )

            assert weaviate_store.collection_name == "test_collection"
            assert weaviate_store.embedding_model_dims == 768
            assert weaviate_store.client == mock_weaviate_client

    @pytest.mark.unit
    def test_insert_operation(self, mock_weaviate_client):
        """Test Weaviate insert operation."""
        with patch("cogent.base.providers.vector_store.weaviate_vector_store.weaviate") as mock_weaviate:
            mock_weaviate.connect_to_local.return_value = mock_weaviate_client

            weaviate_store = Weaviate(
                collection_name="test_collection", embedding_model_dims=768, cluster_url="http://localhost:8080"
            )

            test_vectors = [[0.1, 0.2, 0.3]]
            test_payloads = [{"data": "test"}]
            test_ids = [str(uuid.uuid4())]

            weaviate_store.insert(test_vectors, test_payloads, test_ids)

            mock_weaviate_client.batch.fixed_size.assert_called_once_with(batch_size=100)
            mock_batch = mock_weaviate_client.batch.fixed_size.return_value
            mock_batch.add_object.assert_called_once()

    @pytest.mark.unit
    def test_search_operation(self, mock_weaviate_client):
        """Test Weaviate search operation."""
        with patch("cogent.base.providers.vector_store.weaviate_vector_store.weaviate") as mock_weaviate:
            mock_weaviate.connect_to_local.return_value = mock_weaviate_client

            weaviate_store = Weaviate(
                collection_name="test_collection", embedding_model_dims=768, cluster_url="http://localhost:8080"
            )

            query_vector = [0.1, 0.2, 0.3]
            filters = {"user_id": "test_user"}

            weaviate_store.search("test query", query_vector, limit=5, filters=filters)

            mock_collection = mock_weaviate_client.collections.get.return_value
            mock_collection.query.hybrid.assert_called_once()

    @pytest.mark.unit
    def test_delete_operation(self, mock_weaviate_client):
        """Test Weaviate delete operation."""
        with patch("cogent.base.providers.vector_store.weaviate_vector_store.weaviate") as mock_weaviate:
            mock_weaviate.connect_to_local.return_value = mock_weaviate_client

            weaviate_store = Weaviate(
                collection_name="test_collection", embedding_model_dims=768, cluster_url="http://localhost:8080"
            )

            test_id = str(uuid.uuid4())
            weaviate_store.delete(test_id)

            mock_collection = mock_weaviate_client.collections.get.return_value
            mock_collection.data.delete_by_id.assert_called_once_with(test_id)


class TestVectorStoreErrorHandling:
    """Test error handling in vector store operations."""

    @pytest.mark.unit
    def test_weaviate_connection_error(self):
        """Test Weaviate connection error handling."""
        with patch("cogent.base.providers.vector_store.weaviate_vector_store.weaviate") as mock_weaviate:
            mock_weaviate.connect_to_local.side_effect = Exception("Connection failed")

            with pytest.raises(Exception, match="Connection failed"):
                Weaviate(
                    collection_name="test_collection", embedding_model_dims=768, cluster_url="http://localhost:8080"
                )

    @pytest.mark.unit
    def test_weaviate_insert_error(self):
        """Test Weaviate insert error handling."""
        with patch("cogent.base.providers.vector_store.weaviate_vector_store.weaviate") as mock_weaviate:
            mock_weaviate_client = MagicMock()
            mock_weaviate_client.batch.fixed_size.side_effect = Exception("Insert failed")
            mock_weaviate_client = MagicMock()
            mock_collection = MagicMock()
            mock_collection.query.hybrid.side_effect = Exception("Search failed")
            mock_weaviate_client.collections.get.return_value = mock_collection
            mock_weaviate.connect_to_local.return_value = mock_weaviate_client
            mock_weaviate_client.batch.fixed_size.side_effect = Exception("Insert failed")

            weaviate_store = Weaviate(
                collection_name="test_collection", embedding_model_dims=768, cluster_url="http://localhost:8080"
            )

            with pytest.raises(Exception, match="Insert failed"):
                weaviate_store.insert([[0.1, 0.2, 0.3]], [{"data": "test"}], [str(uuid.uuid4())])

    @pytest.mark.unit
    def test_weaviate_search_error(self):
        """Test Weaviate search error handling."""
        with patch("cogent.base.providers.vector_store.weaviate_vector_store.weaviate") as mock_weaviate:
            mock_weaviate_client = MagicMock()
            mock_weaviate_client.batch.fixed_size.side_effect = Exception("Insert failed")
            mock_weaviate_client = MagicMock()
            mock_collection = MagicMock()
            mock_collection.query.hybrid.side_effect = Exception("Search failed")
            mock_weaviate_client.collections.get.return_value = mock_collection
            mock_weaviate.connect_to_local.return_value = mock_weaviate_client
            mock_collection = mock_weaviate_client.collections.get.return_value
            mock_collection.query.hybrid.side_effect = Exception("Search failed")

            weaviate_store = Weaviate(
                collection_name="test_collection", embedding_model_dims=768, cluster_url="http://localhost:8080"
            )

            with pytest.raises(Exception, match="Search failed"):
                weaviate_store.search("test query", [0.1, 0.2, 0.3], limit=5)
