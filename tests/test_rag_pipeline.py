# tests/test_rag_pipeline.py

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Optional

from src.rag_pipeline import (
    RAGPipeline, 
    create_rag_pipeline, 
    create_hybrid_retriever
)


class TestRAGPipeline:
    """Test cases for RAG pipeline class."""

    @pytest.fixture
    def mock_embedding_service(self):
        """Mock embedding Service fixture."""
        mock = Mock()
        mock.get_model_info.return_value = {"model": "test-model", "dim": 384}
        return mock
    
    @pytest.fixture
    def mock_vector_db_service(self):
        """Mock vector database service fixture."""
        mock = Mock()
        mock.get_collection_info.return_value = {"collection": "test-collection", "count": 100}
        return mock

    @pytest.fixture
    def mock_retriever(self):
        """Mock hybrid retriever fixture."""
        mock = Mock()
        mock.vector_weight = 0.7
        mock.keyword_weight = 0.3
        mock.get_relevant_contexts.return_value = [
            {"context": "Test context 1", "score": 0.95, "metadata": {"id": "doc1"}},
            {"context": "Test context 2", "score": 0.87, "metadata": {"id": "doc2"}}
        ]
        mock.update_weights.return_value = None
        return mock

    @pytest.fixture
    def rag_pipeline(self, mock_embedding_service,  mock_vector_db_service, mock_retriever):
        """RAG Pipeline fixture with mocked dependencies."""
        with patch('src.rag_pipeline.QueryEmbeddingService') as mock_emb_cls, \
             patch('src.rag_pipeline.VectorDatabaseService') as mock_vdb_cls, \
             patch('src.rag_pipeline.Retriever') as mock_ret_cls:
            
            mock_emb_cls.return_value = mock_embedding_service
            mock_vdb_cls.return_value = mock_vector_db_service
            mock_ret_cls.return_value = mock_retriever
            
            pipeline = RAGPipeline()
            return pipeline

    def test_initialization_success(self, rag_pipeline):
        """Test successful RAG pipeline initialization."""
        assert rag_pipeline.embedding_service is not None
        assert rag_pipeline.vector_db_service is not None
        assert rag_pipeline.retriever is not None

    def test_initialization_failure(self):
        """Test RAG pipeline initialization failure."""
        with patch('src.rag_pipeline.QueryEmbeddingService', 
                   side_effect=Exception("Embedding service failed")):
            with pytest.raises(Exception, match="Embedding service failed"):
                RAGPipeline()

    def test_query_success(self, rag_pipeline):
        """Test successful query execution."""
        query_text = "Test query"
        results = rag_pipeline.query(query_text, k=2)
        
        assert len(results) == 2
        assert results[0]["context"] == "Test context 1"
        assert results[0]["score"] == 0.95
        assert results[1]["context"] == "Test context 2"
        
        rag_pipeline.retriever.get_relevant_contexts.assert_called_once_with(
            query_text=query_text,
            query_embedding=None,
            k=2
        )

    def test_query_with_default_k(self, rag_pipeline):
        """Test query with default k parameter."""
        query_text = "Test query"
        rag_pipeline.query(query_text)
        
        rag_pipeline.retriever.get_relevant_contexts.assert_called_once_with(
            query_text=query_text,
            query_embedding=None,
            k=5
        )

    def test_query_failure(self, rag_pipeline):
        """Test query execution failure."""
        rag_pipeline.retriever.get_relevant_contexts.side_effect = Exception("Retrieval failed")
        
        results = rag_pipeline.query("test query")
        assert results == []

    def test_query_with_custom_embedding_success(self, rag_pipeline):
        """Test successful query with custom embedding."""
        query_text = "Test query"
        custom_embedding = [0.1, 0.2, 0.3]
        
        results = rag_pipeline.query_with_custom_embedding(query_text, custom_embedding, k=3)
        
        assert len(results) == 2  # Based on mock return value
        rag_pipeline.retriever.get_relevant_contexts.assert_called_once_with(
            query_text=query_text,
            query_embedding=custom_embedding,
            k=3
        )

    def test_query_with_custom_embedding_failure(self, rag_pipeline):
        """Test query with custom embedding failure."""
        rag_pipeline.retriever.get_relevant_contexts.side_effect = Exception("Custom embedding failed")
        
        results = rag_pipeline.query_with_custom_embedding("test", [0.1, 0.2])
        assert results == []

    def test_get_pipeline_info_success(self, rag_pipeline):
        """Test getting pipeline information successfully."""
        info = rag_pipeline.get_pipeline_info()
        
        assert "embedding_service" in info
        assert "retriever" in info
        assert "vector_database" in info
        
        assert info["embedding_service"]["model"] == "test-model"
        assert info["retriever"]["type"] == "Retriever"
        assert info["retriever"]["vector_weight"] == 0.7
        assert info["retriever"]["keyword_weight"] == 0.3
    
    def test_get_pipeline_info_without_collection_info(self, rag_pipeline):
        """Test getting pipeline info when vector db doesn't have get_collection_info."""
        del rag_pipeline.vector_db_service.get_collection_info
        
        info = rag_pipeline.get_pipeline_info()
        assert info["vector_database"] == "Available"

    def test_get_pipeline_info_failure(self, rag_pipeline):
        """Test getting pipeline info failure."""
        rag_pipeline.embedding_service.get_model_info.side_effect = Exception("Info failed")
        
        info = rag_pipeline.get_pipeline_info()
        assert info == {}

    def test_update_retriever_weights_success(self, rag_pipeline):
        """Test successful retriever weights update."""
        result = rag_pipeline.update_retriever_weights(0.8, 0.2)
        
        assert result is True
        rag_pipeline.retriever.update_weights.assert_called_once_with(0.8, 0.2)

    def test_update_retriever_weights_failure(self, rag_pipeline):
        """Test retriever weights update failure."""
        rag_pipeline.retriever.update_weights.side_effect = Exception("Update failed")
        
        result = rag_pipeline.update_retriever_weights(0.8, 0.2)
        assert result is False

    def test_query_logging(self, rag_pipeline, caplog):
        """Test that query operations are properly logged."""
        module_logger = logging.getLogger("src.rag_pipeline")
        old_propagate = module_logger.propagate
        try:
            with caplog.at_level("INFO"):
                module_logger.propagate = True
                rag_pipeline.query("test query", k=2)
        finally:
            module_logger.propagate = old_propagate

        assert "Processing query: test query" in caplog.text
        assert "Retrieved 2 contexts" in caplog.text

class TestFactoryFunctions:
    """Test cases for factory functions."""

    @patch('src.rag_pipeline.RAGPipeline')
    def test_create_rag_pipeline_success(self, mock_pipeline_cls):
        """Test successful RAG pipeline creation via factory."""
        mock_instance = Mock()
        mock_pipeline_cls.return_value = mock_instance

        config = {"test": "config"}
        result = create_rag_pipeline(config)

        assert result == mock_instance
        mock_pipeline_cls.assert_called_once_with(config=config)

    @patch('src.rag_pipeline.RAGPipeline')
    def test_create_rag_pipeline_failure(self, mock_pipeline_cls):
        """Test RAG pipeline creation failure via factory."""
        mock_pipeline_cls.side_effect = Exception("Creation failed")
        
        with pytest.raises(Exception, match="Creation failed"):
            create_rag_pipeline()

    @patch('src.rag_pipeline.Retriever')
    @patch('src.rag_pipeline.VectorDatabaseService')
    @patch('src.rag_pipeline.QueryEmbeddingService')
    def test_create_hybrid_retriever_success(self, mock_emb_cls, mock_vdb_cls, mock_ret_cls):
        """Test successful hybrid retriever creation via factory."""
        mock_embedding = Mock()
        mock_vector_db = Mock()
        mock_retriever = Mock()
        
        mock_emb_cls.return_value = mock_embedding
        mock_vdb_cls.return_value = mock_vector_db
        mock_ret_cls.return_value = mock_retriever
        
        config = {"test": "config"}
        result = create_hybrid_retriever(config)
        
        assert result == mock_retriever
        mock_ret_cls.assert_called_once_with(
            vector_db_service=mock_vector_db,
            query_embedding_service=mock_embedding,
            config=config
        )

    @patch('src.rag_pipeline.QueryEmbeddingService')
    def test_create_hybrid_retriever_failure(self, mock_emb_cls):
        """Test hybrid retriever creation failure via factory."""
        mock_emb_cls.side_effect = Exception("Service creation failed")
        
        with pytest.raises(Exception, match="Service creation failed"):
            create_hybrid_retriever()


class TestIntegration:
    """Integration test cases."""

    @patch('src.rag_pipeline.QueryEmbeddingService')
    @patch('src.rag_pipeline.VectorDatabaseService') 
    @patch('src.rag_pipeline.Retriever')
    def test_end_to_end_workflow(self, mock_ret_cls, mock_vdb_cls, mock_emb_cls):
        """Test end-to-end RAG pipeline workflow."""
        # Setup mocks
        mock_embedding = Mock()
        mock_vector_db = Mock()
        mock_retriever = Mock()
        
        mock_emb_cls.return_value = mock_embedding
        mock_vdb_cls.return_value = mock_vector_db
        mock_ret_cls.return_value = mock_retriever
        
        mock_embedding.get_model_info.return_value = {"model": "test"}
        mock_retriever.vector_weight = 0.7
        mock_retriever.keyword_weight = 0.3
        mock_retriever.get_relevant_contexts.return_value = [
            {"context": "Relevant context", "score": 0.9}
        ]
        
        # Create pipeline and run workflow
        pipeline = RAGPipeline()
        
        # Test pipeline info
        info = pipeline.get_pipeline_info()
        assert "embedding_service" in info
        
        # Test query
        results = pipeline.query("test query")
        assert len(results) == 1
        
        # Test weight update
        success = pipeline.update_retriever_weights(0.8, 0.2)
        assert success is True


# Test configuration for pytest
@pytest.fixture(scope="session")
def test_config():
    """Test configuration fixture."""
    return {
        "vector_weight": 0.7,
        "keyword_weight": 0.3,
        "k_default": 5
    }


# Parametrized tests
@pytest.mark.parametrize("k_value,expected_calls", [
    (1, 1),
    (5, 1), 
    (10, 1)
])
def test_query_with_different_k_values(k_value, expected_calls):
    """Test query method with different k values."""
    with patch('src.rag_pipeline.QueryEmbeddingService'), \
         patch('src.rag_pipeline.VectorDatabaseService'), \
         patch('src.rag_pipeline.Retriever') as mock_ret_cls:
        
        mock_retriever = Mock()
        mock_retriever.get_relevant_contexts.return_value = []
        mock_ret_cls.return_value = mock_retriever
        
        pipeline = RAGPipeline()
        pipeline.query("test", k=k_value)
        
        assert mock_retriever.get_relevant_contexts.call_count == expected_calls


@pytest.mark.parametrize("vector_w,keyword_w", [
    (0.5, 0.5),
    (0.8, 0.2),
    (0.3, 0.7),
    (1.0, 0.0)
])
def test_weight_updates(vector_w, keyword_w):
    """Test weight updates with different combinations."""
    with patch('src.rag_pipeline.QueryEmbeddingService'), \
         patch('src.rag_pipeline.VectorDatabaseService'), \
         patch('src.rag_pipeline.Retriever') as mock_ret_cls:
        
        mock_retriever = Mock()
        mock_ret_cls.return_value = mock_retriever
        
        pipeline = RAGPipeline()
        result = pipeline.update_retriever_weights(vector_w, keyword_w)
        
        assert result is True
        mock_retriever.update_weights.assert_called_once_with(vector_w, keyword_w)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])