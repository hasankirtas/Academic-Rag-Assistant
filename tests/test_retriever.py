# test_hybrid_retriever.py
import pytest
from unittest.mock import MagicMock
from src.rag.retriever import Retriever, create_hybrid_retriever
from src.implementations.embedding.query_embedding import QueryEmbeddingService

# Sample data
sample_documents = [
    {"text": "Das Bruttoinlandsprodukt (BIP) ist ein Indikator."},
    {"text": "Inflation ist ein Anstieg des Preisniveaus."},
    {"text": "Volkswirtschaftslehre analysiert wirtschaftliche Zusammenhänge."}
]

sample_queries = [
    "Was versteht man unter Inflation?",
    "Erklären Sie das BIP."
]

@pytest.fixture
def mock_vector_db():
    mock_db = MagicMock()
    # Mock search_by_embedding to return sample documents with similarity scores
    mock_results = []
    for doc in sample_documents:
        result = MagicMock()
        result.metadata = doc
        result.similarity = 0.8
        mock_results.append(result)
    mock_db.search_by_embedding.return_value = mock_results
    return mock_db

@pytest.fixture
def mock_embedding_service():
    mock_service = MagicMock(spec=QueryEmbeddingService)
    # Mock embed_text to return fixed-length vectors
    mock_service.embed_text.return_value = [0.1] * 768
    return mock_service

def test_hybrid_retriever_initialization(mock_vector_db, mock_embedding_service):
    retriever = Retriever(mock_vector_db, mock_embedding_service)
    assert retriever.vector_weight == 0.85
    assert retriever.keyword_weight == 0.15
    assert retriever.query_embedding_service is not None

def test_get_relevant_contexts(mock_vector_db, mock_embedding_service):
    retriever = Retriever(mock_vector_db, mock_embedding_service)
    query_text = "Was ist das BIP?"
    contexts = retriever.get_relevant_contexts(query_text=query_text, k=2)
    assert len(contexts) == 2
    for context in contexts:
        assert "hybrid_score" in context
        assert "vector_score" in context
        assert "keyword_score" in context
        assert "text" in context

def test_update_weights(mock_vector_db, mock_embedding_service):
    retriever = Retriever(mock_vector_db, mock_embedding_service)
    retriever.update_weights(vector_weight=0.5, keyword_weight=0.5)
    assert retriever.vector_weight == 0.5
    assert retriever.keyword_weight == 0.5

def test_get_vector_results_only(mock_vector_db, mock_embedding_service):
    retriever = Retriever(mock_vector_db, mock_embedding_service)
    vector_results = retriever.get_vector_results_only([0.1]*768, k=3)
    assert len(vector_results) == 3
    for result in vector_results:
        assert "vector_score" in result
        assert "retrieval_method" in result

@pytest.fixture(scope="module")
def retriever_for_edge_cases():
    embedding_service = MagicMock(spec=QueryEmbeddingService)
    embedding_service.embed_text.return_value = [0.1] * 768
    vector_db = MagicMock()
    vector_db.search_by_embedding.return_value = []
    return Retriever(vector_db, embedding_service)

def test_empty_query(retriever_for_edge_cases):
    results = retriever_for_edge_cases.get_relevant_contexts(query_text="", k=5)
    assert results == []

def test_short_query(retriever_for_edge_cases):
    results = retriever_for_edge_cases.get_relevant_contexts(query_text="BIP", k=3)
    assert isinstance(results, list)
    assert all(isinstance(r, dict) for r in results)

def test_update_weights_edge_cases(retriever_for_edge_cases):
    retriever_for_edge_cases.update_weights(vector_weight=0, keyword_weight=0)
    assert 0 < retriever_for_edge_cases.vector_weight <= 1
    assert 0 <= retriever_for_edge_cases.keyword_weight <= 1

    retriever_for_edge_cases.update_weights(vector_weight=10, keyword_weight=0)
    assert retriever_for_edge_cases.vector_weight == 1.0
    assert retriever_for_edge_cases.keyword_weight == 0.0

    retriever_for_edge_cases.update_weights(vector_weight=3, keyword_weight=7)
    assert abs(retriever_for_edge_cases.vector_weight - 0.3) < 0.01
    assert abs(retriever_for_edge_cases.keyword_weight - 0.7) < 0.01