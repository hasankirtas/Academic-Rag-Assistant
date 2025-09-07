# test_embedder.py
import pytest
from src.implementations.embedding.huggingface_embedder import HuggingFaceEmbedder, HuggingFaceEmbedderError

# Use a small, fast model for testing
TEST_MODEL = "sshleifer/tiny-distilbert-base-cased"

# Example texts for testing
sample_documents = [
    "Kapitel 1: Einführung in die Volkswirtschaftslehre",
    "Die Volkswirtschaftslehre beschäftigt sich mit der Analyse wirtschaftlicher Zusammenhänge.",
    "1.1 Grundlegende Konzepte",
    "Das Bruttoinlandsprodukt (BIP) ist ein wichtiger Indikator für die wirtschaftliche Leistung eines Landes."
]

sample_queries = [
    "Was versteht man unter Inflation?",
    "Erklären Sie die Bedeutung des Bruttoinlandsprodukts."
]

def test_embedder_initialization():
    embedder = HuggingFaceEmbedder(model_name=TEST_MODEL, batch_size=2)
    assert embedder.device in ["cpu", "cuda"], "Device not set correctly"
    assert embedder.get_embedding_dimension() > 0, "Embedding dimension should be positive"
    assert embedder.model_name == TEST_MODEL, "Model name should be set correctly"

def test_batch_embedding():
    embedder = HuggingFaceEmbedder(model_name=TEST_MODEL, batch_size=2)
    embeddings = embedder.embed_documents(sample_documents)
    assert len(embeddings) == len(sample_documents), "Batch embedding length mismatch"
    for emb in embeddings:
        assert isinstance(emb, list), "Each embedding should be a list of floats"
        assert len(emb) == embedder.get_embedding_dimension(), "Embedding dimension mismatch"

def test_single_query_embedding():
    embedder = HuggingFaceEmbedder(model_name=TEST_MODEL)
    for query in sample_queries:
        q_emb = embedder.embed_query(query)
        assert isinstance(q_emb, list), "Query embedding should be a list"
        assert len(q_emb) == embedder.get_embedding_dimension(), "Query embedding dimension mismatch"

def test_get_model_info():
    embedder = HuggingFaceEmbedder(model_name=TEST_MODEL)
    info = embedder.get_model_info()
    required_keys = [
        "model_name", "provider", "device", 
        "embedding_dimension", "max_length", 
        "batch_size", "normalize_embeddings"
    ]
    for key in required_keys:
        assert key in info, f"Model info missing key: {key}"

def test_empty_documents():
    embedder = HuggingFaceEmbedder(model_name=TEST_MODEL)
    embeddings = embedder.embed_documents([])
    assert embeddings == [], "Empty document list should return empty embeddings"

def test_empty_query_raises_error():
    embedder = HuggingFaceEmbedder(model_name=TEST_MODEL)
    with pytest.raises(ValueError):
        embedder.embed_query("   ")

def test_invalid_model_name():
    with pytest.raises(HuggingFaceEmbedderError):
        HuggingFaceEmbedder(model_name="non_existent_model_12345")

if __name__ == "__main__":
    pytest.main([__file__])