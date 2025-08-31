"""
Test file for the new vector database architecture.
Tests core functionality of Strategy Pattern, Factory Pattern, Module, and Service layers.
"""

import os
import random
import shutil
import pytest
from typing import List, Dict, Any

from src.core.factories.vector_repository_factory import VectorRepositoryFactory, VectorDBType
from src.core.vector_database_module import VectorDatabaseModule, EmbeddingData, SearchResult
from src.implementations.vector_db.vector_database_service import VectorDatabaseService, DocumentChunk
from src.implementations.vector_db.chroma_repository import ChromaRepository


# ---------------------------
# Fixtures
# ---------------------------

@pytest.fixture
def persist_dir(tmp_path_factory):
    """Create a temporary directory for testing."""
    dir_path = tmp_path_factory.mktemp("chroma_test")
    yield str(dir_path)
    shutil.rmtree(dir_path, ignore_errors=True)


@pytest.fixture
def factory():
    """Create VectorRepositoryFactory instance."""
    return VectorRepositoryFactory()


@pytest.fixture
def module(persist_dir):
    """Create VectorDatabaseModule instance."""
    return VectorDatabaseModule(
        VectorDBType.CHROMA,
        persist_directory=persist_dir,
        collection_name="test_module_collection"
    )


@pytest.fixture
def service(persist_dir):
    """Create VectorDatabaseService instance."""
    return VectorDatabaseService(
        VectorDBType.CHROMA,
        persist_directory=persist_dir,
        collection_name="test_service_collection"
    )


@pytest.fixture
def dummy_embeddings():
    """Create dummy embedding data for testing."""
    random.seed(42)
    dim = 8
    texts = [
        "Makroökonomie: Einführung in das Bruttoinlandsprodukt",
        "Inflation und Geldpolitik: zentrale Begriffe",
        "Arbeitslosigkeit: Typen und natürliche Arbeitslosenquote"
    ]
    embeddings = [[random.random() for _ in range(dim)] for _ in texts]
    metadatas = [
        {"text": t, "content": t, "page": i+1, "section": f"Section {i}"}
        for i, t in enumerate(texts)
    ]
    return [EmbeddingData(embedding=e, metadata=m) for e, m in zip(embeddings, metadatas)]


@pytest.fixture
def document_chunks():
    """Create document chunks for testing."""
    return [
        DocumentChunk(
            text="Makroökonomie: Einführung in das Bruttoinlandsprodukt",
            page=1,
            section="Einführung"
        ),
        DocumentChunk(
            text="Inflation und Geldpolitik: zentrale Begriffe",
            page=2,
            section="Geldpolitik"
        ),
        DocumentChunk(
            text="Arbeitslosigkeit: Typen und natürliche Quote",
            page=3,
            section="Arbeitslosigkeit"
        )
    ]


@pytest.fixture
def chunk_embeddings():
    """Create embeddings for document chunks."""
    random.seed(42)
    dim = 8
    return [[random.random() for _ in range(dim)] for _ in range(3)]


# ---------------------------
# Factory Pattern Tests
# ---------------------------

def test_factory_supported_types(factory):
    """Test factory supported types."""
    supported_types = factory.get_supported_types()
    assert isinstance(supported_types, list)
    assert "chroma" in supported_types


def test_factory_create_chroma_repository(factory, persist_dir):
    """Test factory creating ChromaDB repository."""
    repo = factory.create_repository(
        VectorDBType.CHROMA,
        persist_directory=persist_dir,
        collection_name="test_collection"
    )
    assert isinstance(repo, ChromaRepository)
    assert repo.collection_name == "test_collection"


def test_factory_create_unsupported_repository(factory):
    """Test factory with unsupported database type."""
    with pytest.raises(NotImplementedError):
        factory.create_repository(
            VectorDBType.PINECONE,
            api_key="test"
        )


# ---------------------------
# Module Layer Tests
# ---------------------------

def test_module_initialization(module):
    """Test module initialization."""
    assert module is not None
    assert hasattr(module, 'repository')
    assert hasattr(module, 'factory')


def test_module_store_embeddings(module, dummy_embeddings):
    """Test storing embeddings in module."""
    success = module.store_embeddings(dummy_embeddings, batch_size=2)
    assert success is True


def test_module_search_similar(module, dummy_embeddings):
    """Test searching similar embeddings in module."""
    module.store_embeddings(dummy_embeddings)
    query_embedding = dummy_embeddings[0].embedding
    results = module.search_similar(query_embedding, top_k=3)
    assert len(results) > 0
    assert all(isinstance(r, SearchResult) for r in results)


def test_module_update_embedding(module, dummy_embeddings):
    """Test updating embedding in module."""
    module.store_embeddings(dummy_embeddings)
    query_embedding = dummy_embeddings[0].embedding
    results = module.search_similar(query_embedding, top_k=1)
    assert len(results) > 0
    
    first_id = results[0].id
    new_embedding = [x + 0.1 for x in dummy_embeddings[0].embedding]
    success = module.update_embedding(
        embedding_id=first_id,
        new_embedding=new_embedding,
        new_metadata={"text": "Updated document"}
    )
    assert success is True


def test_module_delete_embedding(module, dummy_embeddings):
    """Test deleting embedding from module."""
    module.store_embeddings(dummy_embeddings)
    query_embedding = dummy_embeddings[0].embedding
    results = module.search_similar(query_embedding, top_k=1)
    assert len(results) > 0
    
    first_id = results[0].id
    success = module.delete_embedding(first_id)
    assert success is True


def test_module_get_database_stats(module):
    """Test getting database statistics from module."""
    stats = module.get_database_stats()
    assert isinstance(stats, dict)
    assert "document_count" in stats


def test_module_clear_database(module, dummy_embeddings):
    """Test clearing database in module."""
    module.store_embeddings(dummy_embeddings)
    success = module.clear_database()
    assert success is True
    
    stats = module.get_database_stats()
    assert stats["document_count"] == 0


# ---------------------------
# Service Layer Tests
# ---------------------------

def test_service_initialization(service):
    """Test service initialization."""
    assert service is not None
    assert hasattr(service, 'module')


def test_service_store_document_chunks(service, document_chunks, chunk_embeddings):
    """Test storing document chunks in service."""
    success = service.store_document_chunks(document_chunks, chunk_embeddings)
    assert success is True


def test_service_search_by_embedding(service, document_chunks, chunk_embeddings):
    """Test searching by embedding in service."""
    service.store_document_chunks(document_chunks, chunk_embeddings)
    query_embedding = chunk_embeddings[0]
    results = service.search_by_embedding(query_embedding, top_k=3)
    assert len(results) > 0
    assert all(isinstance(r, SearchResult) for r in results)


def test_service_update_document_chunk(service, document_chunks, chunk_embeddings):
    """Test updating document chunk in service."""
    service.store_document_chunks(document_chunks, chunk_embeddings)
    query_embedding = chunk_embeddings[0]
    results = service.search_by_embedding(query_embedding, top_k=1)
    assert len(results) > 0
    
    first_id = results[0].id
    new_text = "Updated document content"
    new_embedding = [x + 0.1 for x in chunk_embeddings[0]]
    success = service.update_document_chunk(
        chunk_id=first_id,
        new_text=new_text,
        new_embedding=new_embedding,
        new_metadata={"updated": True}
    )
    assert success is True


def test_service_delete_document_chunk(service, document_chunks, chunk_embeddings):
    """Test deleting document chunk in service."""
    service.store_document_chunks(document_chunks, chunk_embeddings)
    query_embedding = chunk_embeddings[0]
    results = service.search_by_embedding(query_embedding, top_k=1)
    assert len(results) > 0
    
    first_id = results[0].id
    success = service.delete_document_chunk(first_id)
    assert success is True


def test_service_get_database_statistics(service):
    """Test getting database statistics from service."""
    stats = service.get_database_statistics()
    assert isinstance(stats, dict)
    assert "document_count" in stats
    assert "total_chunks" in stats


def test_service_clear_all_data(service, document_chunks, chunk_embeddings):
    """Test clearing all data in service."""
    service.store_document_chunks(document_chunks, chunk_embeddings)
    success = service.clear_all_data()
    assert success is True
    
    stats = service.get_database_statistics()
    assert stats["document_count"] == 0


# ---------------------------
# Integration Tests
# ---------------------------

def test_full_workflow_integration(service, document_chunks, chunk_embeddings):
    """Test complete workflow integration."""
    # 1. Store document chunks
    success = service.store_document_chunks(document_chunks, chunk_embeddings)
    assert success is True
    
    # 2. Search by embedding
    query_embedding = chunk_embeddings[0]
    results = service.search_by_embedding(query_embedding, top_k=3)
    assert len(results) > 0
    
    # 3. Update a result
    if results:
        first_id = results[0].id
        new_text = "Updated content"
        new_embedding = [x + 0.1 for x in query_embedding]
        update_success = service.update_document_chunk(
            chunk_id=first_id,
            new_text=new_text,
            new_embedding=new_embedding
        )
        assert update_success is True
    
    # 4. Get statistics
    stats = service.get_database_statistics()
    assert stats["document_count"] > 0
    
    # 5. Clear database
    clear_success = service.clear_all_data()
    assert clear_success is True
