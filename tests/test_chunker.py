# test_document_chunker.py
import pytest
from src.preprocessing.chunker import DocumentChunker

# Sample structured data
sample_structured_data = [
    {'text': 'Kapitel 1: Einführung in die Volkswirtschaftslehre', 'page_number': 1, 'font_size': 18.0, 'is_bold': True, 'element_type': 'header'},
    {'text': 'Die Volkswirtschaftslehre untersucht wirtschaftliche Prozesse.', 'page_number': 1, 'font_size': 12.0, 'is_bold': False, 'element_type': 'paragraph'},
    {'text': '1.1 Grundlegende Konzepte', 'page_number': 1, 'font_size': 16.0, 'is_bold': True, 'element_type': 'subheader'},
    {'text': 'Das Bruttoinlandsprodukt (BIP) ist ein zentraler Indikator.', 'page_number': 2, 'font_size': 12.0, 'is_bold': False, 'element_type': 'paragraph'},
    {'text': '1.2 Wirtschaftskreislauf', 'page_number': 2, 'font_size': 16.0, 'is_bold': True, 'element_type': 'subheader'},
    {'text': 'Der Wirtschaftskreislauf zeigt, wie Geld und Güter zirkulieren.', 'page_number': 2, 'font_size': 12.0, 'is_bold': False, 'element_type': 'paragraph'},
]

def test_default_chunking():
    chunker = DocumentChunker()
    chunks = chunker.chunk_text(sample_structured_data)
    
    assert len(chunks) > 0, "No chunks were created"
    for chunk in chunks:
        assert 'content' in chunk, "Chunk missing 'content'"
        assert 'metadata' in chunk, "Chunk missing 'metadata'"
        assert len(chunk['content']) >= 10, "Chunk content too short"
    
    stats = chunker.get_chunking_stats()
    assert stats['total_chunks_created'] == len(chunks), "Stats mismatch"
    assert stats['strategy_used'] == "ChainedChunker", "Default strategy should be ChainedChunker"

def test_empty_input():
    chunker = DocumentChunker()
    chunks = chunker.chunk_text([])
    assert chunks == [], "Empty input should return empty list"

def test_invalid_input():
    chunker = DocumentChunker()
    with pytest.raises(ValueError):
        chunker.chunk_text("invalid_input")

def test_missing_required_field():
    chunker = DocumentChunker()
    invalid_data = [{'page_number': 1}]  # Missing 'text'
    with pytest.raises(Exception, match="Element 0 missing required field 'text'"):
        chunker.chunk_text(invalid_data)

def test_batch_chunking():
    chunker = DocumentChunker()
    documents = [sample_structured_data, sample_structured_data]
    batch_chunks = chunker.chunk_batch(documents)
    
    assert isinstance(batch_chunks, list), "Batch chunking should return a list"
    assert len(batch_chunks) == 2, "Batch chunking did not process all documents"
    for doc_chunks in batch_chunks:
        assert all('content' in c and 'metadata' in c for c in doc_chunks), "Chunk missing content or metadata in batch"

def test_preview_functionality():
    chunker = DocumentChunker()
    preview = chunker.preview_chunking(sample_structured_data, max_preview_length=50)
    
    assert 'preview_chunks' in preview, "Preview missing 'preview_chunks'"
    assert len(preview['preview_chunks']) > 0, "Preview produced no chunks"
    for p in preview['preview_chunks']:
        assert len(p['content_preview']) <= 53, "Preview content too long (includes ellipsis)"
        assert 'metadata_summary' in p, "Preview chunk missing 'metadata_summary'"

if __name__ == "__main__":
    pytest.main([__file__])