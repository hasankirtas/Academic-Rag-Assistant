"""
Core module for the Academic RAG Assistant.

This module contains the core abstractions, factories, and main components
of the RAG system architecture.
"""

from .vector_database_module import VectorDatabaseModule, EmbeddingData, SearchResult

__all__ = [
    "VectorDatabaseModule",
    "EmbeddingData", 
    "SearchResult"
]
