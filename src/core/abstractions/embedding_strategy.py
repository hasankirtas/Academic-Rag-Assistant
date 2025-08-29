"""
Embedding Strategy Abstract Base Class

This module defines a common interface for different embedding providers.
It forms the basis for the strategy pattern.
"""
from abc import ABC, abstractmethod
from typing import List

class EmbeddingStrategy(ABC):
    """
    Abstract base class for embedding strategies.
    
    Concrete classes that can implement this interface:
    - HuggingFaceEmbedder
    - OpenAIEmbedder
    - etc.
    """

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Converts a list of documents into embedding vectors.
        
        Args:
            texts: List of texts to convert into embeddings
        
        Returns:
            A list of embedding vectors, one per text
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        Converts a single query text into an embedding vector.
        
        Args:
            text: Query text to convert into embedding
        
        Returns:
            The embedding vector for the query
        """
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        Converts a single query text into an embedding vector.
        
        Args:
            text: Query text to convert into embedding
        
        Returns:
            The embedding vector for the query
        """
        pass