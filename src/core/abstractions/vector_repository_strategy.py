"""
Vector Repository Strategy Pattern.
Abstract base class for different vector database implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class VectorRepositoryStrategy(ABC):
    """
    Abstract base class for vector database repository strategies.
    Defines the interface that all vector database implementations must follow.
    """

    @abstractmethod
    def add_embeddings(
        self,
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
        batch_size: int = 100
    ) -> bool:
        """
        Add embeddings to the vector database with metadata.

        Args:
            embeddings: List of embedding vectors (2D list, shape: [n, d])
            metadatas: List of metadata dictionaries for each embedding
            ids: Optional list of IDs for embeddings (length must match embeddings)
            batch_size: Size of batches for bulk insertion

        Returns:
            bool: True if successful, False otherwise
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the database.

        Args:
            query_embedding: Query vector to search for
            k: Number of top results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of dictionaries, each containing:
            - "id": str
            - "score": float
            - "metadata": Dict[str, Any]
            - "document": str
        """
        pass

    @abstractmethod
    def delete_collection(self) -> bool:
        """
        Delete the entire collection.

        Returns:
            bool: True if successful, False otherwise
        """
        pass

    @abstractmethod
    def delete_embedding(self, embedding_id: str) -> bool:
        """
        Delete a single embedding from the collection.
        
        Args:
            embedding_id: ID of the embedding to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass

    @abstractmethod
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.
        
        Returns:
            Dictionary containing collection statistics (e.g. size, dimension)
        """
        pass

    @abstractmethod
    def update_embedding(
        self, 
        embedding_id: str, 
        embedding: List[float], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update an existing embedding and optionally its metadata.
        
        Args:
            embedding_id: ID of the embedding to update
            embedding: New embedding vector
            metadata: Optional new metadata
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
