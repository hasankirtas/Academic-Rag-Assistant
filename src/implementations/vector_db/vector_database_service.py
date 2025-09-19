"""
Vector Database Service.
High-level service layer for vector database operations with business logic.
"""

from typing import List, Dict, Any, Optional, Union
import logging
from dataclasses import dataclass

from src.core.vector_database_module import VectorDatabaseModule, EmbeddingData, SearchResult
from src.core.factories.vector_repository_factory import VectorDBType


@dataclass
class DocumentChunk:
    """
    Data class for document chunks with metadata.
    """
    text: str
    page: int
    section: Optional[str] = None
    chunk_index: Optional[int] = None
    additional_metadata: Optional[Dict[str, Any]] = None


@dataclass
class SearchQuery:
    """
    Data class for search queries with parameters.
    """
    query_text: str
    top_k: int = 5
    filters: Optional[Dict[str, Any]] = None
    min_similarity: float = 0.0


class VectorDatabaseService:
    """
    High-level service for vector database operations.
    Provides business logic and orchestration for vector database operations.
    """

    def __init__(
        self,
        db_type: VectorDBType = VectorDBType.CHROMA,
        **config
    ):
        """
        Initialize the vector database service.

        Args:
            db_type: Type of vector database to use
            **config: Configuration parameters for the database
        """
        self.logger = logging.getLogger(__name__)
        self.module = VectorDatabaseModule(db_type, **config)
        self.logger.info(f"VectorDatabaseService initialized with {db_type.value}")

    def store_document_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
        batch_size: int = 100
    ) -> bool:
        """
        Store document chunks with their embeddings.

        Args:
            chunks: List of DocumentChunk objects
            embeddings: List of embedding vectors
            batch_size: Size of batches for bulk insertion

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if len(chunks) != len(embeddings):
                raise ValueError("Length mismatch between chunks and embeddings")

            embedding_data = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                metadata = {
                    "text": chunk.text,
                    "content": chunk.text,
                    "page": chunk.page,
                    "chunk_index": chunk.chunk_index or i,
                    "contains_table": self._detect_table_content(chunk.text),
                    "word_count": len(chunk.text.split()),
                    "char_count": len(chunk.text),
                }

                if chunk.section:
                    metadata["section"] = chunk.section
                if chunk.additional_metadata:
                    metadata.update(chunk.additional_metadata)

                embedding_data.append(EmbeddingData(embedding=embedding, metadata=metadata))

            return self.module.store_embeddings(embedding_data, batch_size=batch_size)

        except Exception as e:
            self.logger.error(f"Error in store_document_chunks: {str(e)}")
            return False

    def search_documents(
        self,
        query: SearchQuery,
        embedding_provider=None
    ) -> List[SearchResult]:
        """
        Search for documents using text query.

        Args:
            query: SearchQuery object containing search parameters
            embedding_provider: Optional embedding provider for text-to-vector conversion

        Returns:
            List of SearchResult objects
        """
        try:
            if embedding_provider is None:
                raise ValueError("Embedding provider is required for text search")

            # Convert text query to embedding
            query_embedding = embedding_provider.embed_text(query.query_text)

            return self.module.search_similar(
                query_embedding=query_embedding,
                top_k=query.top_k,
                filters=query.filters,
                min_similarity=query.min_similarity
            )

        except Exception as e:
            self.logger.error(f"Error in search_documents: {str(e)}")
            return []

    def search_by_embedding(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        min_similarity: float = 0.0
    ) -> List[SearchResult]:
        """
        Search for documents using embedding vector.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            filters: Optional metadata filters
            min_similarity: Minimum similarity threshold

        Returns:
            List of SearchResult objects
        """
        return self.module.search_similar(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
            min_similarity=min_similarity
        )

    def update_document_chunk(
        self,
        chunk_id: str,
        new_text: str,
        new_embedding: List[float],
        new_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update a document chunk.

        Args:
            chunk_id: ID of the chunk to update
            new_text: New text content
            new_embedding: New embedding vector
            new_metadata: Optional new metadata

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            metadata = {
                "text": new_text,
                "content": new_text,
                "word_count": len(new_text.split()),
                "char_count": len(new_text),
                "contains_table": self._detect_table_content(new_text),
            }

            if new_metadata:
                metadata.update(new_metadata)

            return self.module.update_embedding(
                embedding_id=chunk_id,
                new_embedding=new_embedding,
                new_metadata=metadata
            )

        except Exception as e:
            self.logger.error(f"Error in update_document_chunk: {str(e)}")
            return False

    def delete_document_chunk(self, chunk_id: str) -> bool:
        """
        Delete a document chunk.

        Args:
            chunk_id: ID of the chunk to delete

        Returns:
            bool: True if successful, False otherwise
        """
        return self.module.delete_embedding(chunk_id)

    def get_database_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive database statistics.

        Returns:
            Dictionary containing database statistics
        """
        try:
            stats = self.module.get_database_stats()
            
            # Add additional computed statistics
            if "document_count" in stats:
                stats["total_chunks"] = stats["document_count"]
                stats["estimated_size_mb"] = stats.get("document_count", 0) * 0.001  # Rough estimate
            
            return stats

        except Exception as e:
            self.logger.error(f"Error getting database statistics: {str(e)}")
            return {"error": str(e)}

    def clear_all_data(self) -> bool:
        """
        Clear all data from the database.

        Returns:
            bool: True if successful, False otherwise
        """
        return self.module.clear_database()

    def search_by_page(
        self,
        query_embedding: List[float],
        page_number: int,
        top_k: int = 5,
        min_similarity: float = 0.0
    ) -> List[SearchResult]:
        """
        Search for documents on a specific page.

        Args:
            query_embedding: Query embedding vector
            page_number: Page number to search in
            top_k: Number of top results to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of SearchResult objects
        """
        filters = {"page": page_number}
        return self.search_by_embedding(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
            min_similarity=min_similarity
        )

    def search_by_section(
        self,
        query_embedding: List[float],
        section_name: str,
        top_k: int = 5,
        min_similarity: float = 0.0
    ) -> List[SearchResult]:
        """
        Search for documents in a specific section.

        Args:
            query_embedding: Query embedding vector
            section_name: Section name to search in
            top_k: Number of top results to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of SearchResult objects
        """
        filters = {"section": section_name}
        return self.search_by_embedding(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
            min_similarity=min_similarity
        )

    def get_chunks_by_page(self, page_number: int, top_k: int = 100) -> List[SearchResult]:
        """
        Get all chunks from a specific page.

        Args:
            page_number: Page number to retrieve
            top_k: Maximum number of chunks to return

        Returns:
            List of SearchResult objects
        """
        try:
            # Use a dummy embedding to get all results from the page
            dummy_embedding = [0.0] * 768  # Assuming 768-dimensional embeddings
            return self.search_by_page(
                query_embedding=dummy_embedding,
                page_number=page_number,
                top_k=top_k,
                min_similarity=0.0
            )
        except Exception as e:
            self.logger.error(f"Error getting chunks by page: {str(e)}")
            return []

    def _detect_table_content(self, text: str) -> bool:
        """
        Detect if text contains table content.

        Args:
            text: Text to analyze

        Returns:
            bool: True if table content is detected
        """
        table_indicators = [
            "|", "\t", "Tabelle", "Tab.", "Spalte", "Zeile",
            "siehe Tabelle", "in der Tabelle", "─", "│"
        ]
        return any(ind in text for ind in table_indicators)
