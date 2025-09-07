"""
Vector Database Module.
Main module that orchestrates vector database operations using Strategy and Factory patterns.
"""

from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass

from src.core.abstractions.vector_repository_strategy import VectorRepositoryStrategy
from src.core.factories.vector_repository_factory import VectorRepositoryFactory, VectorDBType


@dataclass
class EmbeddingData:
    """
    Data class for embedding with metadata.
    """
    embedding: List[float]
    metadata: Dict[str, Any]
    id: Optional[str] = None


@dataclass
class SearchResult:
    """
    Data class for search results.
    """
    id: str
    metadata: Dict[str, Any]
    document: str
    score: float
    distance: Optional[float] = None


class VectorDatabaseModule:
    """
    Main module for vector database operations.
    Uses Strategy and Factory patterns to provide a unified interface.
    """

    def __init__(
        self,
        db_type: VectorDBType = VectorDBType.CHROMA,
        **config
    ):
        """
        Initialize the vector database module.

        Args:
            db_type: Type of vector database to use
            **config: Configuration parameters for the database
        """
        self.logger = logging.getLogger(__name__)
        self.factory = VectorRepositoryFactory()
        self.repository = self.factory.create_repository(db_type, **config)
        self.logger.info(f"VectorDatabaseModule initialized with {db_type.value}")

    def store_embeddings(
        self,
        embedding_data: List[EmbeddingData],
        batch_size: int = 100
    ) -> bool:
        """
        Store multiple embeddings in the database.

        Args:
            embedding_data: List of EmbeddingData objects
            batch_size: Size of batches for bulk insertion

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not embedding_data:
                self.logger.warning("No embedding data provided")
                return False

            embeddings = [d.embedding for d in embedding_data]
            metadatas = [d.metadata for d in embedding_data]
            ids = [d.id for d in embedding_data] if all(d.id is not None for d in embedding_data) else None

            success = self.repository.add_embeddings(
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
                batch_size=batch_size
            )

            if success:
                self.logger.info(f"Stored {len(embedding_data)} embeddings")
            else:
                self.logger.error("Failed to store embeddings")

            return success
        except Exception as e:
            self.logger.error(f"Error in store_embeddings: {str(e)}")
            return False

    def store_single_embedding(
        self,
        embedding: List[float],
        metadata: Dict[str, Any],
        embedding_id: Optional[str] = None
    ) -> bool:
        """
        Store a single embedding in the database.

        Args:
            embedding: Embedding vector
            metadata: Metadata dictionary
            embedding_id: Optional ID for the embedding

        Returns:
            bool: True if successful, False otherwise
        """
        return self.store_embeddings(
            [EmbeddingData(embedding=embedding, metadata=metadata, id=embedding_id)],
            batch_size=1
        )

    def search_similar(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        min_similarity: float = 0.0
    ) -> List[SearchResult]:
        """
        Search for similar vectors in the database.

        Args:
            query_embedding: Query vector to search for
            top_k: Number of top results to return
            filters: Optional metadata filters
            min_similarity: Minimum similarity threshold

        Returns:
            List of SearchResult objects
        """
        try:
            raw_results = self.repository.search(
                query_embedding=query_embedding,
                k=top_k,
                filter_metadata=filters
            )

            search_results = []
            for r in raw_results:
                score = r.get("score") or r.get("similarity_score", 1 - r.get("distance", 1))
                if score >= min_similarity:
                    search_results.append(
                        SearchResult(
                            id=r["id"],
                            metadata=r["metadata"],
                            document=r["document"],
                            score=score,
                            distance=r.get("distance")
                        )
                    )

            self.logger.info(f"Found {len(search_results)} results above threshold")
            return search_results
        except Exception as e:
            self.logger.error(f"Error in search_similar: {str(e)}")
            return []

    def update_embedding(
        self,
        embedding_id: str,
        new_embedding: List[float],
        new_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update an existing embedding.

        Args:
            embedding_id: ID of the embedding to update
            new_embedding: New embedding vector
            new_metadata: Optional new metadata

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            success = self.repository.update_embedding(
                embedding_id=embedding_id,
                embedding=new_embedding,
                metadata=new_metadata or {}
            )

            if success:
                self.logger.info(f"Updated embedding {embedding_id}")
            else:
                self.logger.warning(f"Failed to update embedding {embedding_id}")
            return success
        except Exception as e:
            self.logger.error(f"Error updating embedding {embedding_id}: {str(e)}")
            return False

    def delete_embedding(self, embedding_id: str) -> bool:
        """
        Delete an embedding from the database.

        Args:
            embedding_id: ID of the embedding to delete

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            success = self.repository.delete_embedding(embedding_id)
            if success:
                self.logger.info(f"Deleted embedding {embedding_id}")
            else:
                self.logger.warning(f"Failed to delete embedding {embedding_id}")
            return success
        except Exception as e:
            self.logger.error(f"Error deleting embedding {embedding_id}: {str(e)}")
            return False

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary containing database statistics
        """
        try:
            return self.repository.get_collection_info()
        except Exception as e:
            self.logger.error(f"Error getting database stats: {str(e)}")
            return {"error": str(e)}

    def clear_database(self) -> bool:
        """
        Clear all data from the database.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            success = self.repository.delete_collection()
            if success:
                self.logger.info("Database cleared")
            else:
                self.logger.warning("Failed to clear database")
            return success
        except Exception as e:
            self.logger.error(f"Error clearing database: {str(e)}")
            return False

    def bulk_store_from_chunks(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        page_numbers: List[int],
        sections: Optional[List[str]] = None,
        additional_metadata: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 100
    ) -> bool:
        """
        Bulk store embeddings from document chunks.

        Args:
            chunks: List of text chunks
            embeddings: List of embedding vectors
            page_numbers: List of page numbers
            sections: Optional list of section names
            additional_metadata: Optional list of additional metadata
            batch_size: Size of batches for bulk insertion

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not (len(chunks) == len(embeddings) == len(page_numbers)):
                raise ValueError("Length mismatch between chunks, embeddings, and page numbers")

            embedding_data = []
            for i, (chunk, embedding, page) in enumerate(zip(chunks, embeddings, page_numbers)):
                metadata = {
                    "text": chunk,
                    "content": chunk,
                    "page": page,
                    "chunk_index": i,
                    "contains_table": self._detect_table_content(chunk),
                    "word_count": len(chunk.split()),
                    "char_count": len(chunk),
                }
                if sections and i < len(sections):
                    metadata["section"] = sections[i]
                if additional_metadata and i < len(additional_metadata):
                    metadata.update(additional_metadata[i])

                embedding_data.append(EmbeddingData(embedding=embedding, metadata=metadata))

            return self.store_embeddings(embedding_data, batch_size=batch_size)

        except Exception as e:
            self.logger.error(f"Error in bulk_store_from_chunks: {str(e)}")
            return False

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


if __name__ == "__main__":
    """
    Test script for VectorDatabaseModule.
    Tests basic functionality of the module.
    """
    import logging
    import random
    import os

    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    
    # Set random seed for reproducible tests
    random.seed(42)
    
    print("Testing VectorDatabaseModule")
    print("=" * 40)
    
    try:
        # Create module instance
        module = VectorDatabaseModule(
            persist_directory="./test_module_db",
            collection_name="test_module_collection"
        )
        
        # Test data
        dim = 8
        embeddings = [[random.random() for _ in range(dim)] for _ in range(3)]
        metadatas = [
            {"text": f"Test document {i}", "page": i + 1, "section": f"Section {i}"}
            for i in range(3)
        ]
        
        embedding_data = [
            EmbeddingData(embedding=emb, metadata=meta)
            for emb, meta in zip(embeddings, metadatas)
        ]
        
        # Test 1: Store embeddings
        print("Test 1: Storing embeddings...")
        success = module.store_embeddings(embedding_data, batch_size=2)
        print(f"✓ Store embeddings: {success}")
        
        # Test 2: Search similar
        print("Test 2: Searching similar embeddings...")
        query_embedding = [random.random() for _ in range(dim)]
        results = module.search_similar(query_embedding, top_k=2)
        print(f"✓ Search results: {len(results)} found")
        
        # Test 3: Get database stats
        print("Test 3: Getting database statistics...")
        stats = module.get_database_stats()
        print(f"✓ Database stats: {stats['document_count']} documents")
        
        # Test 4: Update embedding
        print("Test 4: Updating embedding...")
        if results:
            first_id = results[0].id
            new_embedding = [x + 0.1 for x in query_embedding]
            update_success = module.update_embedding(
                embedding_id=first_id,
                new_embedding=new_embedding,
                new_metadata={"text": "Updated document"}
            )
            print(f"✓ Update embedding: {update_success}")
        
        # Test 5: Delete embedding
        print("Test 5: Deleting embedding...")
        if results:
            delete_success = module.delete_embedding(results[0].id)
            print(f"✓ Delete embedding: {delete_success}")
        
        # Test 6: Clear database
        print("Test 6: Clearing database...")
        clear_success = module.clear_database()
        print(f"✓ Clear database: {clear_success}")
        
        # Clean up
        if os.path.exists("./test_module_db"):
            import shutil
            shutil.rmtree("./test_module_db")
        
        print("\n" + "=" * 40)
        print("✓ All tests completed successfully!")
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        # Clean up on error
        if os.path.exists("./test_module_db"):
            import shutil
            shutil.rmtree("./test_module_db")
