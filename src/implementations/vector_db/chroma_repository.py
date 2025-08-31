"""
ChromaDB implementation of VectorRepository.
Handles vector storage and retrieval usiing ChromaDB
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import uuid
import logging
from pathlib import Path

from src.core.abstractions.vector_repository_strategy import VectorRepositoryStrategy


class ChromaRepository(VectorRepositoryStrategy):
    """
    ChromaDB implementation of vector repository.
    """

    def __init__(
        self,
        persist_directory: str,
        collection_name: str = "academic_documents"
    ):
        """
        Initialize ChromaDB repository.

        Args:
            persist_direcitory: Directory to persist ChromaDB data
            collection_name: Name of the collection to use
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name =collection_name
        self.logger = logging.getLogger(__name__) # ?

        # Create persist directory if it doesn't exist
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            self.logger.info(f"Loaded existing collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Academic documents collection"}
            )
            self.logger.info(f"Created new collection: {collection_name}")

    def add_embeddings(
        self,
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
        batch_size: int = 100
    ) -> bool:
        """
        Add embeddings to ChromaDB with metadata in batches.
        
        Args:
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries
            ids: Optional list of IDs, will generate UUIDs if not provided
            batch_size: Size of batches for bulk insertion
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Geberate IDs if not provided
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in range(len(embeddings))]

            
            # Validate input lengths
            if not (len(embeddings) == len(metadatas) == len(ids)):
                raise ValueError("Length mismatch between embeddings, metadatas, and ids")

            # Process in batches
            total_items = len(embeddings)
            for i in range(0, total_items, batch_size):
                end_idx = min(i + batch_size, total_items)

                batch_embeddings = embeddings[i:end_idx]
                batch_metadatas = metadatas[i:end_idx]
                batch_ids = ids[i:end_idx]

                # Add documents field for ChromaDB (using text from metadata if available)
                batch_documents = []
                for metadata in batch_metadatas:
                    doc_text = metadata.get('text', metadata.get('content', f"Document {metadata.get('page', 'unknown')}"))
                    batch_documents.append(doc_text)

                self.collection.add(
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    documents=batch_documents,
                    ids=batch_ids
                )

                self.logger.debug(f"Added batch {i//batch_size + 1}: {end_idx - i} embeddings")

            self.logger.info(f"Successfully added {total_items} embeddings to collection")
            return True

        except Exception as e:
            self.logger.error(f"Error adding embeddings: {str(e)}")
            return False

    def search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in ChromaDB.
        
        Args:
            query_embedding: Query vector to search for
            k: Number of top results to return
            filter_metadata: Optional metadata filters (ChromaDB where clause)
            
        Returns:
            List of dictionaries containing search results
        """
        try:
            where_clause = filter_metadata if filter_metadata else None

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_clause,
                include=['metadatas', 'documents', 'distances']
            )

            formatted_results = []
            if results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    result = {
                        'id': results['ids'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'document': results['documents'][0][i],
                        'score': 1 - results['distances'][0][i]
                    }
                    formatted_results.append(result)

            self.logger.info(f"Retrieved {len(formatted_results)} results for query")
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Error searching embeddings: {str(e)}")
            return []

    def delete_collection(self) -> bool:
        """
        Delete the entire collection.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.client.delete_collection(name=self.collection_name)
            self.logger.info(f"Deleted collection: {self.collection_name}")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting collection: {str(e)}")
            return False

    def delete_embedding(self, embedding_id: str) -> bool:
        """
        Delete a single embedding from the collection.
        
        Args:
            embedding_id: ID of the embedding to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.collection.delete(ids=[embedding_id])
            self.logger.info(f"Deleted embedding: {embedding_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting embedding {embedding_id}: {str(e)}")
            return False

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the ChromaDB collection.
        
        Returns:
            Dictionary containing collection statistics
        """
        try:
            count = self.collection.count()
            collection_metadata = self.collection.metadata

            info = {
                'collection_name': self.collection_name,
                'document_count': count,
                'persist_directory': str(self.persist_directory),
                'collection_metadata': collection_metadata
            }

            self.logger.info(f"Collection info retrieved: {count} documents")
            return info

        except Exception as e:
            self.logger.error(f"Error getting collection info: {str(e)}")
            return {
                    'collection_name': self.collection_name,
                    'document_count': 0,
                    'persist_directory': str(self.persist_directory),
                    'error': str(e)
                }

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
        try:
            doc_text = (metadata or {}).get('text', f"Updated document {embedding_id}")
            
            self.collection.upsert(
                embeddings=[embedding],
                metadatas=[metadata] if metadata else [{}],
                documents=[doc_text],
                ids=[embedding_id]
            )
            
            self.logger.info(f"Updated embedding: {embedding_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating embedding {embedding_id}: {str(e)}")
            return False