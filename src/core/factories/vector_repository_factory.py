"""
Vector Repository Factory Pattern.
Creates and configures different vector database implementations.
"""

from typing import Dict, Any, Optional
import logging
from enum import Enum

from src.core.abstractions.vector_repository_strategy import VectorRepositoryStrategy
from src.utils.config_parser import CONFIG


class VectorDBType(Enum):
    """Supported vector database types."""
    CHROMA = "chroma"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    QDRANT = "qdrant"


class VectorRepositoryFactory:
    """
    Factory class for creating vector repository instances.
    Supports different vector database backends.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def create_repository(
        self,
        db_type: VectorDBType,
        **kwargs
    ) -> VectorRepositoryStrategy:
        """
        Create a vector repository instance based on the specified type.

        Args:
            db_type: Type of vector database to create
            **kwargs: Configuration parameters for the specific database

        Returns:
            VectorRepositoryStrategy: Configured repository instance

        Raises:
            ValueError: If unsupported database type is specified
        """
        try:
            if db_type == VectorDBType.CHROMA:
                return self._create_chroma_repository(**kwargs)
            elif db_type == VectorDBType.PINECONE:
                return self._create_pinecone_repository(**kwargs)
            elif db_type == VectorDBType.WEAVIATE:
                return self._create_weaviate_repository(**kwargs)
            elif db_type == VectorDBType.QDRANT:
                return self._create_qdrant_repository(**kwargs)
            else:
                raise ValueError(f"Unsupported vector database type: {db_type}")

        except Exception as e:
            self.logger.error(f"Error creating {db_type.value} repository: {str(e)}")
            raise

    def _create_chroma_repository(self, **kwargs) -> VectorRepositoryStrategy:
        """Create ChromaDB repository instance."""
        from src.implementations.vector_db.chroma_repository import ChromaRepository
        
        # Read defaults from global CONFIG if available
        vcfg = (CONFIG.get('vectordb', {}) or {}).get('chroma', {})
        persist_directory = kwargs.get('persist_directory', vcfg.get('persist_directory', './chroma_db'))
        collection_name = kwargs.get('collection_name', vcfg.get('collection_name', 'academic_documents'))
        settings = kwargs.get('settings', vcfg.get('settings', {
            'allow_reset': True
        }))
        metadata = kwargs.get('metadata', vcfg.get('metadata', {
            'description': 'Academic documents collection'
        }))
        
        return ChromaRepository(
            persist_directory=persist_directory,
            collection_name=collection_name,
            settings=settings,
            collection_metadata=metadata,
        )

    def _create_pinecone_repository(self, **kwargs) -> VectorRepositoryStrategy:
        """Create Pinecone repository instance."""
        # TODO: Implement Pinecone repository
        raise NotImplementedError("Pinecone repository not yet implemented")

    def _create_weaviate_repository(self, **kwargs) -> VectorRepositoryStrategy:
        """Create Weaviate repository instance."""
        # TODO: Implement Weaviate repository
        raise NotImplementedError("Weaviate repository not yet implemented")

    def _create_qdrant_repository(self, **kwargs) -> VectorRepositoryStrategy:
        """Create Qdrant repository instance."""
        # TODO: Implement Qdrant repository
        raise NotImplementedError("Qdrant repository not yet implemented")

    def get_supported_types(self) -> list:
        """Get list of supported vector database types."""
        return [db_type.value for db_type in VectorDBType]

    def get_repository_config_template(self, db_type: VectorDBType) -> Dict[str, Any]:
        """
        Get configuration template for a specific database type.

        Args:
            db_type: Type of vector database

        Returns:
            Dict containing configuration template
        """
        templates = {
            VectorDBType.CHROMA: {
                "persist_directory": "./chroma_db",
                "collection_name": "academic_documents"
            },
            VectorDBType.PINECONE: {
                "api_key": "your_pinecone_api_key",
                "environment": "your_pinecone_environment",
                "index_name": "academic_documents"
            },
            VectorDBType.WEAVIATE: {
                "url": "http://localhost:8080",
                "class_name": "AcademicDocument"
            },
            VectorDBType.QDRANT: {
                "url": "http://localhost:6333",
                "collection_name": "academic_documents"
            }
        }
        
        return templates.get(db_type, {})
