from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

from src.utils.logger import setup_logger
from src.utils.config_parser import CONFIG

logger = setup_logger(__name__)

class QueryEmbeddingService:
    """
    Service for converting user queries to embeddings using HuggingFace SentenceTransformers.
    Only processes queries; does not handle document embedding or retrieval!
    """

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the query embedding service.

        Args:
            model_name: HuggingFace SentenceTransformer model name
            device: 'cpu', 'cuda', or auto-detect
        """
        embedding_config = CONFIG.get('embedding', {})

        if model_name is None:
            model_name = embedding_config.get('model_name', 'intfloat/multilingual-e5-base')
        
        if device is None:
            device = embedding_config.get('device', 'auto')
            if device == 'auto':
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model_name = model_name
        self.device = device

        logger.info(f"Loading query embedding model: {model_name} on {device}")
        try:
            self.model = SentenceTransformer(model_name, device=device)
            logger.info(f"Query embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load query embedding model: {str(e)}")
            raise

    def embed_text(self, text: str) -> List[float]:
        """
        Convert a single user query to an embedding vector.

        Args:
            text: Query string

        Returns:
            Embedding vector as list of floats
        """
        if not text or not text.strip():
            logger.warning("Empty query provided for embedding")
            return []

        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error embedding query: {str(e)}")
            return []

    def get_embedding_dimension(self) -> int:
        """Return dimension of query embeddings."""
        try:
            return self.model.get_sentence_embedding_dimension()
        except Exception as e:
            logger.error(f"Error retrieving embedding dimension: {str(e)}")
            return 0

    def get_model_info(self) -> dict:
        """
        Return basic info about the query embedding model.
        This is used by RAGPipeline.get_pipeline_info() to avoid AttributeError.
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "embedding_dim": self.get_embedding_dimension()
        }


def create_query_embedding_service(model_name: Optional[str] = None, device: Optional[str] = None) -> QueryEmbeddingService:
    """Factory to instantiate QueryEmbeddingService."""
    return QueryEmbeddingService(model_name=model_name, device=device)


if __name__ == "__main__":
    service = create_query_embedding_service()
    query_text = "Was ist das Bruttoinlandsprodukt?"
    embedding = service.embed_text(query_text)
    print(f"Query embedding dimension: {len(embedding)}")
    print(f"First 10 values: {embedding[:10]}")