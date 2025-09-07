"""
Embedding Service for RAG system.

Provides toxt-to-vector conversation using HuggingFace sentence-transformers.
"""

from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from functools import lru_cache

from src.utils.logger import setup_logger
from src.utils.config_parser import CONFIG

logger = setup_logger(__name__)

class QueryEmbeddingService:
    """
    Service for converting text to embeddings using HuggingFace models.
    """

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the embedding service.
        
        Args:
            model_name: HuggingFace model name, defaults to config or multilingual model
            device: Device to use ('cpu', 'cuda', etc.), defaults to auto-detect
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

        logger.info(f"Loading local embedding model: {model_name} on {device}")
        try:
            self.model = SentenceTransformer(model_name, device=device)
            logger.info(f"Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise

    def embed_text(self, text: str) -> List[float]:
        """
        Convert single text to embedding vector.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return []
        
        return self._embed_text_local(text)

    def _embed_text_local(self, text: str) -> List[float]:
        """Internal method for local embedding."""
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error in local embedding: {str(e)}")
            return []
    
    @lru_cache(maxsize=1000)
    def embed_text_cached(self, text: str) -> List[float]:
        """Cached embedding for repeated texts."""
        return self.embed_text(text)

    def embed_batch(self, texts: List[str], batch_size: int = 32, **kwargs) -> List[List[float]]:
        """
        Convert multiple texts to embedding vectors in batches.
        """
        if not texts:
            logger.warning("Empty text list provided for batch embedding")
            return []

        valid_texts = [text for text in texts if text and text.strip()]
        if not valid_texts:
            logger.warning("No valid texts found for batch embedding")
            return []

        return self._embed_batch_local(valid_texts, batch_size, **kwargs)

    def _embed_batch_local(self, texts: List[str], batch_size: int, **kwargs) -> List[List[float]]:
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                **kwargs
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error in local batch embedding: {str(e)}")
            return []

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        try:
            return self.model.get_sentence_embedding_dimension()
        except Exception as e:
            logger.error(f"Error getting embedding dimension: {str(e)}")
            return 0

    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            if not embedding1 or not embedding2:
                return 0.0
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        try:
            info = {
                'model_name': self.model_name,
                'device': self.device,
                'embedding_dimension': self.get_embedding_dimension(),
                'max_seq_length': getattr(self.model, 'max_seq_length', 'unknown')
            }
            return info
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {}

def create_query_embedding_service(model_name: Optional[str] = None, device: Optional[str] = None) -> QueryEmbeddingService:
    """Factory function to create an QueryEmbeddingService instance."""
    try:
        service = QueryEmbeddingService(model_name=model_name, device=device)
        logger.info("QueryEmbeddingService created successfully")
        return service
    except Exception as e:
        logger.error(f"Error creating QueryEmbeddingService: {str(e)}")
        raise


if __name__ == "__main__":
    from src.implementations.embedding.query_embedding import QueryEmbeddingService
    from src.implementations.retriever.hybrid_retriever import create_hybrid_retriever

    query_embedding_service = QueryEmbeddingService()
    query_text = "Was ist das Bruttoinlandsprodukt?"
    query_embedding = query_embedding_service.embed_text(query_text)
    print(f"Query embedding dimension: {len(query_embedding)}")

    retriever = create_hybrid_retriever(query_embedding_service=query_embedding_service)
    contexts = retriever.get_relevant_contexts(
        query_embedding=query_embedding,
        query_text=query_text,
        k=5
    )

    print(f"Found {len(contexts)} relevant contexts:")
    for i, context in enumerate(contexts, 1):
        print(f"\n{i}. Hybrid Score: {context.get('hybrid_score', 0):.3f}")
        print(f"   Vector Score: {context.get('vector_score', 0):.3f}")
        print(f"   Keyword Score: {context.get('keyword_score', 0):.3f}")
        print(f"   Text Preview: {context.get('text', '')[:100]}...")

    stats = retriever.get_retrieval_stats(contexts)
    print(f"\nRetrieval Statistics: {stats}")
