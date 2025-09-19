"""
HuggingFace Embedding Implementation (local)

This module performs embedding operations using the HuggingFace
Transformers library.
"""

from typing import List, Optional
import logging

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    import numpy as np
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    torch = None
    AutoTokenizer = None
    AutoModel = None
    np = None

from ...core.abstractions.embedding_strategy import EmbeddingStrategy
from ...utils.logger import setup_logger

logger = setup_logger(__name__)


class EmbedderError(Exception):
    """Custom exception for HuggingFace embedder-related errors."""
    pass


class Embedder(EmbeddingStrategy):
    """
    HuggingFace-based embedding implementation (local only for RAG-Pipeline).

    Supports sentence transformers and other HuggingFace models.
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        **kwargs
    ):
        """
        HuggingFace embedder constructor.

        Args:
            model_name: HuggingFace model name
            device: Device to use ('cpu', 'cuda', None=auto)
            max_length: Maximum token length
            batch_size: Batch processing size
            normalize_embeddings: Whether to normalize embeddings
            **kwargs: Additional parameters
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "HuggingFace Transformers library is not installed. "
                "Please run 'pip install transformers torch'."
            )

        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.provider = "huggingface"

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Initializing HuggingFace embedder: {model_name} on {self.device}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.model = AutoModel.from_pretrained(model_name, local_files_only=True)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Model loaded from local cache: {model_name}")

        except Exception:
            logger.warning(
                f"Model '{model_name}' not found in local cache. It will be downloaded automatically."
            )
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                self.model.to(self.device)
                self.model.eval()
            except Exception as e:
                logger.error(f"Model loading error: {str(e)}")
                raise EmbedderError(f"Failed to load HuggingFace model: {str(e)}") from e

        self._embedding_dimension = getattr(self.model.config, "hidden_size", None)
        if self._embedding_dimension is None:
            self._embedding_dimension = getattr(self.model.config, "d_model", None)

        logger.info(f"Model successfully loaded. Embedding dimension: {self._embedding_dimension}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Convert a list of documents into embedding vectors.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        logger.debug(f"Starting embedding for {len(texts)} documents")

        all_embeddings = []

        # Process in batch
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self._embed_batch(batch_texts)
            all_embeddings.extend(batch_embeddings)

            if len(texts) > self.batch_size:
                logger.debug(
                    f"Batch {i // self.batch_size + 1}/"
                    f"{(len(texts) + self.batch_size - 1) // self.batch_size} completed"
                )

        logger.debug(f"Completed embedding for {len(all_embeddings)} documents")
        return all_embeddings

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Convert a batch of texts into embeddings.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        try:
            # Special prefix for documents (E5 models)
            if "e5" in self.model_name.lower():
                texts = [f"passage: {text}" for text in texts]

            # Tokenize
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            # Move to device safely
            encoded = {k: v.to(self.device) for k, v in encoded.items() if v is not None}

            # Forward pass (without gradients)
            with torch.no_grad():
                outputs = self.model(**encoded)

            # Mean pooling
            embeddings = self._mean_pooling(outputs.last_hidden_state, encoded['attention_mask'])

            # Normalize if enabled
            if self.normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            # Move to CPU and convert to list
            embeddings = embeddings.cpu().numpy()
            return [embedding.tolist() for embedding in embeddings]

        except Exception as e:
            logger.error(f"Batch embedding error: {str(e)}")
            raise EmbedderError(f"HuggingFace embedding error: {str(e)}") from e

    def _mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Convert token embeddings to sentence embeddings via mean pooling.

        Args:
            token_embeddings: Token-level embeddings
            attention_mask: Attention mask

        Returns:
            Sentence embeddings
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def get_model_info(self) -> dict:
        """
        Return information about the loaded model.

        Returns:
            Dict containing model info
        """
        return {
            "model_name": self.model_name,
            "provider": self.provider,
            "device": self.device,
            "embedding_dimension": self._embedding_dimension,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "normalize_embeddings": self.normalize_embeddings
        }

    def get_embedding_dimension(self) -> int:
        """
        Return the embedding dimension of the model.
        """
        return self._embedding_dimension

    def embed_query(self, query: str) -> List[float]:
        """
        Convert a query text into an embedding vector.

        Args:
            query: Query text to embed

        Returns:
            Embedding vector
        """
        if not query or not query.strip():
            raise ValueError("Query text cannot be empty")

        # Special prefix for queries (E5 models)
        if "e5" in self.model_name.lower():
            query = f"query: {query}"

        # Use embed_documents for single query
        embeddings = self.embed_documents([query])
        return embeddings[0] if embeddings else []

    def embed_text(self, text: str) -> List[float]:
        """
        Alias for embed_query for compatibility with QueryEmbeddingService.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        return self.embed_query(text)
