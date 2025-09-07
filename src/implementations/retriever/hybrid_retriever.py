"""
Hybrid Retriever Module for German Academic RAG System.

This module implements a hybrid search approach combining vector similarity
and keyword matching for improved and sustainable retrieval quality in German academic content.
"""

from typing import List, Dict, Optional
import numpy as np
from collections import Counter
import re
import math
from src.implementations.vector_db.vector_database_service import VectorDatabaseService
from src.utils.logger import setup_logger
from src.utils.config_parser import CONFIG
from src.implementations.embedding.query_embedding import QueryEmbeddingService

logger = setup_logger(__name__)

class HybridRetriever:
    """
    Hybrid retriever that combines vector similarity and keyword matching
    for enhanced context retrieval in German academic documents.
    """

    def __init__(
        self, 
        vector_db_service: VectorDatabaseService, 
        query_embedding_service: QueryEmbeddingService,
        config: Optional[Dict] = None
    ):
        """
        Initialize the hybrid retriever.

        Args:
            vector_db_service: Vector database service for similarity search
            query_embedding_service: QueryEmbeddingService for converting query text to embeddings (required)
            config: Optional configuration dictionary
        """
        if query_embedding_service is None:
            raise ValueError("HybridRetriever requires a valid QueryEmbeddingService instance.")
        
        self.vector_db = vector_db_service
        self.query_embedding_service = query_embedding_service

        # Load weights from config or use defaults
        retriever_config = config or CONFIG.get('retriever', {})
        self.vector_weight = retriever_config.get('vector_weight', 0.7)
        self.keyword_weight = retriever_config.get('keyword_weight', 0.3)
        self.min_word_length = retriever_config.get('min_word_length', 3)
        self.german_stopwords = self._load_german_stopwords()

        logger.info(f"HybridRetriever initialized with vector_weight={self.vector_weight}, keyword_weight={self.keyword_weight}")

    def _load_german_stopwords(self) -> set:
        """ Load German stopwords for better keyword matching."""
        stopwords = {
            'der', 'die', 'das', 'den', 'dem', 'des', 'ein', 'eine', 'eines', 'einem', 'einen',
            'und', 'oder', 'aber', 'doch', 'sondern', 'jedoch', 'dennoch',
            'ist', 'sind', 'war', 'waren', 'wird', 'werden', 'wurde', 'wurden',
            'hat', 'haben', 'hatte', 'hatten', 'kann', 'können', 'könnte', 'könnten',
            'soll', 'sollen', 'sollte', 'sollten', 'muss', 'müssen', 'musste', 'mussten',
            'für', 'von', 'zu', 'mit', 'bei', 'nach', 'vor', 'über', 'unter', 'durch',
            'auf', 'an', 'in', 'aus', 'zwischen', 'während', 'seit', 'bis',
            'auch', 'noch', 'nur', 'schon', 'bereits', 'immer', 'nie', 'niemals',
            'sehr', 'mehr', 'weniger', 'viel', 'wenig', 'alle', 'jede', 'jeder', 'jedes'
        }
        return stopwords

    def get_relevant_contexts(self, query_text: str, query_embedding: Optional[List[float]] = None, k: int = 5) -> List[Dict]:
        """
        Perform hybrid search combining vector similarity and keyword matching.
        
        Args:
            query_text: Raw query text for keyword matching
            query_embedding: Optional query embedding vector (if not provided, will be generated)
            k: Number of results to return
            
        Returns:
            List of dictionaries containing context metadata and hybrid scores
        """
        try:
            # Generate embedding if not provided
            if query_embedding is None:
                query_embedding = self.query_embedding_service.embed_text(query_text)

            # Fetch vector similarity candidates
            vector_candidates = k * 3
            vector_results = self.vector_db.search_by_embedding(
                query_embedding=query_embedding,
                top_k=vector_candidates
            )

            if not vector_results:
                logger.warning("No vector results found.")
                return []

            results_with_hybrid_score = []

            for result in vector_results:
                metadata = result.metadata if hasattr(result, 'metadata') else {}
                vector_score = result.similarity if hasattr(result, "similarity") else 0.0
                context_text = metadata.get('text', '')

                keyword_score = self._calculate_keyword_score(query_text, context_text)
                hybrid_score = (self.vector_weight * vector_score + self.keyword_weight * keyword_score)

                enhanced_metadata = {
                    **metadata,
                    'vector_score': vector_score,
                    'keyword_score': keyword_score,
                    'hybrid_score': hybrid_score,
                    'retrieval_method': 'hybrid'
                }

                results_with_hybrid_score.append(enhanced_metadata)

            results_with_hybrid_score.sort(key=lambda x: x['hybrid_score'], reverse=True)
            return results_with_hybrid_score[:k]

        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {str(e)}")
            return []

    def _calculate_keyword_score(self, query: str, context: str) -> float:
        """
        Calculate keyword matching score using improved TF-IDF-like approach.
        """
        try:
            query_terms = self._preprocess_text(query)
            context_terms = self._preprocess_text(context)

            if not query_terms or not context_terms:
                return 0.0
            
            query_tf = Counter(query_terms)
            context_tf = Counter(context_terms)

            total_score = 0.0
            total_query_weight = 0.0

            for term, query_freq in query_tf.items():
                if term in context_tf:
                    tf_score = context_tf[term] / len(context_terms)
                    query_importance = query_freq / len(query_terms)
                    term_score = tf_score * query_importance
                    total_score += term_score
                total_query_weight += query_importance

            normalized_score = total_score / total_query_weight if total_query_weight > 0 else 0.0
            final_score = math.sqrt(normalized_score)
            return min(final_score, 1.0)
        except Exception as e:
            logger.error(f"Error calculating keyword score: {str(e)}")
            return 0.0

    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess German text for keyword matching.
        """
        if not text:
            return []
        
        text = text.lower()
        text = re.sub(r'[^\w\säöüß]', ' ', text)
        words = text.split()

        processed_terms = []
        for word in words:
            if (len(word) >= self.min_word_length and word not in self.german_stopwords):
                processed_terms.append(word)
        return processed_terms

    def get_vector_results_only(self, query_embedding: List[float], k:int = 5) -> List[Dict]:
        """
        Get results using only vector similarity (for comparison/fallback).
        """
        try:
            vector_results = self.vector_db.search_by_embedding(
                query_embedding=query_embedding, 
                top_k=k
            )

            enhanced_results = []
            for result in vector_results:
                metadata = result.metadata if hasattr(result, 'metadata') else {}
                enhanced_metadata = {
                    **metadata,
                    'vector_score': result.similarity if hasattr(result, 'similarity') else 0.0,
                    'retrieval_method': 'vector_only'
                }
                enhanced_results.append(enhanced_metadata)
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Error in vector-only retrieval: {str(e)}")
            return []

    def update_weights(self, vector_weight: float, keyword_weight: float) -> None:
        """
        Update hybrid search weights dynamically.
        """
        total_weight = vector_weight + keyword_weight
        if total_weight > 0:
            self.vector_weight = vector_weight / total_weight
            self.keyword_weight = keyword_weight / total_weight
        else:
            logger.warning("Invalid weights provided, keeping current weights")
            return
        
        logger.info(f"Updated weights: vector={self.vector_weight:.2f}, keyword={self.keyword_weight:.2f}")

    def get_retrieval_stats(self, results: List[Dict]) -> Dict:
        """
        Get statistics about retrieval results.
        """
        if not results:
            return {"total_results": 0}
        
        stats = {
            "total_results": len(results),
            "avg_hybrid_score": np.mean([r.get('hybrid_score', 0) for r in results]),
            "avg_vector_score": np.mean([r.get('vector_score', 0) for r in results]),
            "avg_keyword_score": np.mean([r.get('keyword_score', 0) for r in results]),
            "score_range": {
                "min_hybrid": min(r.get('hybrid_score', 0) for r in results),
                "max_hybrid": max(r.get('hybrid_score', 0) for r in results)
            }
        }
        return stats


def create_hybrid_retriever(config: Optional[Dict] = None, query_embedding_service: QueryEmbeddingService = None) -> HybridRetriever:
    """
    Factory function to create a configured HybridRetriever instance.
    QueryEmbeddingService is required.
    """
    if query_embedding_service is None:
        raise ValueError("create_hybrid_retriever requires a valid QueryEmbeddingService instance.")
    
    try:
        vector_db_service = VectorDatabaseService()
        retriever = HybridRetriever(vector_db_service, query_embedding_service, config)
        logger.info("HybridRetriever created successfully")
        return retriever
    except Exception as e:
        logger.error(f"Error creating HybridRetriever: {str(e)}")
        raise


if __name__ == "__main__":
    """Example of how to use the HybridRetriever with embedding service."""
    
    from src.implementations.embedding.query_embedding import create_query_embedding_service

    query_embedding_service = create_query_embedding_service()
    retriever = create_hybrid_retriever(query_embedding_service=query_embedding_service)
    
    query_text = "Was ist das Bruttoinlandsprodukt?"
    contexts = retriever.get_relevant_contexts(query_text=query_text, k=5)
    
    print(f"Found {len(contexts)} relevant contexts:")
    for i, context in enumerate(contexts, 1):
        print(f"\n{i}. Hybrid Score: {context.get('hybrid_score', 0):.3f}")
        print(f"   Vector Score: {context.get('vector_score', 0):.3f}")
        print(f"   Keyword Score: {context.get('keyword_score', 0):.3f}")
        print(f"   Text Preview: {context.get('text', '')[:100]}...")

    stats = retriever.get_retrieval_stats(contexts)
    print(f"\nRetrieval Statistics: {stats}")
