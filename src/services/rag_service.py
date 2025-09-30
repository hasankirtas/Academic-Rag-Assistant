"""
RAG Pipeline

Combines embedding service, retriever, and provides a interface for RAG operations.
"""

from typing import List, Dict, Optional
import logging

# Removed unused import

from src.implementations.embedding.query_embedding import QueryEmbeddingService
from src.implementations.vector_db.vector_database_service import VectorDatabaseService
from src.rag.retriever import Retriever
from src.utils.logger import setup_logger
from src.utils.performance_monitor import get_performance_monitor

logger = setup_logger(__name__)

class RAGPipeline:
    """
    RAP pipeline combining embedding service and retriever.
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the RAG pipeline with embedding service and retriever."""
        try:
            self.embedding_service = QueryEmbeddingService()
            self.vector_db_service = VectorDatabaseService()
            self.performance_monitor = get_performance_monitor()

            self.retriever = Retriever(
                vector_db_service=self.vector_db_service,
                query_embedding_service=self.embedding_service,
                config=config
            )
            
            logger.info("RAG Pipeline initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize RAG Pipeline: {str(e)}")
            raise

    def query(self, query_text: str, k: int = 5) -> Dict:
        """
        Perform a RAG query to find relevant contexts with performance monitoring.

        Args:
            query_text: The query text
            k: Number of results to return

        Returns:
            Dict containing the aggregated answer and individual contexts with metadata and scores
        """
        with self.performance_monitor.time_operation("rag_query", query_length=len(query_text), k=k):
            try:
                logger.info(f"Processing query: {query_text}")

                # Use retriever's built-in embedding generation
                # (retriever will call embedding_service internally)
                contexts = self.retriever.get_relevant_contexts(
                    query_text=query_text,
                    query_embedding=None,  # retriever generate it
                    k=k
                )

                # Aggregate answer text from retrieved contexts
                answer_text = " ".join([ctx.get("context", "") for ctx in contexts])

                logger.info(f"Retrieved {len(contexts)} contexts")

                return {
                    "success": True,
                    "answer": answer_text,
                    "contexts": contexts
                }

            except Exception as e:
                logger.error(f"Error in RAG query: {str(e)}")
                return {
                    "success": False,
                    "answer": "",
                    "contexts": []
                }
    
    def query_with_custom_embedding(self, query_text: str, query_embedding: List[float], k: int = 5):
        """
        Perform a RAG query with pre-computed embedding.
        
        Args:
            query_text: The query text (for keyword matching)
            query_embedding: Pre-computed query embedding
            k: Number of results to return
            
        Returns:
            List of relevant contexts with metadata and scores
        """
        try:
            logger.info(f"Processing query with custom embedding: {query_text}")
            
            contexts = self.retriever.get_relevant_contexts(
                query_text=query_text,
                query_embedding=query_embedding,
                k=k
            )
            
            logger.info(f"Retrieved {len(contexts)} contexts")
            return contexts
            
        except Exception as e:
            logger.error(f"Error in RAG query with custom embedding: {str(e)}")
            return []
        
    def get_pipeline_info(self) -> Dict:
            """
            Get information about the pipeline components.

            Returns:
                Dictionary with pipeline information
            """
            try:
                info = {
                    'embedding_service': self.embedding_service.get_model_info(),
                    'retriever': {
                        'type': 'HybridRetriever',
                        'vector_weight': self.retriever.vector_weight,
                        'keyword_weight': self.retriever.keyword_weight
                    },
                    'vector_database': self.vector_db_service.get_collection_info() if hasattr(self.vector_db_service, 'get_collection_info') else 'Available'
                }
                return info
            except Exception as e:
                logger.error(f"Error getting pipeline info: {str(e)}")
                return {}    

    def update_retriever_weights(self, vector_weight: float, keyword_weight: float) -> bool:
            """
            Update hybrid retriever weights.
            
            Args:
                vector_weight: Weight for vector similarity
                keyword_weight: Weight for keyword matching
                
            Returns:
                bool: True if successful
            """
            try:
                self.retriever.update_weights(vector_weight, keyword_weight)
                logger.info(f"Updated retriever weights: vector={vector_weight}, keyword={keyword_weight}")
                return True
            except Exception as e:
                logger.error(f"Error updating retriever weights: {str(e)}")
                return False    


def create_rag_pipeline(config: Optional[Dict] = None) -> RAGPipeline:
    """
    Factory function to create a configured RAG pipeline.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured RAGPipeline instance
    """
    try:
        pipeline = RAGPipeline(config=config)
        logger.info("RAG Pipeline created successfully via factory")
        return pipeline
    except Exception as e:
        logger.error(f"Error creating RAG Pipeline: {str(e)}")
        raise

def create_hybrid_retriever(config: Optional[Dict] = None) -> Retriever:
    """
    Fixed factory function to create a configured HybridRetriever instance.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured HybridRetriever instance
    """
    try:
        # Create services with proper dependency order
        query_embedding_service = QueryEmbeddingService()
        vector_db_service = VectorDatabaseService()
        
        # Create retriever with proper dependency injection
        retriever = Retriever(
            vector_db_service=vector_db_service,
            query_embedding_service=query_embedding_service,
            config=config
        )
        
        logger.info("HybridRetriever created successfully with proper dependency injection")
        return retriever
        
    except Exception as e:
        logger.error(f"Error creating HybridRetriever: {str(e)}")
        raise


if __name__ == "__main__":
    """Example of how to use the fixed RAG Pipeline."""

    try:
        pipeline = create_rag_pipeline()

        info = pipeline.get_pipeline_info()
        print("Pipeline Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")

        query = "Was ist das Bruttoinlandsprodukt und wie wird es berechnet?"
        results = pipeline.query(query, k=3)

        print(f"\nQuery: {query}")
        print(f"Retrieved {len(results)} contexts")

        for i, ctx in enumerate(results, 1):
            print(f"\nContext {i}:")
            print(f"  Score: {ctx.get('score')}")
            print(f"  Text: {ctx.get('context', '')[:200]}...")

        print("\nTesting weight updates...")
        success = pipeline.update_retriever_weights(vector_weight=0.85, keyword_weight=0.15)
        print(f"Weight update success: {success}")

        print("\n" + "="*50)
        print("✓ Fixed RAG Pipeline test completed successfully!")

    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()