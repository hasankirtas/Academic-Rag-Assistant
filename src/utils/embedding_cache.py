"""
Embedding Cache System for Performance Optimization

Caches query embeddings to avoid recomputation and improve response times.
"""

import hashlib
import json
import os
import pickle
from typing import List, Optional, Dict, Any
import logging
from pathlib import Path

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class EmbeddingCache:
    """
    Simple file-based cache for query embeddings to improve performance.
    """
    
    def __init__(self, cache_dir: str = "./cache/embeddings", max_size: int = 1000):
        """
        Initialize embedding cache.
        
        Args:
            cache_dir: Directory to store cache files
            max_size: Maximum number of cached embeddings
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self.cache_file = self.cache_dir / "embedding_cache.pkl"
        self.index_file = self.cache_dir / "cache_index.json"
        
        # Load existing cache
        self.cache = self._load_cache()
        self.index = self._load_index()
        
        logger.info(f"Embedding cache initialized with {len(self.cache)} entries")
    
    def _load_cache(self) -> Dict[str, List[float]]:
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return {}
    
    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load cache index from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
        return {}
    
    def _save_cache(self):
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            with open(self.index_file, 'w') as f:
                json.dump(self.index, f)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def _get_cache_key(self, text: str, model_name: str) -> str:
        """Generate cache key for text and model."""
        content = f"{text}_{model_name}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def get(self, text: str, model_name: str) -> Optional[List[float]]:
        """
        Get cached embedding for text.
        
        Args:
            text: Input text
            model_name: Model name used for embedding
            
        Returns:
            Cached embedding or None if not found
        """
        cache_key = self._get_cache_key(text, model_name)
        
        if cache_key in self.cache:
            # Update access time
            self.index[cache_key]['last_accessed'] = self._get_timestamp()
            return self.cache[cache_key]
        
        return None
    
    def put(self, text: str, model_name: str, embedding: List[float]):
        """
        Cache embedding for text.
        
        Args:
            text: Input text
            model_name: Model name used for embedding
            embedding: Embedding vector
        """
        cache_key = self._get_cache_key(text, model_name)
        
        # Add to cache
        self.cache[cache_key] = embedding
        self.index[cache_key] = {
            'text': text[:100],  # Store first 100 chars for debugging
            'model_name': model_name,
            'created_at': self._get_timestamp(),
            'last_accessed': self._get_timestamp(),
            'size': len(embedding)
        }
        
        # Cleanup if cache is too large
        if len(self.cache) > self.max_size:
            self._cleanup_cache()
        
        # Save cache periodically
        if len(self.cache) % 10 == 0:
            self._save_cache()
    
    def _cleanup_cache(self):
        """Remove least recently used entries."""
        if len(self.cache) <= self.max_size:
            return
        
        # Sort by last accessed time
        sorted_items = sorted(
            self.index.items(),
            key=lambda x: x[1].get('last_accessed', 0)
        )
        
        # Remove oldest 20% of entries
        remove_count = len(self.cache) - int(self.max_size * 0.8)
        for cache_key, _ in sorted_items[:remove_count]:
            self.cache.pop(cache_key, None)
            self.index.pop(cache_key, None)
        
        logger.info(f"Cleaned up cache, removed {remove_count} entries")
    
    def _get_timestamp(self) -> float:
        """Get current timestamp."""
        import time
        return time.time()
    
    def clear(self):
        """Clear all cached embeddings."""
        self.cache.clear()
        self.index.clear()
        self._save_cache()
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size = sum(entry.get('size', 0) for entry in self.index.values())
        return {
            'total_entries': len(self.cache),
            'total_size_bytes': total_size,
            'cache_file_size': self.cache_file.stat().st_size if self.cache_file.exists() else 0,
            'max_size': self.max_size
        }


# Global cache instance
_embedding_cache = None

def get_embedding_cache() -> EmbeddingCache:
    """Get global embedding cache instance."""
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = EmbeddingCache()
    return _embedding_cache

def clear_embedding_cache():
    """Clear global embedding cache."""
    global _embedding_cache
    if _embedding_cache is not None:
        _embedding_cache.clear()
