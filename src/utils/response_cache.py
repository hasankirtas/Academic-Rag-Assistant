"""
Response Cache System for Performance Optimization

Caches LLM responses to avoid recomputation and improve response times.
"""

import hashlib
import json
import os
import time
from typing import Dict, Any, Optional
import logging
from pathlib import Path

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class ResponseCache:
    """
    Simple file-based cache for LLM responses to improve performance.
    """
    
    def __init__(self, cache_dir: str = "./cache/responses", max_size: int = 500, ttl_hours: int = 24):
        """
        Initialize response cache.
        
        Args:
            cache_dir: Directory to store cache files
            max_size: Maximum number of cached responses
            ttl_hours: Time to live in hours
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self.ttl_seconds = ttl_hours * 3600
        self.cache_file = self.cache_dir / "response_cache.json"
        
        # Load existing cache
        self.cache = self._load_cache()
        
        logger.info(f"Response cache initialized with {len(self.cache)} entries")
    
    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load response cache: {e}")
        return {}
    
    def _save_cache(self):
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save response cache: {e}")
    
    def _get_cache_key(self, query: str, contexts: list, model_name: str) -> str:
        """Generate cache key for query, contexts, and model."""
        # Create a simplified context representation for caching
        context_texts = [ctx.get('text', '')[:200] for ctx in contexts[:3]]  # Only first 3 contexts
        content = f"{query}_{'_'.join(context_texts)}_{model_name}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def get(self, query: str, contexts: list, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get cached response for query.
        
        Args:
            query: User query
            contexts: List of context dictionaries
            model_name: Model name used for generation
            
        Returns:
            Cached response or None if not found/expired
        """
        cache_key = self._get_cache_key(query, contexts, model_name)
        
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            
            # Check if entry is expired
            if time.time() - entry['created_at'] > self.ttl_seconds:
                del self.cache[cache_key]
                return None
            
            # Update access time
            entry['last_accessed'] = time.time()
            logger.debug(f"Cache hit for query: {query[:50]}...")
            return entry['response']
        
        return None
    
    def put(self, query: str, contexts: list, model_name: str, response: Dict[str, Any]):
        """
        Cache response for query.
        
        Args:
            query: User query
            contexts: List of context dictionaries
            model_name: Model name used for generation
            response: Response dictionary to cache
        """
        cache_key = self._get_cache_key(query, contexts, model_name)
        
        # Add to cache
        self.cache[cache_key] = {
            'query': query[:100],  # Store first 100 chars for debugging
            'model_name': model_name,
            'created_at': time.time(),
            'last_accessed': time.time(),
            'response': response
        }
        
        # Cleanup if cache is too large
        if len(self.cache) > self.max_size:
            self._cleanup_cache()
        
        # Save cache periodically
        if len(self.cache) % 5 == 0:
            self._save_cache()
    
    def _cleanup_cache(self):
        """Remove least recently used entries."""
        if len(self.cache) <= self.max_size:
            return
        
        # Sort by last accessed time
        sorted_items = sorted(
            self.cache.items(),
            key=lambda x: x[1].get('last_accessed', 0)
        )
        
        # Remove oldest 20% of entries
        remove_count = len(self.cache) - int(self.max_size * 0.8)
        for cache_key, _ in sorted_items[:remove_count]:
            del self.cache[cache_key]
        
        logger.info(f"Cleaned up response cache, removed {remove_count} entries")
    
    def clear(self):
        """Clear all cached responses."""
        self.cache.clear()
        self._save_cache()
        logger.info("Response cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        current_time = time.time()
        expired_count = sum(
            1 for entry in self.cache.values()
            if current_time - entry['created_at'] > self.ttl_seconds
        )
        
        return {
            'total_entries': len(self.cache),
            'expired_entries': expired_count,
            'active_entries': len(self.cache) - expired_count,
            'cache_file_size': self.cache_file.stat().st_size if self.cache_file.exists() else 0,
            'max_size': self.max_size,
            'ttl_hours': self.ttl_seconds / 3600
        }


# Global cache instance
_response_cache = None

def get_response_cache() -> ResponseCache:
    """Get global response cache instance."""
    global _response_cache
    if _response_cache is None:
        _response_cache = ResponseCache()
    return _response_cache

def clear_response_cache():
    """Clear global response cache."""
    global _response_cache
    if _response_cache is not None:
        _response_cache.clear()
