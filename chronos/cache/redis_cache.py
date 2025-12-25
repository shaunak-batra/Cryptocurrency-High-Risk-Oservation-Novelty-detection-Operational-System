"""
Redis Cache Layer for CHRONOS

Provides caching for predictions and embeddings to achieve <50ms latency.
"""

import pickle
import hashlib
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class RedisCache:
    """Redis caching layer for CHRONOS predictions."""
    
    def __init__(self, redis_uri: str = "redis://localhost:6379", ttl: int = 3600):
        """
        Initialize Redis cache.
        
        Parameters
        ----------
        redis_uri : str
            Redis connection URI
        ttl : int
            Default TTL in seconds (1 hour)
        """
        self.ttl = ttl
        self.redis = None
        
        try:
            import redis
            self.redis = redis.from_url(redis_uri)
            self.redis.ping()
            logger.info(f"Connected to Redis at {redis_uri}")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Caching disabled.")
            self.redis = None
    
    @property
    def is_connected(self) -> bool:
        """Check if Redis is connected."""
        return self.redis is not None
    
    def _make_key(self, identifier: str, prefix: str = "pred") -> str:
        """Generate cache key."""
        return f"chronos:{prefix}:{identifier}"
    
    def get_prediction(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached prediction.
        
        Parameters
        ----------
        transaction_id : str
            Transaction identifier
            
        Returns
        -------
        dict or None
            Cached prediction or None if not found
        """
        if not self.redis:
            return None
            
        try:
            key = self._make_key(transaction_id, "pred")
            cached = self.redis.get(key)
            if cached:
                return pickle.loads(cached)
        except Exception as e:
            logger.error(f"Redis get error: {e}")
        return None
    
    def set_prediction(self, transaction_id: str, prediction: Dict[str, Any]):
        """
        Cache prediction result.
        
        Parameters
        ----------
        transaction_id : str
            Transaction identifier
        prediction : dict
            Prediction result to cache
        """
        if not self.redis:
            return
            
        try:
            key = self._make_key(transaction_id, "pred")
            self.redis.setex(key, self.ttl, pickle.dumps(prediction))
        except Exception as e:
            logger.error(f"Redis set error: {e}")
    
    def get_embedding(self, node_id: str) -> Optional[Any]:
        """
        Retrieve cached node embedding.
        
        Parameters
        ----------
        node_id : str
            Node identifier
            
        Returns
        -------
        numpy.ndarray or None
            Cached embedding or None
        """
        if not self.redis:
            return None
            
        try:
            key = self._make_key(node_id, "emb")
            cached = self.redis.get(key)
            if cached:
                return pickle.loads(cached)
        except Exception as e:
            logger.error(f"Redis get embedding error: {e}")
        return None
    
    def set_embedding(self, node_id: str, embedding: Any, ttl: Optional[int] = None):
        """
        Cache node embedding.
        
        Parameters
        ----------
        node_id : str
            Node identifier
        embedding : numpy.ndarray
            Embedding vector
        ttl : int, optional
            Custom TTL (uses default if not specified)
        """
        if not self.redis:
            return
            
        try:
            key = self._make_key(node_id, "emb")
            self.redis.setex(key, ttl or self.ttl, pickle.dumps(embedding))
        except Exception as e:
            logger.error(f"Redis set embedding error: {e}")
    
    def invalidate(self, transaction_id: str):
        """
        Invalidate cached prediction.
        
        Parameters
        ----------
        transaction_id : str
            Transaction identifier to invalidate
        """
        if not self.redis:
            return
            
        try:
            key = self._make_key(transaction_id, "pred")
            self.redis.delete(key)
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns
        -------
        dict
            Cache statistics
        """
        if not self.redis:
            return {"status": "disconnected"}
            
        try:
            info = self.redis.info("stats")
            return {
                "status": "connected",
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": info.get("keyspace_hits", 0) / max(
                    info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1
                )
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}


# Singleton instance
_cache_instance = None


def get_cache(redis_uri: Optional[str] = None) -> RedisCache:
    """Get or create cache instance."""
    global _cache_instance
    if _cache_instance is None:
        import os
        uri = redis_uri or os.getenv("REDIS_URI", "redis://localhost:6379")
        _cache_instance = RedisCache(uri)
    return _cache_instance
