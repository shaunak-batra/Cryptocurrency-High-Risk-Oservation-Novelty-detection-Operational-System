"""
CHRONOS API Security Middleware

Production security features: API key authentication and rate limiting.
"""

import os
import time
from typing import Optional
from collections import defaultdict

from fastapi import Request, HTTPException
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware


# ==========================================================================
# API KEY AUTHENTICATION
# ==========================================================================
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Load valid API keys from environment (comma-separated)
VALID_API_KEYS = set(
    key.strip() 
    for key in os.getenv("API_KEYS", "chronos-dev-key").split(",")
    if key.strip()
)

# Endpoints that don't require authentication
PUBLIC_ENDPOINTS = {"/", "/health", "/docs", "/redoc", "/openapi.json", "/metrics"}


async def verify_api_key(request: Request) -> Optional[str]:
    """
    Verify API key from request header.
    
    Returns API key if valid, raises HTTPException otherwise.
    """
    # Skip auth for public endpoints
    if request.url.path in PUBLIC_ENDPOINTS:
        return None
    
    # Skip auth if no keys configured (dev mode)
    if not VALID_API_KEYS or VALID_API_KEYS == {"chronos-dev-key"}:
        return None
    
    api_key = request.headers.get("X-API-Key")
    
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Include 'X-API-Key' header."
        )
    
    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key."
        )
    
    return api_key


# ==========================================================================
# RATE LIMITING
# ==========================================================================
class RateLimiter:
    """
    Simple in-memory rate limiter using sliding window.
    
    For production, use Redis-based rate limiting.
    """
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.window_size = 60  # seconds
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed for client."""
        now = time.time()
        window_start = now - self.window_size
        
        # Clean old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > window_start
        ]
        
        # Check limit
        if len(self.requests[client_id]) >= self.requests_per_minute:
            return False
        
        # Record request
        self.requests[client_id].append(now)
        return True
    
    def get_remaining(self, client_id: str) -> int:
        """Get remaining requests in current window."""
        now = time.time()
        window_start = now - self.window_size
        
        current_count = len([
            req_time for req_time in self.requests[client_id]
            if req_time > window_start
        ])
        
        return max(0, self.requests_per_minute - current_count)


# Global rate limiter
rate_limiter = RateLimiter(
    requests_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "100"))
)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce rate limiting."""
    
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks
        if request.url.path in {"/health", "/metrics"}:
            return await call_next(request)
        
        # Get client identifier (IP or API key)
        client_id = request.headers.get("X-API-Key") or request.client.host
        
        if not rate_limiter.is_allowed(client_id):
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "retry_after_seconds": 60
                },
                headers={"Retry-After": "60"}
            )
        
        # Add rate limit headers to response
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(rate_limiter.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(
            rate_limiter.get_remaining(client_id)
        )
        
        return response


# ==========================================================================
# SECURITY HEADERS
# ==========================================================================
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        return response


# Import for JSONResponse
from fastapi.responses import JSONResponse
