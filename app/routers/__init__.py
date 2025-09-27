"""
A2R RAG API Routers Module

This module contains the FastAPI route handlers:
- health: Health check endpoints for monitoring
- query: Main RAG query endpoints
"""

from .health import router as health_router
from .query import router as query_router

__all__ = [
    "health_router",
    "query_router"
]