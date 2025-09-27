from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import time

from ..services import retriever, llm_client, spell_checker
from ..core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["health"])

@router.get("/healthz")
def basic_health_check() -> Dict[str, Any]:
    """
    Basic health check for Cloud Run health checks.
    Fast and lightweight - only checks if service is running.
    """
    return {
        "status": "healthy",
        "service": "a2r-rag-api",
        "timestamp": time.time()
    }

@router.get("/health/detailed")
def detailed_health_check() -> Dict[str, Any]:
    """
    Detailed health check that tests all service dependencies.
    Use this for monitoring and debugging.
    """
    start_time = time.time()
    health_status = {
        "status": "healthy",
        "service": "a2r-rag-api",
        "timestamp": start_time,
        "components": {}
    }
    
    overall_healthy = True
    
    # Check MongoDB Atlas / Retriever
    try:
        logger.info("Checking retriever health")
        retriever_health = retriever.health_check()
        health_status["components"]["retriever"] = retriever_health
        
        if retriever_health.get("status") != "healthy":
            overall_healthy = False
            
    except Exception as e:
        logger.error("Retriever health check failed", extra={
            "error": str(e),
            "error_type": type(e).__name__
        })
        health_status["components"]["retriever"] = {
            "status": "error",
            "error": str(e)
        }
        overall_healthy = False
    
    # Check LLM Client
    try:
        logger.info("Checking LLM client health")
        llm_health = llm_client.health_check()
        health_status["components"]["llm"] = llm_health
        
        if llm_health.get("status") != "healthy":
            overall_healthy = False
            
    except Exception as e:
        logger.error("LLM health check failed", extra={
            "error": str(e),
            "error_type": type(e).__name__
        })
        health_status["components"]["llm"] = {
            "status": "error", 
            "error": str(e)
        }
        overall_healthy = False
    
    # Check Spell Checker (optional component)
    try:
        logger.info("Checking spell checker health")
        spell_health = spell_checker.health_check()
        health_status["components"]["spell_checker"] = spell_health
        
        # Spell checker is optional, so don't fail overall health
        
    except Exception as e:
        logger.warning("Spell checker health check failed", extra={
            "error": str(e),
            "error_type": type(e).__name__
        })
        health_status["components"]["spell_checker"] = {
            "status": "warning",
            "error": str(e)
        }
    
    # Set overall status
    if not overall_healthy:
        health_status["status"] = "unhealthy"
        
    # Add response time
    health_status["response_time_ms"] = round((time.time() - start_time) * 1000, 2)
    
    logger.info("Health check completed", extra={
        "overall_status": health_status["status"],
        "response_time_ms": health_status["response_time_ms"]
    })
    
    # Return 503 if unhealthy
    if not overall_healthy:
        raise HTTPException(
            status_code=503, 
            detail=health_status
        )
    
    return health_status