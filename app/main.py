import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import time
import traceback

from .core.config import settings
from .core.logging import configure_logging, get_logger
from .routers import health_router, query_router

# Configure logging before any other imports
configure_logging(
    level=settings.LOG_LEVEL,
    service_name="a2r-rag-api",
    version=settings.API_VERSION
)

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting A2R RAG API", extra={
        "version": settings.API_VERSION,
        "environment": settings.ENVIRONMENT,
        "region": "us-east4",
        "mongo_db": settings.MONGO_DB,
        "ollama_model": settings.OLLAMA_MODEL
    })
    
    # Test critical services on startup
    try:
        from .services import retriever, llm_client
        
        # Quick health check to ensure services are working
        retriever_status = retriever.health_check()
        llm_status = llm_client.health_check()
        
        logger.info("Service health check on startup", extra={
            "retriever_status": retriever_status.get("status"),
            "llm_status": llm_status.get("status")
        })
        
        if retriever_status.get("status") != "healthy":
            logger.error("Retriever service unhealthy on startup", extra=retriever_status)
        
        if llm_status.get("status") != "healthy":
            logger.warning("LLM service unhealthy on startup", extra=llm_status)
            
    except Exception as e:
        logger.error("Failed to check services on startup", extra={
            "error": str(e),
            "error_type": type(e).__name__
        })
    
    yield
    
    # Shutdown
    logger.info("Shutting down A2R RAG API")
    
    try:
        from .services import retriever
        retriever.close()
        logger.info("MongoDB connection closed")
    except Exception as e:
        logger.error("Error during shutdown", extra={
            "error": str(e),
            "error_type": type(e).__name__
        })

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title=settings.API_TITLE,
        version=settings.API_VERSION,
        description="Production RAG API for A2R Software Solutions - us-east4 deployment",
        docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
        redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
        lifespan=lifespan
    )
    
    # Security Middleware - Trusted Hosts
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.TRUSTED_HOSTS
    )
    
    # CORS Middleware - Production Security
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=[
            "Accept",
            "Accept-Language", 
            "Content-Language",
            "Content-Type",
            "Authorization",
            "X-Request-ID",
            "X-API-Key"
        ],
        max_age=600  # Cache preflight for 10 minutes
    )
    
    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.time()
        request_id = request.headers.get("x-request-id", f"req-{int(start_time)}")
        
        # Log request
        logger.info("Request started", extra={
            "method": request.method,
            "url": str(request.url),
            "user_agent": request.headers.get("user-agent"),
            "request_id": request_id,
            "client_ip": request.client.host if request.client else None,
            "region": "us-east4"
        })
        
        # Process request
        try:
            response = await call_next(request)
            processing_time = time.time() - start_time
            
            # Log response
            logger.info("Request completed", extra={
                "method": request.method,
                "url": str(request.url),
                "status_code": response.status_code,
                "processing_time_ms": round(processing_time * 1000, 2),
                "request_id": request_id
            })
            
            # Add custom headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Processing-Time"] = str(round(processing_time * 1000, 2))
            response.headers["X-Region"] = "us-east4"
            response.headers["X-Service"] = "a2r-rag-api"
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            logger.error("Request failed", extra={
                "method": request.method,
                "url": str(request.url),
                "error": str(e),
                "error_type": type(e).__name__,
                "processing_time_ms": round(processing_time * 1000, 2),
                "request_id": request_id,
                "traceback": traceback.format_exc()
            })
            
            # Return generic error response
            return JSONResponse(
                status_code=500,
                content={
                    "detail": "Internal server error",
                    "request_id": request_id,
                    "error_type": type(e).__name__
                },
                headers={
                    "X-Request-ID": request_id,
                    "X-Region": "us-east4"
                }
            )
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        request_id = request.headers.get("x-request-id", "unknown")
        
        logger.error("Unhandled exception", extra={
            "error": str(exc),
            "error_type": type(exc).__name__,
            "url": str(request.url),
            "method": request.method,
            "request_id": request_id,
            "traceback": traceback.format_exc()
        })
        
        return JSONResponse(
            status_code=500,
            content={
                "detail": "An unexpected error occurred",
                "request_id": request_id,
                "error_type": type(exc).__name__,
                "region": "us-east4"
            },
            headers={
                "X-Request-ID": request_id,
                "X-Region": "us-east4"
            }
        )
    
    # Include routers
    app.include_router(health_router)
    app.include_router(query_router)
    
    # Root endpoint
    @app.get("/")
    def root():
        """API root endpoint."""
        return {
            "service": "A2R RAG API",
            "version": settings.API_VERSION,
            "status": "running",
            "region": "us-east4",
            "environment": settings.ENVIRONMENT,
            "docs_url": "/docs" if settings.ENVIRONMENT != "production" else None,
            "mongodb_connected": bool(settings.MONGO_URI),
            "langsmith_enabled": bool(settings.LANGCHAIN_API_KEY)
        }
    
    return app

# Create the app instance
app = create_app()

# For Cloud Run compatibility
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", settings.PORT))
    
    logger.info("Starting uvicorn server", extra={
        "host": "0.0.0.0",
        "port": port,
        "region": "us-east4"
    })
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        log_config=None,  # Use our custom logging
        access_log=False  # We handle access logs in middleware
    )