import os
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """
    Production settings for A2R RAG API.
    Updated for us-east4 region and proper secret management.
    """
    
    # CORS Configuration (Production domains)
    ALLOWED_ORIGINS: List[str] = [
        "https://www.a2rsoftwaresolution.com",
        "https://a2rsoftwaresolution.com",
        "http://localhost:3000",  # For local development
        "http://localhost:8000"   # For local API testing
    ]
    
    # MongoDB Atlas Configuration
    MONGO_URI: str = os.getenv("MONGO_URI", "")
    MONGO_DB: str = os.getenv("MONGO_DB", "rag_db")
    MONGO_COLLECTION: str = os.getenv("MONGO_COLLECTION", "documents")
    VECTOR_INDEX: str = os.getenv("VECTOR_INDEX", "rag-chatbot-index")
    
    # Embeddings Configuration
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Ollama Configuration
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral")
    
    # LangChain / LangSmith Configuration
    LANGCHAIN_TRACING_V2: str = os.getenv("LANGCHAIN_TRACING_V2", "true")
    LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
    LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "A2R-RAG")
    
    # Application Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    API_TITLE: str = os.getenv("API_TITLE", "A2R RAG API")
    API_VERSION: str = os.getenv("API_VERSION", "1.0.0")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    # Cloud Run Configuration
    PORT: int = int(os.getenv("PORT", "8080"))
    
    # Rate Limiting & Performance
    MAX_QUERY_LENGTH: int = int(os.getenv("MAX_QUERY_LENGTH", "1000"))
    DEFAULT_SEARCH_RESULTS: int = int(os.getenv("DEFAULT_SEARCH_RESULTS", "4"))
    MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", "10"))
    
    # Vector Search Configuration
    VECTOR_SEARCH_CANDIDATES: int = int(os.getenv("VECTOR_SEARCH_CANDIDATES", "200"))
    MIN_SIMILARITY_SCORE: float = float(os.getenv("MIN_SIMILARITY_SCORE", "0.75"))
    
    # Security Configuration
    TRUSTED_HOSTS: List[str] = [
        "www.a2rsoftwaresolution.com",
        "a2rsoftwaresolution.com", 
        "*.run.app",  # Cloud Run domains
        "localhost",   # Local development
        "127.0.0.1"    # Local development
    ]

# Global settings instance
settings = Settings()

# Validation for production
if settings.ENVIRONMENT == "production":
    if not settings.MONGO_URI:
        raise ValueError("MONGO_URI environment variable is required for production")
    
    if not settings.LANGCHAIN_API_KEY:
        print("Warning: LANGCHAIN_API_KEY not set - LangSmith monitoring will be disabled")

# Development validation
if settings.ENVIRONMENT == "development" and not settings.MONGO_URI:
    print("Warning: MONGO_URI not set - using development defaults")