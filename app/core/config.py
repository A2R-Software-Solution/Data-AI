# from pydantic_settings import BaseSettings, SettingsConfigDict
# from typing import List, Optional

# class Settings(BaseSettings):
#     """
#     Production settings for A2R RAG API.
#     All sensitive values come from Google Secret Manager in production.
#     """
    
#     # CORS Configuration (Production only)
#     ALLOWED_ORIGINS: List[str] = [
#         "https://www.a2rsoftwaresolution.com",
#         "https://a2rsoftwaresolution.com"
#     ]
    
#     # MongoDB Atlas Configuration
#     MONGO_URI: str  # From Secret Manager in production
#     MONGO_DB: str = "rag_db"
#     MONGO_COLLECTION: str = "documents"
#     VECTOR_INDEX: str = "rag-chatbot-index"  # Fixed: removed the 's' from your env.txt
    
#     # Embeddings Configuration
#     EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
#     # Ollama Configuration (Remote service)
#     OLLAMA_BASE_URL: str  # e.g., http://ollama-runner.internal:11434
#     OLLAMA_MODEL: str = "mistral"
    
#     # LangChain / LangSmith Configuration
#     LANGCHAIN_TRACING_V2: str = "true"
#     LANGCHAIN_API_KEY: Optional[str] = None  # From Secret Manager
#     LANGCHAIN_PROJECT: str = "A2R-RAG"
    
#     # Application Configuration
#     LOG_LEVEL: str = "INFO"
#     API_TITLE: str = "A2R RAG API"
#     API_VERSION: str = "1.0.0"
    
#     # Cloud Run Configuration
#     PORT: int = 8080
    
#     # Rate Limiting & Performance
#     MAX_QUERY_LENGTH: int = 1000
#     DEFAULT_SEARCH_RESULTS: int = 4
#     MAX_SEARCH_RESULTS: int = 10
    
#     # Vector Search Configuration
#     VECTOR_SEARCH_CANDIDATES: int = 200
#     MIN_SIMILARITY_SCORE: float = 0.75
    
#     model_config = SettingsConfigDict(
#         env_file=".env",
#         env_file_encoding="utf-8",
#         case_sensitive=True
#     )

# # Global settings instance
# settings = Settings()



import os
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """
    Simple settings class without pydantic-settings dependency.
    Temporary fix for Python compatibility issues.
    """
    
    # CORS Configuration (Production only)
    ALLOWED_ORIGINS: List[str] = [
        "https://www.a2rsoftwaresolution.com",
        "https://a2rsoftwaresolution.com"
    ]
    
    # MongoDB Atlas Configuration
    MONGO_URI: str = os.getenv("MONGO_URI", "")
    MONGO_DB: str = os.getenv("MONGO_DB", "rag_db")
    MONGO_COLLECTION: str = os.getenv("MONGO_COLLECTION", "documents")
    VECTOR_INDEX: str = os.getenv("VECTOR_INDEX", "rag-chatbot-index")
    
    # Embeddings Configuration
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Ollama Configuration (Remote service)
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
    
    # Cloud Run Configuration
    PORT: int = int(os.getenv("PORT", "8080"))
    
    # Rate Limiting & Performance
    MAX_QUERY_LENGTH: int = int(os.getenv("MAX_QUERY_LENGTH", "1000"))
    DEFAULT_SEARCH_RESULTS: int = int(os.getenv("DEFAULT_SEARCH_RESULTS", "4"))
    MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", "10"))
    
    # Vector Search Configuration
    VECTOR_SEARCH_CANDIDATES: int = int(os.getenv("VECTOR_SEARCH_CANDIDATES", "200"))
    MIN_SIMILARITY_SCORE: float = float(os.getenv("MIN_SIMILARITY_SCORE", "0.75"))

# Global settings instance
settings = Settings()

# Validation
if not settings.MONGO_URI:
    raise ValueError("MONGO_URI environment variable is required")