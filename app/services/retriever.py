from typing import List, Dict, Any, Optional
from pymongo import MongoClient
from pymongo.collection import Collection
from langchain_huggingface import HuggingFaceEmbeddings

from ..core.config import settings
from ..core.logging import get_logger

logger = get_logger(__name__)

class MongoRetriever:
    """MongoDB Atlas vector search retriever for RAG."""
    
    def __init__(self):
        self._client: Optional[MongoClient] = None
        self._collection: Optional[Collection] = None
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._initialize()
    
    def _initialize(self):
        """Initialize MongoDB connection and embeddings model."""
        try:
            logger.info("Initializing MongoDB Atlas connection", extra={
                "mongo_db": settings.MONGO_DB,
                "mongo_collection": settings.MONGO_COLLECTION,
                "vector_index": settings.VECTOR_INDEX
            })
            
            # Initialize MongoDB client
            self._client = MongoClient(settings.MONGO_URI)
            
            # Test connection
            self._client.admin.command('ping')
            logger.info("MongoDB Atlas connection established")
            
            # Get collection
            self._collection = self._client[settings.MONGO_DB][settings.MONGO_COLLECTION]
            
            # Initialize embeddings model
            logger.info("Loading embeddings model", extra={
                "model": settings.EMBEDDING_MODEL
            })
            self._embeddings = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL
            )
            
            logger.info("MongoRetriever initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize MongoRetriever", extra={
                "error": str(e),
                "error_type": type(e).__name__
            })
            raise
    
    def search(
        self, 
        query: str, 
        k: int = None,
        min_score: float = None
    ) -> List[Dict[str, Any]]:
        """
        Perform vector search using MongoDB Atlas $vectorSearch.
        
        Args:
            query: Search query text
            k: Number of results to return (default from settings)
            min_score: Minimum similarity score threshold
            
        Returns:
            List of search results with text, metadata, and scores
        """
        if not query.strip():
            logger.warning("Empty query provided to search")
            return []
        
        # Use defaults from settings
        k = k or settings.DEFAULT_SEARCH_RESULTS
        k = min(k, settings.MAX_SEARCH_RESULTS)  # Enforce max limit
        min_score = min_score or settings.MIN_SIMILARITY_SCORE
        
        try:
            logger.info("Performing vector search", extra={
                "query_length": len(query),
                "k": k,
                "min_score": min_score
            })
            
            # Generate query embedding
            query_vector = self._embeddings.embed_query(query)
            
            # Build aggregation pipeline
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": settings.VECTOR_INDEX,
                        "queryVector": query_vector,
                        "path": "embedding",
                        "numCandidates": settings.VECTOR_SEARCH_CANDIDATES,
                        "limit": k
                    }
                },
                {
                    "$addFields": {
                        "score": {"$meta": "vectorSearchScore"}
                    }
                },
                {
                    "$match": {
                        "score": {"$gte": min_score}
                    }
                },
                {
                    "$project": {
                        "_id": 0,
                        "text": 1,
                        "metadata": 1,
                        "score": 1
                    }
                }
            ]
            
            # Execute search
            results = list(self._collection.aggregate(pipeline))
            
            logger.info("Vector search completed", extra={
                "results_count": len(results),
                "query_preview": query[:50]
            })
            
            return results
            
        except Exception as e:
            logger.error("Vector search failed", extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "query_preview": query[:50]
            })
            raise
    
    def get_context(self, query: str, k: int = None) -> str:
        """
        Get concatenated context from search results.
        
        Args:
            query: Search query
            k: Number of results to retrieve
            
        Returns:
            Concatenated text context for LLM
        """
        results = self.search(query, k)
        
        if not results:
            logger.warning("No search results found for context", extra={
                "query_preview": query[:50]
            })
            return ""
        
        context = "\n\n".join(result["text"] for result in results)
        
        logger.info("Context generated", extra={
            "context_length": len(context),
            "num_sources": len(results)
        })
        
        return context
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on MongoDB connection and collection.
        
        Returns:
            Health status information
        """
        try:
            # Test connection
            self._client.admin.command('ping')
            
            # Get collection stats
            doc_count = self._collection.estimated_document_count()
            
            # Test vector search (simple query)
            test_results = self.search("test", k=1)
            
            return {
                "status": "healthy",
                "document_count": doc_count,
                "vector_search_working": len(test_results) >= 0,
                "embeddings_loaded": self._embeddings is not None
            }
            
        except Exception as e:
            logger.error("Health check failed", extra={
                "error": str(e),
                "error_type": type(e).__name__
            })
            return {
                "status": "unhealthy",
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    def close(self):
        """Close MongoDB connection."""
        if self._client:
            self._client.close()
            logger.info("MongoDB connection closed")

# Global retriever instance
retriever = MongoRetriever()