from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import time
import uuid

from ..services import retriever, llm_client, spell_checker
from ..core.config import settings
from ..core.logging import get_logger, RequestLogger

logger = get_logger(__name__)
router = APIRouter(prefix="/query", tags=["query"])

class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    question: str = Field(..., min_length=1, max_length=settings.MAX_QUERY_LENGTH)
    k: Optional[int] = Field(default=None, ge=1, le=settings.MAX_SEARCH_RESULTS)
    min_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    
    @validator('question')
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError('Question cannot be empty or whitespace only')
        return v.strip()

class SourceInfo(BaseModel):
    """Source information for retrieved documents."""
    file_name: str
    page_number: Any  # Can be int or "N/A"
    chunk_index: Any  # Can be int or "N/A" 
    source_path: str
    similarity_score: float

class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    request_id: str
    question: str
    corrected_question: Optional[str] = None
    answer: str
    sources: List[SourceInfo]
    metadata: Dict[str, Any]

def generate_request_id() -> str:
    """Generate a unique request ID for tracking."""
    return str(uuid.uuid4())

@router.post("", response_model=QueryResponse)
def query_rag(request: QueryRequest) -> QueryResponse:
    """
    Main RAG query endpoint.
    
    Processes a user question through:
    1. Spell correction
    2. Vector search in MongoDB Atlas
    3. LLM response generation
    4. Source citation
    """
    request_id = generate_request_id()
    request_logger = RequestLogger(request_id=request_id)
    start_time = time.time()
    
    request_logger.info("RAG query started", 
        question_length=len(request.question),
        k=request.k,
        min_score=request.min_score
    )
    
    try:
        # Step 1: Spell correction
        corrected_question = None
        query_text = request.question
        
        if spell_checker.is_available():
            corrected_text = spell_checker.correct(request.question)
            if corrected_text != request.question:
                corrected_question = corrected_text
                query_text = corrected_text
                request_logger.info("Spell correction applied",
                    original=request.question,
                    corrected=corrected_text
                )
        
        # Step 2: Vector search
        search_k = request.k or settings.DEFAULT_SEARCH_RESULTS
        search_results = retriever.search(
            query_text, 
            k=search_k,
            min_score=request.min_score
        )
        
        request_logger.info("Vector search completed",
            results_found=len(search_results),
            search_k=search_k
        )
        
        if not search_results:
            request_logger.warning("No search results found")
            return QueryResponse(
                request_id=request_id,
                question=request.question,
                corrected_question=corrected_question,
                answer="I don't know. I couldn't find relevant information to answer your question.",
                sources=[],
                metadata={
                    "processing_time_ms": round((time.time() - start_time) * 1000, 2),
                    "search_results_count": 0,
                    "spell_corrected": corrected_question is not None
                }
            )
        
        # Step 3: Prepare context and generate answer
        context = "\n\n".join(result["text"] for result in search_results)
        
        answer = llm_client.generate_answer(
            question=query_text,
            context=context
        )
        
        request_logger.info("LLM response generated",
            answer_length=len(answer),
            context_length=len(context)
        )
        
        # Step 4: Format sources
        sources = []
        for result in search_results:
            metadata = result.get("metadata", {}) or {}
            sources.append(SourceInfo(
                file_name=metadata.get("file_name", "Unknown"),
                page_number=metadata.get("page_number", "N/A"),
                chunk_index=metadata.get("chunk_index", "N/A"),
                source_path=metadata.get("source", "Unknown"),
                similarity_score=round(result.get("score", 0.0), 4)
            ))
        
        # Step 5: Build response
        processing_time = round((time.time() - start_time) * 1000, 2)
        
        response = QueryResponse(
            request_id=request_id,
            question=request.question,
            corrected_question=corrected_question,
            answer=answer,
            sources=sources,
            metadata={
                "processing_time_ms": processing_time,
                "search_results_count": len(search_results),
                "spell_corrected": corrected_question is not None,
                "context_length": len(context),
                "answer_length": len(answer)
            }
        )
        
        request_logger.info("RAG query completed successfully",
            processing_time_ms=processing_time,
            sources_count=len(sources)
        )
        
        return response
        
    except Exception as e:
        processing_time = round((time.time() - start_time) * 1000, 2)
        
        request_logger.error("RAG query failed",
            error=str(e),
            error_type=type(e).__name__,
            processing_time_ms=processing_time
        )
        
        # Return user-friendly error
        raise HTTPException(
            status_code=500,
            detail={
                "message": "I encountered an error while processing your question. Please try again.",
                "request_id": request_id,
                "error_type": type(e).__name__,
                "processing_time_ms": processing_time
            }
        )

@router.get("/suggestions/{text}")
def get_spelling_suggestions(text: str) -> Dict[str, Any]:
    """
    Get spelling suggestions for a given text.
    Useful for auto-complete or suggestion features.
    """
    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text parameter cannot be empty"
        )
    
    if not spell_checker.is_available():
        raise HTTPException(
            status_code=503,
            detail="Spell checker is not available"
        )
    
    try:
        suggestions = spell_checker.get_suggestions(text, max_suggestions=5)
        
        return {
            "original_text": text,
            "suggestions": suggestions,
            "spell_checker_available": True
        }
        
    except Exception as e:
        logger.error("Failed to get spelling suggestions", extra={
            "error": str(e),
            "text": text[:100]
        })
        raise HTTPException(
            status_code=500,
            detail="Failed to generate spelling suggestions"
        )

@router.get("/search/{query}")
def search_only(
    query: str,
    k: int = settings.DEFAULT_SEARCH_RESULTS,
    min_score: float = settings.MIN_SIMILARITY_SCORE
) -> Dict[str, Any]:
    """
    Vector search only endpoint (no LLM generation).
    Useful for testing search functionality or building custom interfaces.
    """
    if not query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query parameter cannot be empty"
        )
    
    if k > settings.MAX_SEARCH_RESULTS:
        k = settings.MAX_SEARCH_RESULTS
    
    try:
        results = retriever.search(query, k=k, min_score=min_score)
        
        return {
            "query": query,
            "results_count": len(results),
            "results": [
                {
                    "text": result["text"],
                    "metadata": result.get("metadata", {}),
                    "similarity_score": round(result.get("score", 0.0), 4)
                }
                for result in results
            ],
            "parameters": {
                "k": k,
                "min_score": min_score
            }
        }
        
    except Exception as e:
        logger.error("Search failed", extra={
            "error": str(e),
            "query": query[:100]
        })
        raise HTTPException(
            status_code=500,
            detail="Search operation failed"
        )