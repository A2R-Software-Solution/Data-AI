from typing import Optional, Dict, Any
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from ..core.config import settings
from ..core.logging import get_logger

logger = get_logger(__name__)

class LLMClient:
    """Ollama LLM client for generating responses."""
    
    def __init__(self):
        self._llm: Optional[ChatOllama] = None
        self._chain: Optional[LLMChain] = None
        self._prompt_template = self._get_prompt_template()
        self._initialize()
    
    def _get_prompt_template(self) -> str:
        """Get the QA prompt template."""
        return """You are a helpful assistant for A2R Software Solutions.
Use ONLY the context provided below to answer the question.
If you cannot answer using the context, reply with: "I don't know."

Context:
{context}

Question:
{question}

Answer:"""
    
    def _initialize(self):
        """Initialize Ollama client and chain."""
        try:
            logger.info("Initializing Ollama LLM client", extra={
                "base_url": settings.OLLAMA_BASE_URL,
                "model": settings.OLLAMA_MODEL
            })
            
            # Initialize Ollama client
            self._llm = ChatOllama(
                model=settings.OLLAMA_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=0.1,  # Low temperature for consistent responses
                timeout=30.0,     # 30 second timeout
            )
            
            # Create prompt template
            prompt = PromptTemplate(
                template=self._prompt_template,
                input_variables=["context", "question"]
            )
            
            # Create chain
            self._chain = LLMChain(
                llm=self._llm,
                prompt=prompt,
                verbose=False
            )
            
            logger.info("LLM client initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize LLM client", extra={
                "error": str(e),
                "error_type": type(e).__name__
            })
            raise
    
    def generate_answer(
        self, 
        question: str, 
        context: str,
        **kwargs
    ) -> str:
        """
        Generate an answer using the LLM.
        
        Args:
            question: User's question
            context: Retrieved context from vector search
            **kwargs: Additional parameters for the LLM
            
        Returns:
            Generated answer
        """
        if not question.strip():
            logger.warning("Empty question provided to LLM")
            return "Please provide a valid question."
        
        try:
            logger.info("Generating LLM response", extra={
                "question_length": len(question),
                "context_length": len(context),
                "has_context": bool(context.strip())
            })
            
            # Truncate inputs if they're too long
            question = question[:settings.MAX_QUERY_LENGTH]
            context = context[:8000]  # Reasonable context limit
            
            # Generate response
            result = self._chain.invoke({
                "question": question,
                "context": context,
                **kwargs
            })
            
            # Extract answer from result
            if isinstance(result, dict):
                answer = result.get("text", "").strip()
            else:
                answer = str(result).strip()
            
            if not answer:
                answer = "I couldn't generate a response. Please try rephrasing your question."
            
            logger.info("LLM response generated", extra={
                "answer_length": len(answer),
                "question_preview": question[:50]
            })
            
            return answer
            
        except Exception as e:
            logger.error("LLM generation failed", extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "question_preview": question[:50]
            })
            return "I'm sorry, I encountered an error while processing your question. Please try again."
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on LLM service.
        
        Returns:
            Health status information
        """
        try:
            # Test with a simple question
            test_answer = self.generate_answer(
                question="What is AI?",
                context="AI stands for Artificial Intelligence."
            )
            
            return {
                "status": "healthy",
                "model": settings.OLLAMA_MODEL,
                "base_url": settings.OLLAMA_BASE_URL,
                "test_response_length": len(test_answer),
                "llm_available": self._llm is not None
            }
            
        except Exception as e:
            logger.error("LLM health check failed", extra={
                "error": str(e),
                "error_type": type(e).__name__
            })
            return {
                "status": "unhealthy",
                "error": str(e),
                "error_type": type(e).__name__,
                "model": settings.OLLAMA_MODEL,
                "base_url": settings.OLLAMA_BASE_URL
            }
    
    def update_prompt_template(self, new_template: str):
        """
        Update the prompt template and reinitialize the chain.
        
        Args:
            new_template: New prompt template string
        """
        try:
            self._prompt_template = new_template
            
            # Recreate prompt and chain
            prompt = PromptTemplate(
                template=self._prompt_template,
                input_variables=["context", "question"]
            )
            
            self._chain = LLMChain(
                llm=self._llm,
                prompt=prompt,
                verbose=False
            )
            
            logger.info("Prompt template updated successfully")
            
        except Exception as e:
            logger.error("Failed to update prompt template", extra={
                "error": str(e),
                "error_type": type(e).__name__
            })
            raise

# Global LLM client instance
llm_client = LLMClient()