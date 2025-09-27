from pathlib import Path
from typing import Optional
from symspellpy import SymSpell, Verbosity

from ..core.logging import get_logger

logger = get_logger(__name__)

class SpellChecker:
    """SymSpell-based spell checker for query correction."""
    
    def __init__(self, dictionary_path: str = "frequency_dictionary_en_82_765.txt"):
        self._sym_spell: Optional[SymSpell] = None
        self._dictionary_path = dictionary_path
        self._initialize()
    
    def _initialize(self):
        """Initialize SymSpell with dictionary."""
        try:
            logger.info("Initializing SymSpell spell checker")
            
            self._sym_spell = SymSpell(
                max_dictionary_edit_distance=2,
                prefix_length=7
            )
            
            # Look for dictionary file in the app root
            dict_file = Path(self._dictionary_path)
            if not dict_file.exists():
                # Try in app root directory
                dict_file = Path(__file__).resolve().parents[2] / self._dictionary_path
            
            if dict_file.exists():
                success = self._sym_spell.load_dictionary(
                    str(dict_file), 
                    term_index=0, 
                    count_index=1
                )
                if success:
                    logger.info("SymSpell dictionary loaded successfully", extra={
                        "dictionary_path": str(dict_file),
                        "word_count": self._sym_spell.word_count
                    })
                else:
                    logger.warning("Failed to load SymSpell dictionary", extra={
                        "dictionary_path": str(dict_file)
                    })
            else:
                logger.warning("SymSpell dictionary file not found", extra={
                    "searched_paths": [
                        self._dictionary_path,
                        str(dict_file)
                    ]
                })
                
        except Exception as e:
            logger.error("Failed to initialize spell checker", extra={
                "error": str(e),
                "error_type": type(e).__name__
            })
            # Don't raise - spell checking is optional functionality
    
    def correct(self, text: str, max_edit_distance: int = 2) -> str:
        """
        Correct spelling in the given text.
        
        Args:
            text: Input text to correct
            max_edit_distance: Maximum edit distance for suggestions
            
        Returns:
            Corrected text or original if no correction found
        """
        if not text or not text.strip():
            return text
        
        if not self._sym_spell:
            logger.debug("Spell checker not initialized, returning original text")
            return text
        
        try:
            # Get suggestions for the entire text
            suggestions = self._sym_spell.lookup(
                text, 
                Verbosity.CLOSEST, 
                max_edit_distance=max_edit_distance
            )
            
            if suggestions and len(suggestions) > 0:
                corrected = suggestions[0].term
                
                if corrected != text:
                    logger.info("Spell correction applied", extra={
                        "original": text,
                        "corrected": corrected,
                        "edit_distance": suggestions[0].distance
                    })
                    return corrected
            
            return text
            
        except Exception as e:
            logger.error("Spell correction failed", extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "input_text": text[:100]  # Log first 100 chars
            })
            return text  # Return original on error
    
    def get_suggestions(
        self, 
        text: str, 
        max_suggestions: int = 5,
        max_edit_distance: int = 2
    ) -> list:
        """
        Get multiple spelling suggestions for text.
        
        Args:
            text: Input text
            max_suggestions: Maximum number of suggestions to return
            max_edit_distance: Maximum edit distance for suggestions
            
        Returns:
            List of suggestion objects with term, distance, and count
        """
        if not text or not self._sym_spell:
            return []
        
        try:
            suggestions = self._sym_spell.lookup(
                text,
                Verbosity.ALL,
                max_edit_distance=max_edit_distance,
                max_suggestions=max_suggestions
            )
            
            return [{
                "term": suggestion.term,
                "distance": suggestion.distance,
                "count": suggestion.count
            } for suggestion in suggestions]
            
        except Exception as e:
            logger.error("Failed to get spelling suggestions", extra={
                "error": str(e),
                "input_text": text[:100]
            })
            return []
    
    def is_available(self) -> bool:
        """Check if spell checker is available and ready."""
        return self._sym_spell is not None and self._sym_spell.word_count > 0
    
    def health_check(self) -> dict:
        """Perform health check on spell checker."""
        try:
            if not self._sym_spell:
                return {
                    "status": "unavailable",
                    "message": "SymSpell not initialized"
                }
            
            # Test with a simple correction
            test_result = self.correct("teh")
            
            return {
                "status": "healthy",
                "word_count": self._sym_spell.word_count,
                "test_correction": {
                    "input": "teh",
                    "output": test_result
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

# Global spell checker instance
spell_checker = SpellChecker()