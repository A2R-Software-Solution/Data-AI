import logging
import sys
from typing import Optional
from pythonjsonlogger import jsonlogger

def configure_logging(
    level: str = "INFO",
    service_name: str = "a2r-rag-api",
    version: str = "1.0.0"
) -> None:
    """
    Configure structured JSON logging for Google Cloud Logging.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        service_name: Name of the service for log correlation
        version: API version for tracking
    """
    
    # Create JSON formatter for structured logging
    formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(funcName)s %(lineno)d %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        static_fields={
            "service": service_name,
            "version": version,
            "environment": "production"
        }
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level.upper())
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add stdout handler for Cloud Run
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    stdout_handler.setLevel(level.upper())
    root_logger.addHandler(stdout_handler)
    
    # Configure specific loggers to reduce noise
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    
    logging.info(
        "Logging configured",
        extra={
            "log_level": level,
            "service": service_name,
            "version": version
        }
    )

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Usually __name__ from the calling module
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)

class RequestLogger:
    """Context manager for request-specific logging with correlation IDs."""
    
    def __init__(self, request_id: Optional[str] = None, user_id: Optional[str] = None):
        self.request_id = request_id
        self.user_id = user_id
        self.logger = get_logger("request")
    
    def info(self, message: str, **kwargs):
        """Log info with request context."""
        extra = self._get_extra(**kwargs)
        self.logger.info(message, extra=extra)
    
    def warning(self, message: str, **kwargs):
        """Log warning with request context."""
        extra = self._get_extra(**kwargs)
        self.logger.warning(message, extra=extra)
    
    def error(self, message: str, **kwargs):
        """Log error with request context."""
        extra = self._get_extra(**kwargs)
        self.logger.error(message, extra=extra)
    
    def _get_extra(self, **kwargs):
        """Build extra fields for logging."""
        extra = kwargs.copy()
        if self.request_id:
            extra["request_id"] = self.request_id
        if self.user_id:
            extra["user_id"] = self.user_id
        return extra













# import logging
# import sys
# from typing import Optional
# from pythonjsonlogger import jsonlogger

# def configure_logging(
#     level: str = "INFO",
#     service_name: str = "a2r-rag-api",
#     version: str = "1.0.0"
# ) -> None:
#     """
#     Configure structured JSON logging for Google Cloud Logging.
    
#     Args:
#         level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
#         service_name: Name of the service for log correlation
#         version: API version for tracking
#     """
    
#     # Create JSON formatter for structured logging
#     formatter = jsonlogger.JsonFormatter(
#         fmt="%(asctime)s %(levelname)s %(name)s %(funcName)s %(lineno)d %(message)s",
#         datefmt="%Y-%m-%dT%H:%M:%S",
#         static_fields={
#             "service": service_name,
#             "version": version,
#             "environment": "production"
#         }
#     )
    
#     # Configure root logger
#     root_logger = logging.getLogger()
#     root_logger.setLevel(level.upper())
    
#     # Remove existing handlers to avoid duplicates
#     for handler in root_logger.handlers[:]:
#         root_logger.removeHandler(handler)
    
#     # Add stdout handler for Cloud Run
#     stdout_handler = logging.StreamHandler(sys.stdout)
#     stdout_handler.setFormatter(formatter)
#     stdout_handler.setLevel(level.upper())
#     root_logger.addHandler(stdout_handler)
    
#     # Configure specific loggers to reduce noise
#     logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
#     logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
#     logging.getLogger("transformers").setLevel(logging.WARNING)
#     logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    
#     logging.info(
#         "Logging configured",
#         extra={
#             "log_level": level,
#             "service": service_name,
#             "version": version
#         }
#     )

# def get_logger(name: str) -> logging.Logger:
#     """
#     Get a logger instance for a specific module.
    
#     Args:
#         name: Usually __name__ from the calling module
        
#     Returns:
#         Configured logger instance
#     """
#     return logging.getLogger(name)

# class RequestLogger:
#     """Context manager for request-specific logging with correlation IDs."""
    
#     def __init__(self, request_id: Optional[str] = None, user_id: Optional[str] = None):
#         self.request_id = request_id
#         self.user_id = user_id
#         self.logger = get_logger("request")
    
#     def info(self, message: str, **kwargs):
#         """Log info with request context."""
#         extra = self._get_extra(**kwargs)
#         self.logger.info(message, extra=extra)
    
#     def warning(self, message: str, **kwargs):
#         """Log warning with request context."""
#         extra = self._get_extra(**kwargs)
#         self.logger.warning(message, extra=extra)
    
#     def error(self, message: str, **kwargs):
#         """Log error with request context."""
#         extra = self._get_extra(**kwargs)
#         self.logger.error(message, extra=extra)
    
#     def _get_extra(self, **kwargs):
#         """Build extra fields for logging."""
#         extra = kwargs.copy()
#         if self.request_id:
#             extra["request_id"] = self.request_id
#         if self.user_id:
#             extra["user_id"] = self.user_id
#         return extra