"""
Comprehensive error handling module for the PDF Structure Extractor.

This module provides centralized error handling, custom exceptions, graceful degradation,
and timeout management for all components of the system.
"""

import logging
import signal
import time
import traceback
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from contextlib import contextmanager
from functools import wraps

try:
    from .models import DocumentStructure, Heading
    from .config import config
except ImportError:
    from models import DocumentStructure, Heading
    from config import config


logger = logging.getLogger(__name__)


# Custom Exception Classes
class PDFStructureExtractorError(Exception):
    """Base exception for PDF Structure Extractor errors."""
    
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.details = details or {}
        self.timestamp = time.time()


class PDFProcessingError(PDFStructureExtractorError):
    """Exception raised when PDF processing fails."""
    pass


class ModelLoadingError(PDFStructureExtractorError):
    """Exception raised when model loading fails."""
    pass


class ClassificationError(PDFStructureExtractorError):
    """Exception raised when text classification fails."""
    pass


class FeatureExtractionError(PDFStructureExtractorError):
    """Exception raised when feature extraction fails."""
    pass


class StructureAnalysisError(PDFStructureExtractorError):
    """Exception raised when structure analysis fails."""
    pass


class OutputGenerationError(PDFStructureExtractorError):
    """Exception raised when JSON output generation fails."""
    pass


class TimeoutError(PDFStructureExtractorError):
    """Exception raised when processing exceeds timeout limit."""
    pass


class ValidationError(PDFStructureExtractorError):
    """Exception raised when data validation fails."""
    pass


# Error Handler Classes
class ErrorHandler:
    """Central error handler with graceful degradation capabilities."""
    
    def __init__(self):
        self.error_counts = {}
        self.fallback_enabled = True
        self.max_retries = 3
        self.retry_delay = 0.5
    
    def handle_pdf_processing_error(self, error: Exception, pdf_path: str) -> Dict[str, Any]:
        """
        Handle PDF processing errors with fallback strategies.
        
        Args:
            error: Exception that occurred
            pdf_path: Path to the PDF file that failed
            
        Returns:
            Error response dictionary
        """
        logger.error(f"PDF processing failed for {pdf_path}: {error}")
        
        error_response = {
            "title": f"Error: Failed to process {Path(pdf_path).name}",
            "outline": [],
            "error": {
                "type": "PDFProcessingError",
                "message": str(error),
                "details": f"Failed to process PDF: {pdf_path}",
                "error_code": getattr(error, 'error_code', 'PDF_PROCESSING_FAILED'),
                "timestamp": time.time()
            }
        }
        
        # Log detailed error information
        self._log_detailed_error(error, "PDF Processing", {"pdf_path": pdf_path})
        
        return error_response
    
    def handle_model_loading_error(self, error: Exception, model_path: str) -> Dict[str, Any]:
        """
        Handle model loading errors with fallback to rule-based classification.
        
        Args:
            error: Exception that occurred
            model_path: Path to the model that failed to load
            
        Returns:
            Error response dictionary
        """
        logger.error(f"Model loading failed for {model_path}: {error}")
        
        if self.fallback_enabled:
            logger.info("Falling back to rule-based classification")
            return {
                "fallback_mode": True,
                "classification_method": "rule_based",
                "model_error": str(error)
            }
        
        error_response = {
            "title": "Error: Model Loading Failed",
            "outline": [],
            "error": {
                "type": "ModelLoadingError",
                "message": str(error),
                "details": f"Failed to load model from: {model_path}",
                "error_code": getattr(error, 'error_code', 'MODEL_LOADING_FAILED'),
                "timestamp": time.time()
            }
        }
        
        self._log_detailed_error(error, "Model Loading", {"model_path": model_path})
        
        return error_response
    
    def handle_classification_error(self, error: Exception, text: str, fallback_classifier=None) -> tuple:
        """
        Handle classification errors with fallback to rule-based classification.
        
        Args:
            error: Exception that occurred
            text: Text that failed to classify
            fallback_classifier: Fallback classifier to use
            
        Returns:
            Tuple of (predicted_class, confidence)
        """
        logger.warning(f"Classification failed for text: {text[:50]}... Error: {error}")
        
        if fallback_classifier and self.fallback_enabled:
            try:
                logger.debug("Using fallback rule-based classification")
                return fallback_classifier.classify_text_only(text)
            except Exception as fallback_error:
                logger.error(f"Fallback classification also failed: {fallback_error}")
        
        # Ultimate fallback - classify as regular text
        return "text", 0.1
    
    def handle_feature_extraction_error(self, error: Exception, block_text: str) -> Dict[str, Any]:
        """
        Handle feature extraction errors with default feature values.
        
        Args:
            error: Exception that occurred
            block_text: Text block that failed feature extraction
            
        Returns:
            Default feature dictionary
        """
        logger.warning(f"Feature extraction failed for block: {block_text[:50]}... Error: {error}")
        
        # Return default feature values
        default_features = {
            'font_size_ratio': 1.0,
            'is_bold': False,
            'is_italic': False,
            'position_x': 0.0,
            'position_y': 0.0,
            'text_length': len(block_text),
            'capitalization_score': 0.0,
            'whitespace_ratio': 0.0
        }
        
        self._log_detailed_error(error, "Feature Extraction", {"block_text": block_text[:100]})
        
        return default_features
    
    def handle_structure_analysis_error(self, error: Exception, classified_blocks: List) -> DocumentStructure:
        """
        Handle structure analysis errors with minimal document structure.
        
        Args:
            error: Exception that occurred
            classified_blocks: Classification results that failed analysis
            
        Returns:
            Minimal DocumentStructure
        """
        logger.error(f"Structure analysis failed: {error}")
        
        # Try to extract at least some basic structure
        try:
            basic_headings = []
            for block in classified_blocks[:10]:  # Limit to first 10 to avoid further errors
                if hasattr(block, 'predicted_class') and block.predicted_class in ['h1', 'h2', 'h3']:
                    heading = Heading(
                        level=block.predicted_class.upper(),
                        text=getattr(block.block, 'text', 'Unknown')[:100],  # Limit text length
                        page=getattr(block.block, 'page_number', 1),
                        confidence=getattr(block, 'confidence', 0.1)
                    )
                    basic_headings.append(heading)
            
            return DocumentStructure(
                title="Document (Structure Analysis Failed)",
                headings=basic_headings,
                metadata={
                    'error': str(error),
                    'structure_analysis_failed': True,
                    'partial_extraction': True
                }
            )
        except Exception as fallback_error:
            logger.error(f"Fallback structure analysis also failed: {fallback_error}")
            
            return DocumentStructure(
                title="Error: Structure Analysis Failed",
                headings=[],
                metadata={
                    'error': str(error),
                    'fallback_error': str(fallback_error),
                    'structure_analysis_failed': True
                }
            )
    
    def handle_output_generation_error(self, error: Exception, structure: DocumentStructure = None) -> Dict[str, Any]:
        """
        Handle output generation errors with minimal valid JSON.
        
        Args:
            error: Exception that occurred
            structure: DocumentStructure that failed to serialize
            
        Returns:
            Minimal valid JSON response
        """
        logger.error(f"Output generation failed: {error}")
        
        # Try to salvage some information from the structure
        title = "Error: Output Generation Failed"
        outline = []
        
        if structure:
            try:
                title = structure.title or title
                # Try to extract basic outline information
                for heading in structure.headings[:5]:  # Limit to first 5 headings
                    try:
                        outline_entry = {
                            "level": heading.level,
                            "text": str(heading.text)[:200],  # Limit text length
                            "page": int(heading.page) if isinstance(heading.page, (int, float)) else 1
                        }
                        outline.append(outline_entry)
                    except Exception:
                        continue  # Skip problematic headings
            except Exception as extraction_error:
                logger.warning(f"Failed to extract structure information: {extraction_error}")
        
        error_response = {
            "title": title,
            "outline": outline,
            "error": {
                "type": "OutputGenerationError",
                "message": str(error),
                "details": "Failed to generate proper JSON output",
                "error_code": getattr(error, 'error_code', 'OUTPUT_GENERATION_FAILED'),
                "timestamp": time.time()
            }
        }
        
        self._log_detailed_error(error, "Output Generation", {"structure_available": structure is not None})
        
        return error_response
    
    def handle_timeout_error(self, pdf_path: str, timeout_seconds: int) -> Dict[str, Any]:
        """
        Handle timeout errors with appropriate response.
        
        Args:
            pdf_path: Path to the PDF that timed out
            timeout_seconds: Timeout limit that was exceeded
            
        Returns:
            Timeout error response
        """
        logger.error(f"Processing timeout for {pdf_path} (limit: {timeout_seconds}s)")
        
        return {
            "title": f"Error: Processing timeout for {Path(pdf_path).name}",
            "outline": [],
            "error": {
                "type": "TimeoutError",
                "message": f"Processing exceeded {timeout_seconds} second limit",
                "details": f"PDF processing timed out: {pdf_path}",
                "error_code": "PROCESSING_TIMEOUT",
                "timeout_limit": timeout_seconds,
                "timestamp": time.time()
            }
        }
    
    def handle_validation_error(self, error: Exception, data: Any = None) -> Dict[str, Any]:
        """
        Handle validation errors with corrected output.
        
        Args:
            error: Validation error that occurred
            data: Data that failed validation
            
        Returns:
            Corrected or minimal valid response
        """
        logger.error(f"Validation failed: {error}")
        
        return {
            "title": "Error: Invalid Output Format",
            "outline": [],
            "error": {
                "type": "ValidationError",
                "message": str(error),
                "details": "Generated output failed format validation",
                "error_code": getattr(error, 'error_code', 'VALIDATION_FAILED'),
                "timestamp": time.time()
            }
        }
    
    def _log_detailed_error(self, error: Exception, context: str, additional_info: Dict[str, Any] = None):
        """Log detailed error information for debugging."""
        error_info = {
            "context": context,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "additional_info": additional_info or {}
        }
        
        logger.error(f"Detailed error information: {error_info}")
        
        # Track error counts for monitoring
        error_key = f"{context}:{type(error).__name__}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        return {
            "error_counts": self.error_counts.copy(),
            "total_errors": sum(self.error_counts.values()),
            "fallback_enabled": self.fallback_enabled
        }


class TimeoutManager:
    """Manages processing timeouts with graceful handling."""
    
    def __init__(self, default_timeout: int = None):
        self.default_timeout = default_timeout or config.processing_timeout
        self.active_timeouts = {}
    
    @contextmanager
    def timeout_context(self, timeout_seconds: int = None, operation_name: str = "operation"):
        """
        Context manager for timeout handling.
        
        Args:
            timeout_seconds: Timeout limit in seconds
            operation_name: Name of the operation for logging
        """
        timeout = timeout_seconds or self.default_timeout
        
        def timeout_handler(signum, frame):
            raise TimeoutError(
                f"{operation_name} exceeded {timeout} second limit",
                error_code="OPERATION_TIMEOUT",
                details={"operation": operation_name, "timeout": timeout}
            )
        
        # Set up signal handler
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        
        try:
            logger.debug(f"Starting {operation_name} with {timeout}s timeout")
            yield
            logger.debug(f"Completed {operation_name} within timeout")
        except TimeoutError:
            logger.error(f"{operation_name} timed out after {timeout}s")
            raise
        finally:
            # Clean up
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


class RetryManager:
    """Manages retry logic for transient failures."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 0.5):
        self.max_retries = max_retries
        self.base_delay = base_delay
    
    def retry_with_backoff(self, func: Callable, *args, **kwargs):
        """
        Retry function with exponential backoff.
        
        Args:
            func: Function to retry
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Last exception if all retries fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    delay = self.base_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries + 1} attempts failed. Last error: {e}")
        
        raise last_exception


def error_handler_decorator(error_type: type = Exception, fallback_value: Any = None):
    """
    Decorator for automatic error handling.
    
    Args:
        error_type: Type of exception to catch
        fallback_value: Value to return on error
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except error_type as e:
                logger.error(f"Error in {func.__name__}: {e}")
                return fallback_value
        return wrapper
    return decorator


def safe_operation(operation_name: str, fallback_value: Any = None):
    """
    Decorator for safe operations with logging and fallback.
    
    Args:
        operation_name: Name of the operation for logging
        fallback_value: Value to return on error
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                logger.debug(f"Starting {operation_name}")
                result = func(*args, **kwargs)
                logger.debug(f"Completed {operation_name}")
                return result
            except Exception as e:
                logger.error(f"Error in {operation_name}: {e}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
                
                # Track error statistics
                global_error_handler._log_detailed_error(e, operation_name)
                
                return fallback_value
        return wrapper
    return decorator


# Global error handler instance
global_error_handler = ErrorHandler()
timeout_manager = TimeoutManager()
retry_manager = RetryManager()