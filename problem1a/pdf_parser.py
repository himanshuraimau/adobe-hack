"""
PDF parsing module using PyMuPDF for text extraction with formatting information.

This module handles PDF document parsing, extracting text blocks with their
formatting metadata including font information, positioning, and page numbers.
"""

from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF
import logging
from pathlib import Path
try:
    from .models import TextBlock
    from .config import config
    from .error_handler import (
        PDFProcessingError, global_error_handler, timeout_manager,
        safe_operation, error_handler_decorator
    )
except ImportError:
    # Handle relative imports when running as script
    from models import TextBlock
    from config import config
    from error_handler import (
        PDFProcessingError, global_error_handler, timeout_manager,
        safe_operation, error_handler_decorator
    )


logger = logging.getLogger(__name__)


class PDFParser:
    """PDF parser that extracts text blocks with formatting information using PyMuPDF."""
    
    def __init__(self):
        self.config = config.get_pdf_config()
        self.document = None
        self._page_cache = {}
        self._cache_enabled = True
        self._max_cache_size = 10  # Limit cache size for memory efficiency
    
    @safe_operation("PDF document parsing", fallback_value=[])
    def parse_document(self, pdf_path: str) -> List[TextBlock]:
        """
        Parse a PDF document and extract all text blocks with formatting information.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of TextBlock objects containing extracted text and metadata
            
        Raises:
            PDFProcessingError: If the PDF cannot be opened or parsed
        """
        try:
            # Validate input
            if not pdf_path or not Path(pdf_path).exists():
                raise PDFProcessingError(
                    f"PDF file does not exist: {pdf_path}",
                    error_code="FILE_NOT_FOUND",
                    details={"pdf_path": pdf_path}
                )
            
            # Check file size (basic validation)
            file_size = Path(pdf_path).stat().st_size
            if file_size == 0:
                raise PDFProcessingError(
                    f"PDF file is empty: {pdf_path}",
                    error_code="EMPTY_FILE",
                    details={"pdf_path": pdf_path, "file_size": file_size}
                )
            
            # Open the PDF document with timeout
            with timeout_manager.timeout_context(5, "PDF document opening"):
                try:
                    self.document = fitz.open(pdf_path)
                except Exception as e:
                    if "password" in str(e).lower() or "encrypted" in str(e).lower():
                        raise PDFProcessingError(
                            f"PDF is password protected: {pdf_path}",
                            error_code="PASSWORD_PROTECTED",
                            details={"pdf_path": pdf_path}
                        )
                    elif "corrupt" in str(e).lower() or "damaged" in str(e).lower():
                        raise PDFProcessingError(
                            f"PDF file is corrupted: {pdf_path}",
                            error_code="CORRUPTED_FILE",
                            details={"pdf_path": pdf_path}
                        )
                    else:
                        raise PDFProcessingError(
                            f"Cannot open PDF file: {pdf_path}. Error: {e}",
                            error_code="OPEN_FAILED",
                            details={"pdf_path": pdf_path, "original_error": str(e)}
                        )
            
            # Check if document is valid
            if self.document.is_closed:
                raise PDFProcessingError(
                    f"PDF document is closed after opening: {pdf_path}",
                    error_code="DOCUMENT_CLOSED",
                    details={"pdf_path": pdf_path}
                )
            
            # Check page count
            try:
                page_count = len(self.document)
            except Exception as e:
                raise PDFProcessingError(
                    f"Cannot determine page count: {pdf_path}. Error: {e}",
                    error_code="PAGE_COUNT_FAILED",
                    details={"pdf_path": pdf_path, "original_error": str(e)}
                )
            
            if page_count == 0:
                raise PDFProcessingError(
                    f"PDF has no pages: {pdf_path}",
                    error_code="NO_PAGES",
                    details={"pdf_path": pdf_path}
                )
            
            # Apply page limit
            max_pages = config.max_pages
            if page_count > max_pages:
                logger.warning(f"Document has {page_count} pages, limiting to {max_pages}")
                page_count = max_pages
            
            # Extract text blocks from all pages with error recovery
            all_blocks = []
            failed_pages = []
            
            for page_num in range(page_count):
                try:
                    page_blocks = self.extract_page_text(page_num)
                    all_blocks.extend(page_blocks)
                except Exception as e:
                    failed_pages.append(page_num + 1)
                    logger.error(f"Error extracting text from page {page_num + 1}: {e}")
                    # Continue processing other pages
                    continue
            
            # Log results
            if failed_pages:
                logger.warning(f"Failed to process {len(failed_pages)} pages: {failed_pages}")
            
            if not all_blocks:
                raise PDFProcessingError(
                    f"No text blocks extracted from PDF: {pdf_path}",
                    error_code="NO_TEXT_EXTRACTED",
                    details={
                        "pdf_path": pdf_path,
                        "page_count": page_count,
                        "failed_pages": failed_pages
                    }
                )
            
            logger.info(f"Extracted {len(all_blocks)} text blocks from {page_count - len(failed_pages)}/{page_count} pages")
            return all_blocks
            
        except PDFProcessingError:
            # Re-raise our custom errors
            raise
        except Exception as e:
            # Wrap unexpected errors
            raise PDFProcessingError(
                f"Unexpected error parsing PDF document: {e}",
                error_code="UNEXPECTED_ERROR",
                details={"pdf_path": pdf_path, "original_error": str(e)}
            )
    
    @safe_operation("page text extraction", fallback_value=[])
    def extract_page_text(self, page_num: int) -> List[TextBlock]:
        """
        Extract text blocks from a specific page.
        
        Args:
            page_num: Page number to extract text from (0-based)
            
        Returns:
            List of TextBlock objects for the specified page
            
        Raises:
            PDFProcessingError: If the page cannot be processed
        """
        try:
            # Validate inputs
            if not self.document:
                raise PDFProcessingError(
                    "No document loaded",
                    error_code="NO_DOCUMENT",
                    details={"page_num": page_num}
                )
            
            if page_num < 0:
                raise PDFProcessingError(
                    f"Invalid page number: {page_num} (must be >= 0)",
                    error_code="INVALID_PAGE_NUMBER",
                    details={"page_num": page_num}
                )
            
            if page_num >= len(self.document):
                raise PDFProcessingError(
                    f"Page {page_num} does not exist (document has {len(self.document)} pages)",
                    error_code="PAGE_NOT_FOUND",
                    details={"page_num": page_num, "total_pages": len(self.document)}
                )
            
            # Check cache first
            if page_num in self._page_cache:
                logger.debug(f"Using cached text blocks for page {page_num + 1}")
                return self._page_cache[page_num]
            
            # Extract text with timeout
            with timeout_manager.timeout_context(3, f"page {page_num + 1} text extraction"):
                try:
                    page = self.document[page_num]
                    
                    # Get text blocks with formatting information
                    blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)
                    
                    if not blocks or "blocks" not in blocks:
                        logger.warning(f"No text blocks found on page {page_num + 1}")
                        return []
                    
                except Exception as e:
                    raise PDFProcessingError(
                        f"Failed to access page {page_num + 1}: {e}",
                        error_code="PAGE_ACCESS_FAILED",
                        details={"page_num": page_num, "original_error": str(e)}
                    )
                
                # Process text blocks with error recovery
                text_blocks = []
                block_errors = 0
                
                for block_idx, block in enumerate(blocks["blocks"]):
                    try:
                        # Skip image blocks
                        if "image" in block:
                            continue
                        
                        # Process text blocks
                        if "lines" in block:
                            for line_idx, line in enumerate(block["lines"]):
                                for span_idx, span in enumerate(line["spans"]):
                                    try:
                                        raw_text = span.get("text", "")
                                        
                                        # Normalize text encoding for multilingual support
                                        text = self._normalize_text_encoding(raw_text)
                                        
                                        # Skip empty text
                                        if not text:
                                            continue
                                        
                                        # Extract formatting information with defaults
                                        font_size = max(span.get("size", 12.0), 1.0)  # Ensure positive font size
                                        font_name = span.get("font", "unknown")
                                        font_flags = span.get("flags", 0)
                                        bbox = span.get("bbox", (0, 0, 0, 0))
                                        
                                        # Validate bbox
                                        if len(bbox) != 4 or any(not isinstance(x, (int, float)) for x in bbox):
                                            bbox = (0, 0, 0, 0)
                                        
                                        # Create TextBlock object
                                        text_block = TextBlock(
                                            text=text,
                                            page_number=page_num + 1,  # Convert to 1-based
                                            bbox=bbox,
                                            font_size=font_size,
                                            font_name=font_name,
                                            font_flags=font_flags
                                        )
                                        
                                        text_blocks.append(text_block)
                                        
                                    except Exception as span_error:
                                        block_errors += 1
                                        logger.debug(f"Error processing span {span_idx} in line {line_idx} of block {block_idx} on page {page_num + 1}: {span_error}")
                                        continue  # Skip problematic spans
                                        
                    except Exception as block_error:
                        block_errors += 1
                        logger.debug(f"Error processing block {block_idx} on page {page_num + 1}: {block_error}")
                        continue  # Skip problematic blocks
                
                # Log block processing results
                if block_errors > 0:
                    logger.warning(f"Encountered {block_errors} block processing errors on page {page_num + 1}")
                
                # Cache the results with memory management
                if self._cache_enabled and len(self._page_cache) < self._max_cache_size:
                    self._page_cache[page_num] = text_blocks
                elif self._cache_enabled and len(self._page_cache) >= self._max_cache_size:
                    # Remove oldest cache entry to make room
                    oldest_key = next(iter(self._page_cache))
                    del self._page_cache[oldest_key]
                    self._page_cache[page_num] = text_blocks
                
                logger.debug(f"Extracted {len(text_blocks)} text blocks from page {page_num + 1}")
                return text_blocks
                
        except PDFProcessingError:
            # Re-raise our custom errors
            raise
        except Exception as e:
            # Wrap unexpected errors
            raise PDFProcessingError(
                f"Unexpected error extracting text from page {page_num + 1}: {e}",
                error_code="UNEXPECTED_PAGE_ERROR",
                details={"page_num": page_num, "original_error": str(e)}
            )
    
    def get_document_metadata(self) -> Dict[str, Any]:
        """
        Get document metadata including title, author, creation date, etc.
        
        Returns:
            Dictionary containing document metadata
        """
        if not self.document:
            return {}
        
        try:
            metadata = self.document.metadata
            
            # Clean and normalize metadata
            cleaned_metadata = {}
            for key, value in metadata.items():
                if value and isinstance(value, str):
                    # Handle encoding issues
                    try:
                        cleaned_value = value.encode('utf-8').decode('utf-8')
                        cleaned_metadata[key] = cleaned_value.strip()
                    except (UnicodeDecodeError, UnicodeEncodeError):
                        # Fallback for encoding issues
                        cleaned_metadata[key] = str(value).strip()
                elif value:
                    cleaned_metadata[key] = value
            
            # Add document statistics
            cleaned_metadata.update({
                'page_count': len(self.document),
                'is_encrypted': self.document.needs_pass,
                'is_pdf': True
            })
            
            return cleaned_metadata
            
        except Exception as e:
            logger.error(f"Error extracting document metadata: {e}")
            return {}
    
    def _normalize_text_encoding(self, text: str) -> str:
        """
        Normalize text encoding to handle multilingual content with enhanced support.
        
        Args:
            text: Raw text string
            
        Returns:
            Normalized text string
        """
        if not text:
            return ""
        
        try:
            # Handle common encoding issues
            # First try to decode as UTF-8 if it's bytes
            if isinstance(text, bytes):
                try:
                    text = text.decode('utf-8')
                except UnicodeDecodeError:
                    # Try other common encodings
                    for encoding in ['latin1', 'cp1252', 'iso-8859-1', 'cp1251', 'shift_jis', 'gb2312']:
                        try:
                            text = text.decode(encoding)
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        # If all fail, use error handling
                        text = text.decode('utf-8', errors='replace')
            
            # Normalize Unicode characters for consistent representation
            import unicodedata
            text = unicodedata.normalize('NFC', text)
            
            # Handle various Unicode space characters
            text = text.replace('\ufeff', '')  # Remove BOM
            text = text.replace('\u00a0', ' ')  # Non-breaking space to regular space
            text = text.replace('\u2000', ' ')  # En quad
            text = text.replace('\u2001', ' ')  # Em quad
            text = text.replace('\u2002', ' ')  # En space
            text = text.replace('\u2003', ' ')  # Em space
            text = text.replace('\u2004', ' ')  # Three-per-em space
            text = text.replace('\u2005', ' ')  # Four-per-em space
            text = text.replace('\u2006', ' ')  # Six-per-em space
            text = text.replace('\u2007', ' ')  # Figure space
            text = text.replace('\u2008', ' ')  # Punctuation space
            text = text.replace('\u2009', ' ')  # Thin space
            text = text.replace('\u200a', ' ')  # Hair space
            text = text.replace('\u202f', ' ')  # Narrow no-break space
            text = text.replace('\u205f', ' ')  # Medium mathematical space
            text = text.replace('\u3000', ' ')  # Ideographic space
            
            # Handle line and paragraph separators
            text = text.replace('\u2028', '\n')  # Line separator to newline
            text = text.replace('\u2029', '\n\n')  # Paragraph separator to double newline
            
            # Handle directional marks (important for RTL languages like Arabic)
            text = text.replace('\u200e', '')  # Left-to-right mark
            text = text.replace('\u200f', '')  # Right-to-left mark
            text = text.replace('\u202a', '')  # Left-to-right embedding
            text = text.replace('\u202b', '')  # Right-to-left embedding
            text = text.replace('\u202c', '')  # Pop directional formatting
            text = text.replace('\u202d', '')  # Left-to-right override
            text = text.replace('\u202e', '')  # Right-to-left override
            
            # Handle zero-width characters
            text = text.replace('\u200b', '')  # Zero width space
            text = text.replace('\u200c', '')  # Zero width non-joiner
            text = text.replace('\u200d', '')  # Zero width joiner
            text = text.replace('\u00ad', '')  # Soft hyphen
            
            # Normalize quotation marks for consistency
            text = text.replace('\u201c', '"')  # Left double quotation mark
            text = text.replace('\u201d', '"')  # Right double quotation mark
            text = text.replace('\u2018', "'")  # Left single quotation mark
            text = text.replace('\u2019', "'")  # Right single quotation mark
            text = text.replace('\u201e', '"')  # Double low-9 quotation mark
            text = text.replace('\u201a', "'")  # Single low-9 quotation mark
            
            # Clean up excessive whitespace while preserving structure
            # Replace multiple spaces with single space
            text = ' '.join(text.split())
            
            return text.strip()
            
        except Exception as e:
            logger.warning(f"Text encoding normalization failed: {e}")
            return str(text).strip() if text else ""
    
    def _is_bold_font(self, font_flags: int) -> bool:
        """Check if font is bold based on flags."""
        return bool(font_flags & 2**4)  # Bold flag
    
    def _is_italic_font(self, font_flags: int) -> bool:
        """Check if font is italic based on flags."""
        return bool(font_flags & 2**1)  # Italic flag
    
    def close(self):
        """Close the PDF document and free resources."""
        if self.document:
            self.document.close()
            self.document = None
        self._page_cache.clear()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()