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
except ImportError:
    # Handle relative imports when running as script
    from models import TextBlock
    from config import config


logger = logging.getLogger(__name__)


class PDFParsingError(Exception):
    """Exception raised when PDF parsing fails."""
    pass


class PDFParser:
    """PDF parser that extracts text blocks with formatting information using PyMuPDF."""
    
    def __init__(self):
        self.config = config.get_pdf_config()
        self.document = None
        self._page_cache = {}
    
    def parse_document(self, pdf_path: str) -> List[TextBlock]:
        """
        Parse a PDF document and extract all text blocks with formatting information.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of TextBlock objects containing extracted text and metadata
            
        Raises:
            PDFParsingError: If the PDF cannot be opened or parsed
        """
        try:
            # Open the PDF document
            self.document = fitz.open(pdf_path)
            
            # Check if document is valid
            if self.document.is_closed:
                raise PDFParsingError(f"Cannot open PDF file: {pdf_path}")
            
            # Check page count limit
            page_count = len(self.document)
            max_pages = config.max_pages
            if page_count > max_pages:
                logger.warning(f"Document has {page_count} pages, limiting to {max_pages}")
                page_count = max_pages
            
            # Extract text blocks from all pages
            all_blocks = []
            for page_num in range(page_count):
                try:
                    page_blocks = self.extract_page_text(page_num)
                    all_blocks.extend(page_blocks)
                except Exception as e:
                    logger.error(f"Error extracting text from page {page_num + 1}: {e}")
                    continue
            
            logger.info(f"Extracted {len(all_blocks)} text blocks from {page_count} pages")
            return all_blocks
            
        except Exception as e:
            raise PDFParsingError(f"Failed to parse PDF document: {e}")
    
    def extract_page_text(self, page_num: int) -> List[TextBlock]:
        """
        Extract text blocks from a specific page.
        
        Args:
            page_num: Page number to extract text from (0-based)
            
        Returns:
            List of TextBlock objects for the specified page
            
        Raises:
            PDFParsingError: If the page cannot be processed
        """
        if not self.document:
            raise PDFParsingError("No document loaded")
        
        if page_num >= len(self.document):
            raise PDFParsingError(f"Page {page_num} does not exist")
        
        # Check cache first
        if page_num in self._page_cache:
            return self._page_cache[page_num]
        
        try:
            page = self.document[page_num]
            
            # Get text blocks with formatting information
            blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)
            
            text_blocks = []
            for block in blocks["blocks"]:
                # Skip image blocks
                if "image" in block:
                    continue
                
                # Process text blocks
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            raw_text = span.get("text", "")
                            
                            # Normalize text encoding for multilingual support
                            text = self._normalize_text_encoding(raw_text)
                            
                            # Skip empty text
                            if not text:
                                continue
                            
                            # Extract formatting information
                            font_size = span.get("size", 12.0)
                            font_name = span.get("font", "unknown")
                            font_flags = span.get("flags", 0)
                            bbox = span.get("bbox", (0, 0, 0, 0))
                            
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
            
            # Cache the results
            self._page_cache[page_num] = text_blocks
            
            logger.debug(f"Extracted {len(text_blocks)} text blocks from page {page_num + 1}")
            return text_blocks
            
        except Exception as e:
            raise PDFParsingError(f"Failed to extract text from page {page_num + 1}: {e}")
    
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
        Normalize text encoding to handle multilingual content.
        
        Args:
            text: Raw text string
            
        Returns:
            Normalized text string
        """
        if not text:
            return ""
        
        try:
            # Try to encode/decode to ensure proper UTF-8
            normalized = text.encode('utf-8', errors='ignore').decode('utf-8')
            return normalized.strip()
        except Exception:
            # Fallback: return original text
            return text.strip()
    
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