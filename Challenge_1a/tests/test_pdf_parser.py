"""
Unit tests for the PDF parser module.

This module contains comprehensive tests for PDF parsing functionality,
including text extraction, formatting information capture, and multilingual support.
"""

import unittest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

try:
    from src.pdf_extractor.core.pdf_parser import PDFParser, PDFParsingError
    from src.pdf_extractor.models.models import TextBlock
except ImportError:
    # Handle relative imports when running as script
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from pdf_parser import PDFParser, PDFParsingError
    from models import TextBlock


class TestPDFParser(unittest.TestCase):
    """Test cases for PDFParser class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parser = PDFParser()
        self.test_data_dir = Path(__file__).parent / "input"
        
        # Sample PDF files for testing
        self.sample_pdf1 = self.test_data_dir / "E0CCG5S239.pdf"
        self.sample_pdf2 = self.test_data_dir / "TOPJUMP-PARTY-INVITATION-20161003-V01.pdf"
    
    def tearDown(self):
        """Clean up after tests."""
        self.parser.close()
    
    def test_init(self):
        """Test PDFParser initialization."""
        parser = PDFParser()
        self.assertIsNotNone(parser.config)
        self.assertIsNone(parser.document)
        self.assertEqual(parser._page_cache, {})
    
    def test_parse_document_with_valid_pdf(self):
        """Test parsing a valid PDF document."""
        if not self.sample_pdf1.exists():
            self.skipTest(f"Sample PDF {self.sample_pdf1} not found")
        
        blocks = self.parser.parse_document(str(self.sample_pdf1))
        
        # Verify we got text blocks
        self.assertIsInstance(blocks, list)
        self.assertGreater(len(blocks), 0)
        
        # Verify each block is a TextBlock instance
        for block in blocks:
            self.assertIsInstance(block, TextBlock)
            self.assertIsInstance(block.text, str)
            self.assertGreater(len(block.text), 0)
            self.assertIsInstance(block.page_number, int)
            self.assertGreater(block.page_number, 0)
            self.assertIsInstance(block.bbox, tuple)
            self.assertEqual(len(block.bbox), 4)
            self.assertIsInstance(block.font_size, float)
            self.assertGreater(block.font_size, 0)
            self.assertIsInstance(block.font_name, str)
            self.assertIsInstance(block.font_flags, int)
    
    def test_parse_document_with_second_pdf(self):
        """Test parsing the second sample PDF."""
        if not self.sample_pdf2.exists():
            self.skipTest(f"Sample PDF {self.sample_pdf2} not found")
        
        blocks = self.parser.parse_document(str(self.sample_pdf2))
        
        # Verify we got text blocks
        self.assertIsInstance(blocks, list)
        self.assertGreater(len(blocks), 0)
        
        # Check that page numbers are properly assigned
        page_numbers = {block.page_number for block in blocks}
        self.assertGreater(len(page_numbers), 0)
        self.assertTrue(all(page_num > 0 for page_num in page_numbers))
    
    def test_parse_nonexistent_file(self):
        """Test parsing a non-existent PDF file."""
        with self.assertRaises(PDFParsingError):
            self.parser.parse_document("nonexistent.pdf")
    
    def test_extract_page_text_without_document(self):
        """Test extracting page text without loading a document."""
        with self.assertRaises(PDFParsingError):
            self.parser.extract_page_text(0)
    
    def test_extract_page_text_invalid_page(self):
        """Test extracting text from an invalid page number."""
        if not self.sample_pdf1.exists():
            self.skipTest(f"Sample PDF {self.sample_pdf1} not found")
        
        self.parser.parse_document(str(self.sample_pdf1))
        
        # Try to extract from a page that doesn't exist
        with self.assertRaises(PDFParsingError):
            self.parser.extract_page_text(9999)
    
    def test_extract_page_text_caching(self):
        """Test that page text extraction uses caching."""
        if not self.sample_pdf1.exists():
            self.skipTest(f"Sample PDF {self.sample_pdf1} not found")
        
        self.parser.parse_document(str(self.sample_pdf1))
        
        # Extract from page 0 twice
        blocks1 = self.parser.extract_page_text(0)
        blocks2 = self.parser.extract_page_text(0)
        
        # Should be the same objects (cached)
        self.assertEqual(len(blocks1), len(blocks2))
        self.assertIn(0, self.parser._page_cache)
    
    def test_get_document_metadata_without_document(self):
        """Test getting metadata without loading a document."""
        metadata = self.parser.get_document_metadata()
        self.assertEqual(metadata, {})
    
    def test_get_document_metadata_with_document(self):
        """Test getting metadata from a loaded document."""
        if not self.sample_pdf1.exists():
            self.skipTest(f"Sample PDF {self.sample_pdf1} not found")
        
        self.parser.parse_document(str(self.sample_pdf1))
        metadata = self.parser.get_document_metadata()
        
        self.assertIsInstance(metadata, dict)
        self.assertIn('page_count', metadata)
        self.assertIn('is_encrypted', metadata)
        self.assertIn('is_pdf', metadata)
        self.assertTrue(metadata['is_pdf'])
        self.assertIsInstance(metadata['page_count'], int)
        self.assertGreater(metadata['page_count'], 0)
    
    def test_normalize_text_encoding(self):
        """Test text encoding normalization."""
        # Test normal text
        normal_text = "Hello World"
        normalized = self.parser._normalize_text_encoding(normal_text)
        self.assertEqual(normalized, "Hello World")
        
        # Test empty text
        empty_normalized = self.parser._normalize_text_encoding("")
        self.assertEqual(empty_normalized, "")
        
        # Test None input
        none_normalized = self.parser._normalize_text_encoding(None)
        self.assertEqual(none_normalized, "")
        
        # Test text with whitespace
        whitespace_text = "  Hello World  "
        whitespace_normalized = self.parser._normalize_text_encoding(whitespace_text)
        self.assertEqual(whitespace_normalized, "Hello World")
    
    def test_font_flag_detection(self):
        """Test font flag detection methods."""
        # Test bold detection
        bold_flags = 16  # 2^4
        self.assertTrue(self.parser._is_bold_font(bold_flags))
        self.assertFalse(self.parser._is_bold_font(0))
        
        # Test italic detection
        italic_flags = 2  # 2^1
        self.assertTrue(self.parser._is_italic_font(italic_flags))
        self.assertFalse(self.parser._is_italic_font(0))
        
        # Test combined flags
        combined_flags = 18  # Bold + Italic
        self.assertTrue(self.parser._is_bold_font(combined_flags))
        self.assertTrue(self.parser._is_italic_font(combined_flags))
    
    def test_context_manager(self):
        """Test PDFParser as context manager."""
        if not self.sample_pdf1.exists():
            self.skipTest(f"Sample PDF {self.sample_pdf1} not found")
        
        with PDFParser() as parser:
            blocks = parser.parse_document(str(self.sample_pdf1))
            self.assertGreater(len(blocks), 0)
        
        # Document should be closed after context exit
        self.assertIsNone(parser.document)
    
    def test_close_method(self):
        """Test the close method."""
        if not self.sample_pdf1.exists():
            self.skipTest(f"Sample PDF {self.sample_pdf1} not found")
        
        self.parser.parse_document(str(self.sample_pdf1))
        self.assertIsNotNone(self.parser.document)
        
        self.parser.close()
        self.assertIsNone(self.parser.document)
        self.assertEqual(self.parser._page_cache, {})
    
    def test_multilingual_text_handling(self):
        """Test handling of multilingual text content."""
        # This test would ideally use a PDF with multilingual content
        # For now, we test the normalization function with various encodings
        
        test_cases = [
            "English text",
            "Français",
            "Español",
            "Deutsch",
            "中文",
            "العربية",
            "русский"
        ]
        
        for text in test_cases:
            normalized = self.parser._normalize_text_encoding(text)
            self.assertIsInstance(normalized, str)
            self.assertGreater(len(normalized), 0)
    
    def test_text_block_properties(self):
        """Test that extracted TextBlocks have correct properties."""
        if not self.sample_pdf1.exists():
            self.skipTest(f"Sample PDF {self.sample_pdf1} not found")
        
        blocks = self.parser.parse_document(str(self.sample_pdf1))
        
        for block in blocks[:10]:  # Test first 10 blocks
            # Text should be non-empty string
            self.assertIsInstance(block.text, str)
            self.assertGreater(len(block.text.strip()), 0)
            
            # Page number should be positive integer
            self.assertIsInstance(block.page_number, int)
            self.assertGreater(block.page_number, 0)
            
            # Bounding box should be 4-tuple of numbers
            self.assertIsInstance(block.bbox, tuple)
            self.assertEqual(len(block.bbox), 4)
            for coord in block.bbox:
                self.assertIsInstance(coord, (int, float))
            
            # Font size should be positive number
            self.assertIsInstance(block.font_size, (int, float))
            self.assertGreater(block.font_size, 0)
            
            # Font name should be string
            self.assertIsInstance(block.font_name, str)
            
            # Font flags should be integer
            self.assertIsInstance(block.font_flags, int)
            self.assertGreaterEqual(block.font_flags, 0)


class TestPDFParsingError(unittest.TestCase):
    """Test cases for PDFParsingError exception."""
    
    def test_pdf_parsing_error_creation(self):
        """Test creating PDFParsingError."""
        error_msg = "Test error message"
        error = PDFParsingError(error_msg)
        self.assertEqual(str(error), error_msg)
        self.assertIsInstance(error, Exception)


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run the tests
    unittest.main()