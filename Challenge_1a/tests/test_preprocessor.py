"""
Unit tests for the text preprocessing module.

This module tests text preprocessing functionality including normalization,
grouping, and structure preservation with various formatting scenarios.
"""

import unittest
from unittest.mock import Mock, patch
import sys
import os

# Add the current directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pdf_extractor.models.models import TextBlock, ProcessedBlock, FeatureVector
from src.pdf_extractor.core.preprocessor import TextPreprocessor, TextNormalizer, StructurePreserver, TextBlockGrouper


class TestTextNormalizer(unittest.TestCase):
    """Test cases for TextNormalizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'normalize_whitespace': True,
            'preserve_formatting': True
        }
        self.normalizer = TextNormalizer(self.config)
    
    def test_normalize_basic_text(self):
        """Test basic text normalization."""
        text = "  Hello   World  "
        result = self.normalizer.normalize_text(text)
        self.assertEqual(result, "Hello World")
    
    def test_normalize_unicode_text(self):
        """Test Unicode normalization for multilingual content."""
        # Test with accented characters
        text = "Café naïve résumé"
        result = self.normalizer.normalize_text(text)
        self.assertEqual(result, "Café naïve résumé")
        
        # Test with non-breaking space
        text = "Hello\u00a0World"
        result = self.normalizer.normalize_text(text)
        self.assertEqual(result, "Hello World")
        
        # Test with BOM
        text = "\ufeffHello World"
        result = self.normalizer.normalize_text(text)
        self.assertEqual(result, "Hello World")
    
    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        # Multiple spaces
        text = "Hello     World"
        result = self.normalizer.normalize_text(text)
        self.assertEqual(result, "Hello World")
        
        # Line breaks
        text = "Line 1\nLine 2\n\nParagraph 2"
        result = self.normalizer.normalize_text(text)
        self.assertEqual(result, "Line 1 Line 2\n\nParagraph 2")
        
        # Tabs
        text = "Hello\tWorld"
        result = self.normalizer.normalize_text(text)
        self.assertEqual(result, "Hello World")
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        # Excessive punctuation
        text = "Hello!!!!! World???"
        result = self.normalizer.normalize_text(text)
        self.assertEqual(result, "Hello!!! World???")
        
        # Repeated commas and dashes
        text = "Hello,, World--"
        result = self.normalizer.normalize_text(text)
        self.assertEqual(result, "Hello, World-")
        
        # Quote normalization
        text = "\u201cHello\u201d \u2018World\u2019"  # Smart quotes
        result = self.normalizer.normalize_text(text)
        self.assertEqual(result, '"Hello" \'World\'')
    
    def test_handle_special_cases(self):
        """Test handling of special formatting cases."""
        # Bullet points
        text = "• First item\n▪ Second item"
        result = self.normalizer.normalize_text(text)
        self.assertIn("• First item", result)
        self.assertIn("• Second item", result)
        
        # Numbered lists
        text = "1) First item\n2. Second item"
        result = self.normalizer.normalize_text(text)
        self.assertIn("1. First item", result)
        self.assertIn("2. Second item", result)
    
    def test_empty_text(self):
        """Test handling of empty or None text."""
        self.assertEqual(self.normalizer.normalize_text(""), "")
        self.assertEqual(self.normalizer.normalize_text(None), "")
        self.assertEqual(self.normalizer.normalize_text("   "), "")


class TestStructurePreserver(unittest.TestCase):
    """Test cases for StructurePreserver class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {}
        self.preserver = StructurePreserver(self.config)
    
    def test_preserve_structure_empty_list(self):
        """Test structure preservation with empty list."""
        result = self.preserver.preserve_structure([])
        self.assertEqual(result, [])
    
    def test_preserve_structure_single_block(self):
        """Test structure preservation with single block."""
        block = TextBlock(
            text="Test text",
            page_number=1,
            bbox=(10, 20, 100, 30),
            font_size=12.0,
            font_name="Arial",
            font_flags=0
        )
        
        result = self.preserver.preserve_structure([block])
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], ProcessedBlock)
        self.assertEqual(result[0].text, "Test text")
        self.assertEqual(result[0].page_number, 1)
        self.assertIsInstance(result[0].features, FeatureVector)
    
    def test_calculate_document_stats(self):
        """Test document statistics calculation."""
        blocks = [
            TextBlock("Text 1", 1, (0, 0, 10, 10), 12.0, "Arial", 0),
            TextBlock("Text 2", 1, (0, 10, 10, 20), 14.0, "Arial", 16),  # Bold
            TextBlock("Text 3", 2, (0, 0, 10, 10), 10.0, "Arial", 0)
        ]
        
        stats = self.preserver._calculate_document_stats(blocks)
        
        self.assertEqual(stats['avg_font_size'], 12.0)
        self.assertEqual(stats['max_font_size'], 14.0)
        self.assertEqual(stats['min_font_size'], 10.0)
        self.assertEqual(stats['total_pages'], 2)
        self.assertEqual(stats['total_blocks'], 3)
    
    def test_create_basic_features(self):
        """Test basic feature creation."""
        block = TextBlock(
            text="TEST TEXT",
            page_number=1,
            bbox=(10, 20, 100, 30),
            font_size=14.0,
            font_name="Arial",
            font_flags=16  # Bold flag
        )
        
        doc_stats = {'avg_font_size': 12.0}
        features = self.preserver._create_basic_features(block, doc_stats)
        
        self.assertAlmostEqual(features.font_size_ratio, 14.0/12.0)
        self.assertTrue(features.is_bold)
        self.assertFalse(features.is_italic)
        self.assertEqual(features.position_x, 10)
        self.assertEqual(features.position_y, 20)
        self.assertEqual(features.text_length, 9)
        self.assertEqual(features.capitalization_score, 1.0)  # All caps
    
    def test_font_style_detection(self):
        """Test font style detection."""
        # Test bold detection
        self.assertTrue(self.preserver._is_bold_font(16))  # 2^4
        self.assertFalse(self.preserver._is_bold_font(0))
        
        # Test italic detection
        self.assertTrue(self.preserver._is_italic_font(2))   # 2^1
        self.assertFalse(self.preserver._is_italic_font(0))
    
    def test_capitalization_score(self):
        """Test capitalization score calculation."""
        # All uppercase
        score = self.preserver._calculate_capitalization_score("HELLO WORLD")
        self.assertEqual(score, 1.0)
        
        # All lowercase
        score = self.preserver._calculate_capitalization_score("hello world")
        self.assertEqual(score, 0.0)
        
        # Mixed case
        score = self.preserver._calculate_capitalization_score("Hello World")
        self.assertEqual(score, 0.2)  # 2 out of 10 letters
        
        # No letters
        score = self.preserver._calculate_capitalization_score("123 456")
        self.assertEqual(score, 0.0)
    
    def test_whitespace_ratio(self):
        """Test whitespace ratio calculation."""
        # No whitespace
        ratio = self.preserver._calculate_whitespace_ratio("HelloWorld")
        self.assertEqual(ratio, 0.0)
        
        # All whitespace
        ratio = self.preserver._calculate_whitespace_ratio("   ")
        self.assertEqual(ratio, 1.0)
        
        # Mixed
        ratio = self.preserver._calculate_whitespace_ratio("Hello World")
        self.assertAlmostEqual(ratio, 1.0/11.0)


class TestTextBlockGrouper(unittest.TestCase):
    """Test cases for TextBlockGrouper class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {}
        self.grouper = TextBlockGrouper(self.config)
    
    def test_group_empty_list(self):
        """Test grouping with empty list."""
        result = self.grouper.group_related_blocks([])
        self.assertEqual(result, [])
    
    def test_group_single_block(self):
        """Test grouping with single block."""
        block = TextBlock("Test", 1, (0, 0, 10, 10), 12.0, "Arial", 0)
        result = self.grouper.group_related_blocks([block])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, "Test")
    
    def test_should_group_blocks_same_page(self):
        """Test block grouping logic for same page."""
        block1 = TextBlock("Hello", 1, (0, 0, 50, 10), 12.0, "Arial", 0)
        block2 = TextBlock("World", 1, (55, 0, 100, 10), 12.0, "Arial", 0)
        
        # Should group - same line, close horizontally
        self.assertTrue(self.grouper._should_group_blocks(block1, block2))
    
    def test_should_group_blocks_different_pages(self):
        """Test block grouping logic for different pages."""
        block1 = TextBlock("Hello", 1, (0, 0, 50, 10), 12.0, "Arial", 0)
        block2 = TextBlock("World", 2, (0, 0, 50, 10), 12.0, "Arial", 0)
        
        # Should not group - different pages
        self.assertFalse(self.grouper._should_group_blocks(block1, block2))
    
    def test_similar_font_characteristics(self):
        """Test font similarity detection."""
        block1 = TextBlock("Text1", 1, (0, 0, 50, 10), 12.0, "Arial", 0)
        block2 = TextBlock("Text2", 1, (55, 0, 100, 10), 12.0, "Arial", 0)
        block3 = TextBlock("Text3", 1, (55, 0, 100, 10), 16.0, "Arial", 0)  # Different size
        block4 = TextBlock("Text4", 1, (55, 0, 100, 10), 12.0, "Times", 0)  # Different font
        
        # Similar fonts
        self.assertTrue(self.grouper._similar_font_characteristics(block1, block2))
        
        # Different font size (too large difference)
        self.assertFalse(self.grouper._similar_font_characteristics(block1, block3))
        
        # Different font name
        self.assertFalse(self.grouper._similar_font_characteristics(block1, block4))
    
    def test_spatially_close(self):
        """Test spatial proximity detection."""
        # Same line, close horizontally
        block1 = TextBlock("Hello", 1, (0, 0, 50, 10), 12.0, "Arial", 0)
        block2 = TextBlock("World", 1, (55, 0, 100, 10), 12.0, "Arial", 0)
        self.assertTrue(self.grouper._spatially_close(block1, block2))
        
        # Same line, too far horizontally
        block3 = TextBlock("Far", 1, (200, 0, 250, 10), 12.0, "Arial", 0)
        self.assertFalse(self.grouper._spatially_close(block1, block3))
        
        # Vertically adjacent with overlap
        block4 = TextBlock("Below", 1, (10, 12, 60, 22), 12.0, "Arial", 0)
        self.assertTrue(self.grouper._spatially_close(block1, block4))
    
    def test_merge_blocks(self):
        """Test block merging functionality."""
        # Single block
        block1 = TextBlock("Hello", 1, (0, 0, 50, 10), 12.0, "Arial", 0)
        result = self.grouper._merge_blocks([block1])
        self.assertEqual(result.text, "Hello")
        
        # Multiple blocks on same line
        block2 = TextBlock("World", 1, (55, 0, 100, 10), 12.0, "Arial", 0)
        result = self.grouper._merge_blocks([block1, block2])
        self.assertEqual(result.text, "Hello World")
        
        # Multiple blocks on different lines
        block3 = TextBlock("Below", 1, (0, 15, 50, 25), 12.0, "Arial", 0)
        result = self.grouper._merge_blocks([block1, block3])
        self.assertEqual(result.text, "Hello\nBelow")
        
        # Check merged bounding box
        self.assertEqual(result.bbox, (0, 0, 50, 25))


class TestTextPreprocessor(unittest.TestCase):
    """Test cases for TextPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = TextPreprocessor()
    
    def test_preprocess_empty_list(self):
        """Test preprocessing with empty list."""
        result = self.preprocessor.preprocess_blocks([])
        self.assertEqual(result, [])
    
    def test_preprocess_valid_blocks(self):
        """Test preprocessing with valid blocks."""
        blocks = [
            TextBlock("  Hello World  ", 1, (0, 0, 100, 10), 12.0, "Arial", 0),
            TextBlock("Test Text", 1, (0, 15, 100, 25), 14.0, "Arial", 16)
        ]
        
        result = self.preprocessor.preprocess_blocks(blocks)
        
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], ProcessedBlock)
        self.assertEqual(result[0].text, "Hello World")
        self.assertEqual(result[1].text, "Test Text")
    
    def test_filter_valid_blocks(self):
        """Test filtering of valid blocks."""
        blocks = [
            TextBlock("Valid text", 1, (0, 0, 100, 10), 12.0, "Arial", 0),
            TextBlock("", 1, (0, 15, 100, 25), 12.0, "Arial", 0),  # Empty
            TextBlock("   ", 1, (0, 30, 100, 40), 12.0, "Arial", 0),  # Whitespace only
            TextBlock("AB", 1, (0, 45, 100, 55), 12.0, "Arial", 0),  # Too short (if min_length > 2)
            TextBlock("Invalid font", 1, (0, 60, 100, 70), 0, "Arial", 0),  # Invalid font size
        ]
        
        valid_blocks = self.preprocessor._filter_valid_blocks(blocks)
        
        # Should keep valid text and "AB" (assuming min_length <= 2)
        self.assertGreaterEqual(len(valid_blocks), 1)
        self.assertEqual(valid_blocks[0].text, "Valid text")
    
    def test_integration_with_multilingual_text(self):
        """Test preprocessing with multilingual content."""
        blocks = [
            TextBlock("English Text", 1, (0, 0, 100, 10), 12.0, "Arial", 0),
            TextBlock("Texto en Español", 1, (0, 15, 100, 25), 12.0, "Arial", 0),
            TextBlock("Texte en Français", 1, (0, 30, 100, 40), 12.0, "Arial", 0),
            TextBlock("中文文本", 1, (0, 45, 100, 55), 12.0, "Arial", 0),
        ]
        
        result = self.preprocessor.preprocess_blocks(blocks)
        
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0].text, "English Text")
        self.assertEqual(result[1].text, "Texto en Español")
        self.assertEqual(result[2].text, "Texte en Français")
        self.assertEqual(result[3].text, "中文文本")
    
    def test_integration_with_formatting_scenarios(self):
        """Test preprocessing with various formatting scenarios."""
        blocks = [
            # Title-like text (large, bold)
            TextBlock("DOCUMENT TITLE", 1, (0, 0, 200, 20), 18.0, "Arial", 16),
            
            # Heading-like text (medium, bold)
            TextBlock("Chapter 1: Introduction", 1, (0, 30, 180, 45), 14.0, "Arial", 16),
            
            # Regular text
            TextBlock("This is regular paragraph text.", 1, (0, 60, 300, 75), 12.0, "Arial", 0),
            
            # Bullet point
            TextBlock("• First bullet point", 1, (20, 90, 200, 105), 12.0, "Arial", 0),
            
            # Numbered item
            TextBlock("1) First numbered item", 1, (20, 120, 200, 135), 12.0, "Arial", 0),
        ]
        
        result = self.preprocessor.preprocess_blocks(blocks)
        
        self.assertEqual(len(result), 5)
        
        # Check that title has high font size ratio
        title_block = result[0]
        self.assertGreater(title_block.features.font_size_ratio, 1.0)
        self.assertTrue(title_block.features.is_bold)
        self.assertGreater(title_block.features.capitalization_score, 0.5)
        
        # Check that heading has medium font size ratio
        heading_block = result[1]
        self.assertGreater(heading_block.features.font_size_ratio, 1.0)
        self.assertTrue(heading_block.features.is_bold)
        
        # Check that regular text has normal characteristics
        regular_block = result[2]
        self.assertLessEqual(regular_block.features.font_size_ratio, 1.0)
        self.assertFalse(regular_block.features.is_bold)
        
        # Check that bullet point is normalized
        bullet_block = result[3]
        self.assertIn("•", bullet_block.text)
        
        # Check that numbered item is normalized
        numbered_block = result[4]
        self.assertIn("1.", numbered_block.text)


if __name__ == '__main__':
    # Set up logging to reduce noise during testing
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # Run the tests
    unittest.main(verbosity=2)