"""
Test suite for the feature extraction system.

This module tests the feature extraction functionality to ensure it correctly
analyzes text blocks and generates appropriate features for heading classification.
"""

import unittest
from unittest.mock import Mock, patch
import sys
import os

# Add the current directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import TextBlock, ProcessedBlock, FeatureVector
from feature_extractor import FeatureExtractor, FontAnalyzer, PositionAnalyzer, ContentAnalyzer
from config import config


class TestFeatureExtractor(unittest.TestCase):
    """Test cases for the main FeatureExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = FeatureExtractor()
        
        # Create sample text blocks
        self.sample_text_block = TextBlock(
            text="Introduction to Machine Learning",
            page_number=1,
            bbox=(72.0, 100.0, 300.0, 120.0),
            font_size=16.0,
            font_name="Arial-Bold",
            font_flags=16  # Bold flag
        )
        
        self.sample_processed_block = ProcessedBlock(
            text="Introduction to Machine Learning",
            page_number=1,
            features=FeatureVector(
                font_size_ratio=1.33,
                is_bold=True,
                is_italic=False,
                position_x=0.3,
                position_y=0.15,
                text_length=32,
                capitalization_score=0.8,
                whitespace_ratio=0.15
            ),
            original_block=self.sample_text_block
        )
    
    def test_extract_features_single_block(self):
        """Test feature extraction for a single block."""
        # Set up document stats
        self.extractor.set_document_stats([self.sample_processed_block])
        
        # Extract features
        features = self.extractor.extract_features(self.sample_processed_block)
        
        # Verify feature vector structure
        self.assertIsInstance(features, FeatureVector)
        self.assertIsInstance(features.font_size_ratio, float)
        self.assertIsInstance(features.is_bold, bool)
        self.assertIsInstance(features.is_italic, bool)
        self.assertIsInstance(features.position_x, float)
        self.assertIsInstance(features.position_y, float)
        self.assertIsInstance(features.text_length, int)
        self.assertIsInstance(features.capitalization_score, float)
        self.assertIsInstance(features.whitespace_ratio, float)
        
        # Verify additional features are added
        self.assertTrue(hasattr(features, 'font_weight_score'))
        self.assertTrue(hasattr(features, 'alignment_score'))
        self.assertTrue(hasattr(features, 'page_position_score'))
        self.assertTrue(hasattr(features, 'punctuation_score'))
        self.assertTrue(hasattr(features, 'word_count'))
        self.assertTrue(hasattr(features, 'numeric_ratio'))
        self.assertTrue(hasattr(features, 'special_char_ratio'))
        self.assertTrue(hasattr(features, 'heading_pattern_score'))
        self.assertTrue(hasattr(features, 'length_score'))
    
    def test_extract_features_batch(self):
        """Test batch feature extraction."""
        blocks = [self.sample_processed_block] * 3
        
        features_list = self.extractor.extract_features_batch(blocks)
        
        self.assertEqual(len(features_list), 3)
        for features in features_list:
            self.assertIsInstance(features, FeatureVector)
    
    def test_document_stats_calculation(self):
        """Test document statistics calculation."""
        blocks = [self.sample_processed_block] * 3
        
        self.extractor.set_document_stats(blocks)
        stats = self.extractor.document_stats
        
        self.assertIn('avg_font_size', stats)
        self.assertIn('max_font_size', stats)
        self.assertIn('min_font_size', stats)
        self.assertIn('total_pages', stats)
        self.assertIn('total_blocks', stats)
        self.assertIn('avg_text_length', stats)
        self.assertIn('page_width', stats)
        self.assertIn('page_height', stats)
        
        self.assertEqual(stats['total_blocks'], 3)
        self.assertEqual(stats['avg_font_size'], 16.0)


class TestFontAnalyzer(unittest.TestCase):
    """Test cases for the FontAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = FontAnalyzer(config.get_feature_config())
        self.analyzer.set_document_stats({
            'avg_font_size': 12.0,
            'max_font_size': 18.0,
            'min_font_size': 10.0
        })
        
        self.bold_block = TextBlock(
            text="Bold Heading",
            page_number=1,
            bbox=(72.0, 100.0, 200.0, 120.0),
            font_size=16.0,
            font_name="Arial-Bold",
            font_flags=16  # Bold flag
        )
        
        self.processed_bold_block = ProcessedBlock(
            text="Bold Heading",
            page_number=1,
            features=Mock(),
            original_block=self.bold_block
        )
    
    def test_font_size_ratio_calculation(self):
        """Test font size ratio calculation."""
        ratio = self.analyzer._calculate_font_size_ratio(16.0)
        self.assertAlmostEqual(ratio, 16.0 / 12.0, places=2)
    
    def test_bold_font_detection(self):
        """Test bold font detection."""
        self.assertTrue(self.analyzer._is_bold_font(16))  # Bold flag set
        self.assertFalse(self.analyzer._is_bold_font(0))  # No flags set
    
    def test_italic_font_detection(self):
        """Test italic font detection."""
        self.assertTrue(self.analyzer._is_italic_font(2))  # Italic flag set
        self.assertFalse(self.analyzer._is_italic_font(0))  # No flags set
    
    def test_font_characteristics_analysis(self):
        """Test complete font characteristics analysis."""
        features = self.analyzer.analyze_font_characteristics(self.processed_bold_block)
        
        self.assertIn('font_size_ratio', features)
        self.assertIn('is_bold', features)
        self.assertIn('is_italic', features)
        self.assertIn('font_weight_score', features)
        self.assertIn('font_style_score', features)
        self.assertIn('relative_size_score', features)
        
        self.assertTrue(features['is_bold'])
        self.assertFalse(features['is_italic'])
        self.assertGreater(features['font_weight_score'], 0.0)


class TestPositionAnalyzer(unittest.TestCase):
    """Test cases for the PositionAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = PositionAnalyzer(config.get_feature_config())
        self.analyzer.set_document_stats({
            'page_width': 612,
            'page_height': 792,
            'total_pages': 5
        })
        
        self.centered_block = TextBlock(
            text="Centered Title",
            page_number=1,
            bbox=(200.0, 50.0, 400.0, 70.0),  # Centered horizontally
            font_size=18.0,
            font_name="Arial-Bold",
            font_flags=16
        )
        
        self.processed_centered_block = ProcessedBlock(
            text="Centered Title",
            page_number=1,
            features=Mock(),
            original_block=self.centered_block
        )
    
    def test_position_normalization(self):
        """Test position coordinate normalization."""
        bbox = (200.0, 50.0, 400.0, 70.0)
        norm_x, norm_y = self.analyzer._normalize_position(bbox)
        
        # Center of bbox: (300, 60)
        expected_x = 300.0 / 612.0
        expected_y = 60.0 / 792.0
        
        self.assertAlmostEqual(norm_x, expected_x, places=3)
        self.assertAlmostEqual(norm_y, expected_y, places=3)
    
    def test_alignment_score_calculation(self):
        """Test alignment score calculation."""
        # Centered bbox
        bbox = (206.0, 50.0, 406.0, 70.0)  # Center at x=306 (close to page center 306)
        score = self.analyzer._calculate_alignment_score(bbox)
        
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_page_position_score(self):
        """Test page position score calculation."""
        bbox = (200.0, 50.0, 400.0, 70.0)  # Near top of page
        score = self.analyzer._calculate_page_position_score(bbox, 1)  # First page
        
        self.assertGreater(score, 0.5)  # Should be high for top of first page
        self.assertLessEqual(score, 1.0)
    
    def test_position_analysis(self):
        """Test complete position analysis."""
        features = self.analyzer.analyze_position(self.processed_centered_block)
        
        self.assertIn('normalized_x', features)
        self.assertIn('normalized_y', features)
        self.assertIn('alignment_score', features)
        self.assertIn('page_position_score', features)
        self.assertIn('whitespace_above', features)
        self.assertIn('whitespace_below', features)
        
        self.assertGreaterEqual(features['normalized_x'], 0.0)
        self.assertLessEqual(features['normalized_x'], 1.0)
        self.assertGreaterEqual(features['normalized_y'], 0.0)
        self.assertLessEqual(features['normalized_y'], 1.0)


class TestContentAnalyzer(unittest.TestCase):
    """Test cases for the ContentAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = ContentAnalyzer(config.get_feature_config())
        self.analyzer.set_document_stats({
            'avg_text_length': 100
        })
        
        self.heading_block = ProcessedBlock(
            text="1. Introduction",
            page_number=1,
            features=Mock(),
            original_block=Mock()
        )
        
        self.title_block = ProcessedBlock(
            text="MACHINE LEARNING FUNDAMENTALS",
            page_number=1,
            features=Mock(),
            original_block=Mock()
        )
    
    def test_capitalization_score(self):
        """Test capitalization score calculation."""
        # All caps text
        score = self.analyzer._calculate_capitalization_score("HELLO WORLD")
        self.assertEqual(score, 1.0)
        
        # Mixed case text
        score = self.analyzer._calculate_capitalization_score("Hello World")
        self.assertEqual(score, 0.2)  # 2 out of 10 letters are uppercase
        
        # No letters
        score = self.analyzer._calculate_capitalization_score("123 456")
        self.assertEqual(score, 0.0)
    
    def test_punctuation_score(self):
        """Test punctuation score calculation."""
        # Colon ending (common in headings)
        score = self.analyzer._calculate_punctuation_score("Introduction:")
        self.assertGreater(score, 0.0)
        
        # Period ending (less common in headings)
        score = self.analyzer._calculate_punctuation_score("This is a sentence.")
        self.assertLess(score, 0.5)
        
        # No ending punctuation (common in headings)
        score = self.analyzer._calculate_punctuation_score("Chapter Title")
        self.assertGreater(score, 0.0)
    
    def test_heading_pattern_score(self):
        """Test heading pattern score calculation."""
        # Numbered heading
        score = self.analyzer._calculate_heading_pattern_score("1. Introduction")
        self.assertGreater(score, 0.0)
        
        # All caps heading
        score = self.analyzer._calculate_heading_pattern_score("CHAPTER ONE")
        self.assertGreater(score, 0.0)
        
        # Title case heading
        score = self.analyzer._calculate_heading_pattern_score("Machine Learning Basics")
        self.assertGreater(score, 0.0)
        
        # Regular text
        score = self.analyzer._calculate_heading_pattern_score("this is regular text without patterns")
        self.assertLessEqual(score, 0.2)
    
    def test_content_analysis(self):
        """Test complete content analysis."""
        features = self.analyzer.analyze_content(self.heading_block)
        
        self.assertIn('text_length', features)
        self.assertIn('word_count', features)
        self.assertIn('capitalization_score', features)
        self.assertIn('whitespace_ratio', features)
        self.assertIn('punctuation_score', features)
        self.assertIn('numeric_ratio', features)
        self.assertIn('special_char_ratio', features)
        self.assertIn('heading_pattern_score', features)
        self.assertIn('length_score', features)
        
        # Verify reasonable values
        self.assertGreater(features['text_length'], 0)
        self.assertGreater(features['word_count'], 0)
        self.assertGreaterEqual(features['capitalization_score'], 0.0)
        self.assertLessEqual(features['capitalization_score'], 1.0)
        self.assertGreaterEqual(features['heading_pattern_score'], 0.0)
        self.assertLessEqual(features['heading_pattern_score'], 1.0)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)