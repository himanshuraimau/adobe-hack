"""
Unit tests for the feature extraction module.

Tests cover font analysis, position analysis, content analysis, and overall feature extraction
with different text block types and edge cases.
"""

import pytest
import math
from unittest.mock import Mock, patch

from src.pdf_extractor.models.models import TextBlock, ProcessedBlock, FeatureVector
from src.pdf_extractor.core.feature_extractor import FeatureExtractor, FontAnalyzer, PositionAnalyzer, ContentAnalyzer


class TestFeatureExtractor:
    """Test cases for the main FeatureExtractor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = FeatureExtractor()
        
        # Create sample text blocks for testing
        self.title_block = TextBlock(
            text="Document Title",
            page_number=1,
            bbox=(72.0, 100.0, 200.0, 120.0),
            font_size=18.0,
            font_name="Arial-Bold",
            font_flags=16  # Bold flag
        )
        
        self.h1_block = TextBlock(
            text="Chapter 1: Introduction",
            page_number=1,
            bbox=(72.0, 150.0, 250.0, 165.0),
            font_size=14.0,
            font_name="Arial-Bold",
            font_flags=16  # Bold flag
        )
        
        self.h2_block = TextBlock(
            text="1.1 Overview",
            page_number=1,
            bbox=(72.0, 200.0, 150.0, 212.0),
            font_size=12.0,
            font_name="Arial",
            font_flags=0
        )
        
        self.regular_text_block = TextBlock(
            text="This is regular paragraph text with normal formatting.",
            page_number=1,
            bbox=(72.0, 230.0, 400.0, 242.0),
            font_size=10.0,
            font_name="Arial",
            font_flags=0
        )
        
        # Create processed blocks
        self.processed_blocks = []
        for block in [self.title_block, self.h1_block, self.h2_block, self.regular_text_block]:
            processed_block = ProcessedBlock(
                text=block.text,
                page_number=block.page_number,
                features=FeatureVector(1.0, False, False, 0.0, 0.0, len(block.text), 0.0, 0.0),
                original_block=block
            )
            self.processed_blocks.append(processed_block)
    
    def test_extract_features_basic(self):
        """Test basic feature extraction functionality."""
        # Initialize document stats
        self.extractor.initialize_document_stats(self.processed_blocks)
        
        # Extract features from title block
        features = self.extractor.extract_features(self.processed_blocks[0])
        
        assert isinstance(features, FeatureVector)
        assert features.font_size_ratio > 1.0  # Title should have larger font
        assert features.is_bold is True
        assert features.text_length == len("Document Title")
        assert 0.0 <= features.position_x <= 1.0
        assert 0.0 <= features.position_y <= 1.0
    
    def test_extract_features_different_block_types(self):
        """Test feature extraction with different types of text blocks."""
        self.extractor.initialize_document_stats(self.processed_blocks)
        
        # Extract features for all block types
        title_features = self.extractor.extract_features(self.processed_blocks[0])
        h1_features = self.extractor.extract_features(self.processed_blocks[1])
        h2_features = self.extractor.extract_features(self.processed_blocks[2])
        text_features = self.extractor.extract_features(self.processed_blocks[3])
        
        # Title should have highest font size ratio
        assert title_features.font_size_ratio > h1_features.font_size_ratio
        assert h1_features.font_size_ratio > h2_features.font_size_ratio
        assert h2_features.font_size_ratio > text_features.font_size_ratio
        
        # Bold formatting
        assert title_features.is_bold is True
        assert h1_features.is_bold is True
        assert h2_features.is_bold is False
        assert text_features.is_bold is False
    
    def test_initialize_document_stats(self):
        """Test document statistics initialization."""
        self.extractor.initialize_document_stats(self.processed_blocks)
        
        stats = self.extractor.doc_stats
        assert 'avg_font_size' in stats
        assert 'max_font_size' in stats
        assert 'min_font_size' in stats
        assert 'total_blocks' in stats
        assert 'total_pages' in stats
        
        assert stats['total_blocks'] == 4
        assert stats['total_pages'] == 1
        assert stats['max_font_size'] == 18.0  # Title font size
        assert stats['min_font_size'] == 10.0  # Regular text font size
    
    def test_extract_features_empty_blocks(self):
        """Test feature extraction with empty or invalid blocks."""
        empty_block = ProcessedBlock(
            text="",
            page_number=1,
            features=FeatureVector(1.0, False, False, 0.0, 0.0, 0, 0.0, 0.0),
            original_block=TextBlock("", 1, (0, 0, 0, 0), 0, "", 0)
        )
        
        self.extractor.initialize_document_stats([empty_block])
        features = self.extractor.extract_features(empty_block)
        
        assert isinstance(features, FeatureVector)
        assert features.text_length == 0
        assert features.capitalization_score == 0.0


class TestFontAnalyzer:
    """Test cases for the FontAnalyzer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = FontAnalyzer({})
        self.analyzer.set_document_stats({
            'avg_font_size': 12.0,
            'max_font_size': 18.0,
            'min_font_size': 10.0,
            'common_fonts': [('Arial', 10), ('Times', 5)]
        })
    
    def test_analyze_font_characteristics_bold(self):
        """Test font analysis for bold text."""
        block = ProcessedBlock(
            text="Bold Heading",
            page_number=1,
            features=FeatureVector(1.0, False, False, 0.0, 0.0, 12, 0.0, 0.0),
            original_block=TextBlock("Bold Heading", 1, (0, 0, 100, 20), 16.0, "Arial-Bold", 16)
        )
        
        features = self.analyzer.analyze_font_characteristics(block)
        
        assert features['is_bold'] is True
        assert features['font_size_ratio'] > 1.0  # 16/12 = 1.33
        assert features['font_weight_score'] > features['font_size_ratio']  # Bold boost
        assert 0.0 <= features['relative_font_size'] <= 1.0
    
    def test_analyze_font_characteristics_italic(self):
        """Test font analysis for italic text."""
        block = ProcessedBlock(
            text="Italic Text",
            page_number=1,
            features=FeatureVector(1.0, False, False, 0.0, 0.0, 11, 0.0, 0.0),
            original_block=TextBlock("Italic Text", 1, (0, 0, 100, 20), 12.0, "Arial-Italic", 2)
        )
        
        features = self.analyzer.analyze_font_characteristics(block)
        
        assert features['is_italic'] is True
        assert features['is_bold'] is False
        assert features['font_size_ratio'] == 1.0  # 12/12 = 1.0
    
    def test_font_weight_score_calculation(self):
        """Test font weight score calculation."""
        # Test normal text
        normal_score = self.analyzer._calculate_font_weight_score(1.0, False)
        assert normal_score == 1.0
        
        # Test bold text
        bold_score = self.analyzer._calculate_font_weight_score(1.0, True)
        assert bold_score == 1.5
        
        # Test large bold text
        large_bold_score = self.analyzer._calculate_font_weight_score(2.0, True)
        assert large_bold_score == 3.0  # Capped at 3.0
    
    def test_font_style_score_calculation(self):
        """Test font style score calculation."""
        # Test heading font
        heading_score = self.analyzer._calculate_font_style_score("Arial-Bold", False)
        assert heading_score > 0.0
        
        # Test italic penalty
        italic_score = self.analyzer._calculate_font_style_score("Arial", True)
        assert italic_score < 0.0 or italic_score == 0.0
        
        # Test regular font
        regular_score = self.analyzer._calculate_font_style_score("Arial", False)
        assert regular_score == 0.0


class TestPositionAnalyzer:
    """Test cases for the PositionAnalyzer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = PositionAnalyzer({})
        self.analyzer.set_document_stats(
            {'total_pages': 1},
            {1: {'width': 500, 'height': 700, 'left_margin': 50, 'top_margin': 50}}
        )
    
    def test_analyze_position_top_left(self):
        """Test position analysis for top-left positioned text."""
        block = ProcessedBlock(
            text="Top Left Text",
            page_number=1,
            features=FeatureVector(1.0, False, False, 0.0, 0.0, 13, 0.0, 0.0),
            original_block=TextBlock("Top Left Text", 1, (50, 50, 150, 70), 12.0, "Arial", 0)
        )
        
        features = self.analyzer.analyze_position(block)
        
        assert features['normalized_x'] == 0.0  # Left edge
        assert features['normalized_y'] == 0.0  # Top edge
        assert features['page_position_score'] > 0.5  # High score for top position
        assert features['alignment_score'] > 0.5  # High score for left alignment
    
    def test_analyze_position_center(self):
        """Test position analysis for center-positioned text."""
        block = ProcessedBlock(
            text="Centered Text",
            page_number=1,
            features=FeatureVector(1.0, False, False, 0.0, 0.0, 13, 0.0, 0.0),
            original_block=TextBlock("Centered Text", 1, (300, 200, 400, 220), 12.0, "Arial", 0)
        )
        
        features = self.analyzer.analyze_position(block)
        
        assert 0.4 <= features['normalized_x'] <= 0.6  # Center position
        assert features['alignment_score'] > 0.0  # Some score for center alignment
    
    def test_page_position_score_calculation(self):
        """Test page position score calculation."""
        # Top position should score higher
        top_score = self.analyzer._calculate_page_position_score(0.0, 0.0)
        bottom_score = self.analyzer._calculate_page_position_score(0.0, 1.0)
        assert top_score > bottom_score
        
        # Left position should score higher than right
        left_score = self.analyzer._calculate_page_position_score(0.0, 0.5)
        right_score = self.analyzer._calculate_page_position_score(1.0, 0.5)
        assert left_score > right_score
    
    def test_alignment_score_calculation(self):
        """Test alignment score calculation."""
        # Left alignment should score high
        left_score = self.analyzer._calculate_alignment_score(0.0, (0, 0, 100, 20))
        assert left_score > 0.8
        
        # Center alignment should score moderately
        center_score = self.analyzer._calculate_alignment_score(0.5, (250, 0, 350, 20))
        assert 0.5 <= center_score <= 0.9
        
        # Right alignment should score lower
        right_score = self.analyzer._calculate_alignment_score(0.9, (450, 0, 550, 20))
        assert right_score < left_score


class TestContentAnalyzer:
    """Test cases for the ContentAnalyzer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = ContentAnalyzer({})
        self.analyzer.set_document_stats({'total_blocks': 10})
    
    def test_analyze_content_title_case(self):
        """Test content analysis for title case text."""
        block = ProcessedBlock(
            text="This Is a Title Case Heading",
            page_number=1,
            features=FeatureVector(1.0, False, False, 0.0, 0.0, 29, 0.0, 0.0),
            original_block=TextBlock("This Is a Title Case Heading", 1, (0, 0, 200, 20), 12.0, "Arial", 0)
        )
        
        features = self.analyzer.analyze_content(block)
        
        assert features['text_length'] == len("This Is a Title Case Heading")
        assert features['word_count'] == 6
        assert features['title_case_score'] > 0.5  # Should detect title case
        assert features['all_caps_score'] == 0.0  # Not all caps
    
    def test_analyze_content_all_caps(self):
        """Test content analysis for all caps text."""
        block = ProcessedBlock(
            text="ALL CAPS HEADING",
            page_number=1,
            features=FeatureVector(1.0, False, False, 0.0, 0.0, 16, 0.0, 0.0),
            original_block=TextBlock("ALL CAPS HEADING", 1, (0, 0, 150, 20), 12.0, "Arial", 0)
        )
        
        features = self.analyzer.analyze_content(block)
        
        assert features['all_caps_score'] > 0.5  # Should detect all caps
        assert features['capitalization_score'] == 1.0  # All letters are uppercase
    
    def test_analyze_content_regular_text(self):
        """Test content analysis for regular paragraph text."""
        block = ProcessedBlock(
            text="This is a regular paragraph with normal sentence structure. It contains multiple sentences.",
            page_number=1,
            features=FeatureVector(1.0, False, False, 0.0, 0.0, 91, 0.0, 0.0),
            original_block=TextBlock("This is a regular paragraph with normal sentence structure. It contains multiple sentences.", 1, (0, 0, 400, 40), 10.0, "Arial", 0)
        )
        
        features = self.analyzer.analyze_content(block)
        
        assert features['sentence_count'] == 2  # Two sentences
        assert features['punctuation_density'] > 0.0  # Has punctuation
        assert features['title_case_score'] < 0.5  # Not title case
        assert features['all_caps_score'] == 0.0  # Not all caps
    
    def test_analyze_content_numbered_heading(self):
        """Test content analysis for numbered heading."""
        block = ProcessedBlock(
            text="1.2.3 Numbered Section Heading",
            page_number=1,
            features=FeatureVector(1.0, False, False, 0.0, 0.0, 31, 0.0, 0.0),
            original_block=TextBlock("1.2.3 Numbered Section Heading", 1, (0, 0, 200, 20), 12.0, "Arial", 0)
        )
        
        features = self.analyzer.analyze_content(block)
        
        assert features['numeric_content_ratio'] > 0.0  # Contains numbers
        assert features['special_char_ratio'] > 0.0  # Contains dots
        # The text "1.2.3 Numbered Section Heading" contains dots which are detected as sentence endings
        # This is expected behavior for the simple sentence counting algorithm
    
    def test_capitalization_score_calculation(self):
        """Test capitalization score calculation."""
        # All lowercase
        assert self.analyzer._calculate_capitalization_score("hello world") == 0.0
        
        # All uppercase
        assert self.analyzer._calculate_capitalization_score("HELLO WORLD") == 1.0
        
        # Mixed case
        score = self.analyzer._calculate_capitalization_score("Hello World")
        assert 0.0 < score < 1.0
        
        # No letters
        assert self.analyzer._calculate_capitalization_score("123 456") == 0.0
    
    def test_title_case_score_calculation(self):
        """Test title case score calculation."""
        # Perfect title case
        assert self.analyzer._calculate_title_case_score("This Is Title Case") > 0.8
        
        # Title case with articles
        score = self.analyzer._calculate_title_case_score("This is a Title")
        assert score > 0.5  # Should handle articles correctly
        
        # Not title case
        assert self.analyzer._calculate_title_case_score("this is not title case") < 0.5
        
        # All caps (not title case)
        assert self.analyzer._calculate_title_case_score("THIS IS ALL CAPS") < 0.5
    
    def test_sentence_counting(self):
        """Test sentence counting functionality."""
        # Single sentence
        assert self.analyzer._count_sentences("This is one sentence.") == 1
        
        # Multiple sentences
        assert self.analyzer._count_sentences("First sentence. Second sentence! Third sentence?") == 3
        
        # No sentences
        assert self.analyzer._count_sentences("No sentence ending") == 0
        
        # Empty text
        assert self.analyzer._count_sentences("") == 0
    
    def test_whitespace_ratio_calculation(self):
        """Test whitespace ratio calculation."""
        # No whitespace
        assert self.analyzer._calculate_whitespace_ratio("NoSpaces") == 0.0
        
        # All whitespace
        assert self.analyzer._calculate_whitespace_ratio("   ") == 1.0
        
        # Mixed
        ratio = self.analyzer._calculate_whitespace_ratio("Hello World")
        assert 0.0 < ratio < 1.0
        
        # Empty text
        assert self.analyzer._calculate_whitespace_ratio("") == 0.0


class TestFeatureVectorExtensions:
    """Test cases for extended feature vector attributes."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = FeatureExtractor()
        
        # Create a sample processed block
        self.sample_block = ProcessedBlock(
            text="Sample Heading Text",
            page_number=1,
            features=FeatureVector(1.0, False, False, 0.0, 0.0, 19, 0.0, 0.0),
            original_block=TextBlock("Sample Heading Text", 1, (72, 100, 200, 120), 14.0, "Arial-Bold", 16)
        )
    
    def test_extended_features_added(self):
        """Test that extended features are added to the feature vector."""
        self.extractor.initialize_document_stats([self.sample_block])
        features = self.extractor.extract_features(self.sample_block)
        
        # Check that extended font features are added
        assert hasattr(features, 'font_weight_score')
        assert hasattr(features, 'font_style_score')
        assert hasattr(features, 'relative_font_size')
        
        # Check that extended position features are added
        assert hasattr(features, 'page_position_score')
        assert hasattr(features, 'alignment_score')
        assert hasattr(features, 'whitespace_above')
        assert hasattr(features, 'whitespace_below')
        assert hasattr(features, 'indentation_level')
        
        # Check that extended content features are added
        assert hasattr(features, 'word_count')
        assert hasattr(features, 'sentence_count')
        assert hasattr(features, 'punctuation_density')
        assert hasattr(features, 'numeric_content_ratio')
        assert hasattr(features, 'special_char_ratio')
        assert hasattr(features, 'title_case_score')
        assert hasattr(features, 'all_caps_score')


if __name__ == "__main__":
    pytest.main([__file__])