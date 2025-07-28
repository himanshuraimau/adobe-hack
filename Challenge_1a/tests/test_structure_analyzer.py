"""
Unit tests for the structure analyzer module.

Tests various document structures, hierarchy building, title detection,
and edge cases like missing levels and inconsistent formatting.
"""

import unittest
from unittest.mock import Mock, patch
from typing import List

from src.pdf_extractor.core.structure_analyzer import StructureAnalyzer, HierarchyBuilder, TitleDetector
from src.pdf_extractor.models.models import (
    ClassificationResult, ProcessedBlock, FeatureVector, TextBlock, 
    Heading, DocumentStructure
)


class TestStructureAnalyzer(unittest.TestCase):
    """Test cases for StructureAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = StructureAnalyzer()
    
    def _create_mock_classification_result(
        self, 
        text: str, 
        predicted_class: str, 
        confidence: float,
        page: int = 1,
        font_size_ratio: float = 1.0,
        is_bold: bool = False,
        position_y: float = 0.5
    ) -> ClassificationResult:
        """Helper to create mock classification results."""
        features = FeatureVector(
            font_size_ratio=font_size_ratio,
            is_bold=is_bold,
            is_italic=False,
            position_x=0.1,
            position_y=position_y,
            text_length=len(text),
            capitalization_score=0.0,
            whitespace_ratio=0.1
        )
        
        text_block = TextBlock(
            text=text,
            page_number=page,
            bbox=(0, 0, 100, 20),
            font_size=12.0,
            font_name="Arial",
            font_flags=0
        )
        
        processed_block = ProcessedBlock(
            text=text,
            page_number=page,
            features=features,
            original_block=text_block
        )
        
        return ClassificationResult(
            block=processed_block,
            predicted_class=predicted_class,
            confidence=confidence
        )
    
    def test_analyze_structure_with_complete_hierarchy(self):
        """Test structure analysis with complete H1-H2-H3 hierarchy."""
        classified_blocks = [
            self._create_mock_classification_result("Document Title", "title", 0.9, 1, 2.0, True, 0.1),
            self._create_mock_classification_result("Chapter 1", "h1", 0.8, 1, 1.8, True, 0.3),
            self._create_mock_classification_result("Section 1.1", "h2", 0.7, 2, 1.5, True, 0.2),
            self._create_mock_classification_result("Subsection 1.1.1", "h3", 0.6, 2, 1.2, False, 0.4),
            self._create_mock_classification_result("Regular text", "text", 0.9, 2, 1.0, False, 0.6),
            self._create_mock_classification_result("Chapter 2", "h1", 0.8, 3, 1.8, True, 0.1),
        ]
        
        result = self.analyzer.analyze_structure(classified_blocks)
        
        self.assertEqual(result.title, "Document Title")
        self.assertEqual(len(result.headings), 4)  # Title filtered out
        
        # Check heading levels
        heading_levels = [h.level for h in result.headings]
        self.assertIn("H1", heading_levels)
        self.assertIn("H2", heading_levels)
        self.assertIn("H3", heading_levels)
        
        # Check metadata
        self.assertEqual(result.metadata['total_blocks'], 6)
        self.assertEqual(result.metadata['heading_blocks'], 5)
        self.assertTrue(result.metadata['title_detected'])
    
    def test_analyze_structure_no_headings(self):
        """Test structure analysis with no headings detected."""
        classified_blocks = [
            self._create_mock_classification_result("Regular text 1", "text", 0.9),
            self._create_mock_classification_result("Regular text 2", "text", 0.8),
        ]
        
        result = self.analyzer.analyze_structure(classified_blocks)
        
        self.assertIsNone(result.title)
        self.assertEqual(len(result.headings), 0)
        self.assertEqual(result.metadata['heading_blocks'], 0)
    
    def test_analyze_structure_empty_input(self):
        """Test structure analysis with empty input."""
        result = self.analyzer.analyze_structure([])
        
        self.assertIsNone(result.title)
        self.assertEqual(len(result.headings), 0)
        self.assertEqual(result.metadata['total_blocks'], 0)
    
    def test_analyze_structure_with_error_handling(self):
        """Test error handling in structure analysis."""
        # Create a mock that raises an exception
        with patch.object(self.analyzer.title_detector, 'detect_title', side_effect=Exception("Test error")):
            classified_blocks = [
                self._create_mock_classification_result("Test Heading", "h1", 0.8)
            ]
            
            result = self.analyzer.analyze_structure(classified_blocks)
            
            # Should return minimal structure on error
            self.assertIsNone(result.title)
            self.assertEqual(len(result.headings), 0)
            self.assertIn('error', result.metadata)


class TestHierarchyBuilder(unittest.TestCase):
    """Test cases for HierarchyBuilder class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.builder = HierarchyBuilder()
    
    def _create_heading(self, text: str, level: str, page: int = 1, confidence: float = 0.8) -> Heading:
        """Helper to create Heading objects."""
        return Heading(
            level=level,
            text=text,
            page=page,
            confidence=confidence
        )
    
    def test_build_hierarchy_proper_order(self):
        """Test hierarchy building with proper H1-H2-H3 order."""
        headings = [
            self._create_heading("Chapter 1", "H1", 1),
            self._create_heading("Section 1.1", "H2", 1),
            self._create_heading("Subsection 1.1.1", "H3", 2),
            self._create_heading("Chapter 2", "H1", 3),
        ]
        
        result = self.builder.build_hierarchy(headings)
        
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0].level, "H1")
        self.assertEqual(result[1].level, "H2")
        self.assertEqual(result[2].level, "H3")
        self.assertEqual(result[3].level, "H1")
    
    def test_build_hierarchy_missing_h1(self):
        """Test hierarchy building when H1 is missing."""
        headings = [
            self._create_heading("Section 1.1", "H2", 1),
            self._create_heading("Subsection 1.1.1", "H3", 1),
            self._create_heading("Section 1.2", "H2", 2),
        ]
        
        result = self.builder.build_hierarchy(headings)
        
        # First H2 should be promoted to H1
        self.assertEqual(result[0].level, "H1")
        self.assertEqual(result[1].level, "H2")  # H3 promoted to H2
        self.assertEqual(result[2].level, "H2")
    
    def test_build_hierarchy_missing_h2(self):
        """Test hierarchy building when H2 is missing."""
        headings = [
            self._create_heading("Chapter 1", "H1", 1),
            self._create_heading("Subsection 1.1.1", "H3", 1),
            self._create_heading("Chapter 2", "H1", 2),
        ]
        
        result = self.builder.build_hierarchy(headings)
        
        self.assertEqual(result[0].level, "H1")
        self.assertEqual(result[1].level, "H2")  # H3 promoted to H2
        self.assertEqual(result[2].level, "H1")
    
    def test_build_hierarchy_empty_input(self):
        """Test hierarchy building with empty input."""
        result = self.builder.build_hierarchy([])
        self.assertEqual(len(result), 0)
    
    def test_build_hierarchy_page_ordering(self):
        """Test that headings are ordered by page number."""
        headings = [
            self._create_heading("Chapter 2", "H1", 3),
            self._create_heading("Chapter 1", "H1", 1),
            self._create_heading("Section 1.1", "H2", 2),
        ]
        
        result = self.builder.build_hierarchy(headings)
        
        # Should be ordered by page
        self.assertEqual(result[0].text, "Chapter 1")
        self.assertEqual(result[1].text, "Section 1.1")
        self.assertEqual(result[2].text, "Chapter 2")
    
    def test_normalize_hierarchy_levels(self):
        """Test hierarchy level normalization."""
        headings = [
            self._create_heading("High confidence H1", "H1", 1, 0.9),
            self._create_heading("Low confidence H1", "H1", 1, 0.3),
            self._create_heading("Medium confidence H2", "H2", 2, 0.6),
        ]
        
        result = self.builder._normalize_hierarchy_levels(headings)
        
        # Should maintain structure but potentially adjust based on confidence
        self.assertEqual(len(result), 3)
        for heading in result:
            self.assertIn(heading.level, ["H1", "H2", "H3"])


class TestTitleDetector(unittest.TestCase):
    """Test cases for TitleDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = TitleDetector()
    
    def _create_mock_classification_result(
        self, 
        text: str, 
        predicted_class: str, 
        confidence: float,
        page: int = 1,
        font_size_ratio: float = 1.0,
        position_y: float = 0.5
    ) -> ClassificationResult:
        """Helper to create mock classification results."""
        features = FeatureVector(
            font_size_ratio=font_size_ratio,
            is_bold=True,
            is_italic=False,
            position_x=0.1,
            position_y=position_y,
            text_length=len(text),
            capitalization_score=0.0,
            whitespace_ratio=0.1
        )
        
        text_block = TextBlock(
            text=text,
            page_number=page,
            bbox=(0, 0, 100, 20),
            font_size=12.0,
            font_name="Arial",
            font_flags=0
        )
        
        processed_block = ProcessedBlock(
            text=text,
            page_number=page,
            features=features,
            original_block=text_block
        )
        
        return ClassificationResult(
            block=processed_block,
            predicted_class=predicted_class,
            confidence=confidence
        )
    
    def test_detect_title_explicit_title_class(self):
        """Test title detection with explicit title classification."""
        classified_blocks = [
            self._create_mock_classification_result("Document Title", "title", 0.9),
            self._create_mock_classification_result("Chapter 1", "h1", 0.8),
        ]
        
        result = self.detector.detect_title(classified_blocks)
        self.assertEqual(result, "Document Title")
    
    def test_detect_title_multiple_titles_highest_confidence(self):
        """Test title detection with multiple title candidates."""
        classified_blocks = [
            self._create_mock_classification_result("Low Confidence Title", "title", 0.5),
            self._create_mock_classification_result("High Confidence Title", "title", 0.9),
            self._create_mock_classification_result("Chapter 1", "h1", 0.8),
        ]
        
        result = self.detector.detect_title(classified_blocks)
        self.assertEqual(result, "High Confidence Title")
    
    def test_detect_title_first_heading_fallback(self):
        """Test title detection using first heading as fallback."""
        classified_blocks = [
            self._create_mock_classification_result("Chapter 1", "h1", 0.8, 1, 1.5, 0.2),
            self._create_mock_classification_result("Section 1.1", "h2", 0.7, 2, 1.2, 0.3),
        ]
        
        result = self.detector.detect_title(classified_blocks)
        self.assertEqual(result, "Chapter 1")
    
    def test_detect_title_largest_font_fallback(self):
        """Test title detection using largest font as fallback."""
        classified_blocks = [
            self._create_mock_classification_result("Small Heading", "h2", 0.7, 1, 1.2, 0.3),
            self._create_mock_classification_result("Large Heading", "h1", 0.8, 1, 2.0, 0.2),
            self._create_mock_classification_result("Medium Heading", "h2", 0.6, 2, 1.5, 0.4),
        ]
        
        result = self.detector.detect_title(classified_blocks)
        self.assertEqual(result, "Large Heading")
    
    def test_detect_title_no_headings(self):
        """Test title detection with no headings."""
        classified_blocks = [
            self._create_mock_classification_result("Regular text", "text", 0.9),
        ]
        
        result = self.detector.detect_title(classified_blocks)
        self.assertIsNone(result)
    
    def test_detect_title_empty_input(self):
        """Test title detection with empty input."""
        result = self.detector.detect_title([])
        self.assertIsNone(result)
    
    def test_find_first_heading(self):
        """Test finding the first heading in document order."""
        classified_blocks = [
            self._create_mock_classification_result("Chapter 2", "h1", 0.8, 2, 1.5, 0.1),
            self._create_mock_classification_result("Chapter 1", "h1", 0.8, 1, 1.5, 0.1),
            self._create_mock_classification_result("Section 1.1", "h2", 0.7, 1, 1.2, 0.3),
        ]
        
        result = self.detector._find_first_heading(classified_blocks)
        self.assertEqual(result.block.text, "Chapter 1")
    
    def test_find_largest_font_heading(self):
        """Test finding the largest font heading on first page."""
        classified_blocks = [
            self._create_mock_classification_result("Small Heading", "h3", 0.6, 1, 1.1, 0.5),
            self._create_mock_classification_result("Large Heading", "h1", 0.8, 1, 2.5, 0.2),
            self._create_mock_classification_result("Medium Heading", "h2", 0.7, 1, 1.5, 0.3),
            self._create_mock_classification_result("Page 2 Heading", "h1", 0.8, 2, 3.0, 0.1),  # Should be ignored
        ]
        
        result = self.detector._find_largest_font_heading(classified_blocks)
        self.assertEqual(result.block.text, "Large Heading")


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for various document structure scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = StructureAnalyzer()
    
    def _create_mock_classification_result(
        self, 
        text: str, 
        predicted_class: str, 
        confidence: float,
        page: int = 1,
        font_size_ratio: float = 1.0,
        is_bold: bool = False,
        position_y: float = 0.5
    ) -> ClassificationResult:
        """Helper to create mock classification results."""
        features = FeatureVector(
            font_size_ratio=font_size_ratio,
            is_bold=is_bold,
            is_italic=False,
            position_x=0.1,
            position_y=position_y,
            text_length=len(text),
            capitalization_score=0.0,
            whitespace_ratio=0.1
        )
        
        text_block = TextBlock(
            text=text,
            page_number=page,
            bbox=(0, 0, 100, 20),
            font_size=12.0,
            font_name="Arial",
            font_flags=0
        )
        
        processed_block = ProcessedBlock(
            text=text,
            page_number=page,
            features=features,
            original_block=text_block
        )
        
        return ClassificationResult(
            block=processed_block,
            predicted_class=predicted_class,
            confidence=confidence
        )
    
    def test_academic_paper_structure(self):
        """Test structure analysis for academic paper format."""
        classified_blocks = [
            self._create_mock_classification_result("Research Paper Title", "title", 0.9, 1, 2.2, True, 0.05),
            self._create_mock_classification_result("Abstract", "h1", 0.8, 1, 1.6, True, 0.2),
            self._create_mock_classification_result("1. Introduction", "h1", 0.8, 2, 1.6, True, 0.1),
            self._create_mock_classification_result("1.1 Background", "h2", 0.7, 2, 1.3, True, 0.3),
            self._create_mock_classification_result("1.2 Motivation", "h2", 0.7, 3, 1.3, True, 0.2),
            self._create_mock_classification_result("2. Methodology", "h1", 0.8, 4, 1.6, True, 0.1),
            self._create_mock_classification_result("2.1 Data Collection", "h2", 0.7, 4, 1.3, True, 0.4),
            self._create_mock_classification_result("3. Results", "h1", 0.8, 6, 1.6, True, 0.1),
        ]
        
        result = self.analyzer.analyze_structure(classified_blocks)
        
        self.assertEqual(result.title, "Research Paper Title")
        self.assertEqual(len(result.headings), 7)
        
        # Check that we have proper hierarchy
        h1_count = sum(1 for h in result.headings if h.level == "H1")
        h2_count = sum(1 for h in result.headings if h.level == "H2")
        
        self.assertGreaterEqual(h1_count, 3)  # Abstract, Introduction, Methodology, Results
        self.assertGreaterEqual(h2_count, 3)  # Background, Motivation, Data Collection
    
    def test_technical_manual_structure(self):
        """Test structure analysis for technical manual format."""
        classified_blocks = [
            self._create_mock_classification_result("USER MANUAL", "title", 0.9, 1, 2.5, True, 0.05),
            self._create_mock_classification_result("Chapter 1: Getting Started", "h1", 0.8, 2, 1.8, True, 0.1),
            self._create_mock_classification_result("Installation", "h2", 0.7, 2, 1.4, True, 0.3),
            self._create_mock_classification_result("System Requirements", "h3", 0.6, 2, 1.2, False, 0.5),
            self._create_mock_classification_result("Configuration", "h2", 0.7, 3, 1.4, True, 0.2),
            self._create_mock_classification_result("Chapter 2: Advanced Features", "h1", 0.8, 5, 1.8, True, 0.1),
        ]
        
        result = self.analyzer.analyze_structure(classified_blocks)
        
        self.assertEqual(result.title, "USER MANUAL")
        self.assertEqual(len(result.headings), 5)
        
        # Verify hierarchy progression
        levels = [h.level for h in result.headings]
        self.assertIn("H1", levels)
        self.assertIn("H2", levels)
        self.assertIn("H3", levels)
    
    def test_inconsistent_formatting_handling(self):
        """Test handling of documents with inconsistent formatting."""
        classified_blocks = [
            self._create_mock_classification_result("Document Title", "h1", 0.6, 1, 2.0, True, 0.1),  # Misclassified title
            self._create_mock_classification_result("First Section", "h2", 0.8, 1, 1.8, True, 0.3),  # Should be H1
            self._create_mock_classification_result("Subsection A", "h1", 0.5, 2, 1.2, False, 0.4),  # Should be H2
            self._create_mock_classification_result("Subsection B", "h3", 0.7, 2, 1.2, False, 0.6),  # Should be H2
            self._create_mock_classification_result("Second Section", "h2", 0.8, 3, 1.8, True, 0.2),  # Should be H1
        ]
        
        result = self.analyzer.analyze_structure(classified_blocks)
        
        # Should handle inconsistencies and create reasonable structure
        self.assertIsNotNone(result.title)
        self.assertGreater(len(result.headings), 0)
        
        # All headings should have valid levels
        for heading in result.headings:
            self.assertIn(heading.level, ["H1", "H2", "H3"])


if __name__ == '__main__':
    unittest.main()