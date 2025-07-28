"""
Unit tests for the heading classification module.

Tests cover MobileBERT model loading, classification logic, fallback rules,
and integration with the feature extraction system.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import torch
from pathlib import Path

from src.pdf_extractor.core.classifier import HeadingClassifier, MobileBERTAdapter, FallbackRuleBasedClassifier
from src.pdf_extractor.models.models import FeatureVector, ProcessedBlock, TextBlock, ClassificationResult


class TestHeadingClassifier(unittest.TestCase):
    """Test cases for the main HeadingClassifier class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.classifier = HeadingClassifier()
        self.sample_features = FeatureVector(
            font_size_ratio=1.5,
            is_bold=True,
            is_italic=False,
            position_x=0.1,
            position_y=0.2,
            text_length=25,
            capitalization_score=0.3,
            whitespace_ratio=0.1
        )
    
    def test_init(self):
        """Test classifier initialization."""
        self.assertIsInstance(self.classifier.model_adapter, MobileBERTAdapter)
        self.assertIsInstance(self.classifier.fallback_classifier, FallbackRuleBasedClassifier)
        self.assertFalse(self.classifier.model_loaded)
    
    def test_classify_block_with_fallback(self):
        """Test classification using fallback when model not loaded."""
        text = "Chapter 1: Introduction"
        result = self.classifier.classify_block(self.sample_features, text)
        
        self.assertIsInstance(result, ClassificationResult)
        self.assertEqual(result.block.text, text)
        self.assertIn(result.predicted_class, ['title', 'h1', 'h2', 'h3', 'text'])
        self.assertGreater(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
    
    @patch('classifier.MobileBERTAdapter')
    def test_classify_block_with_model(self, mock_adapter_class):
        """Test classification with loaded model."""
        # Setup mock
        mock_adapter = Mock()
        mock_adapter.predict.return_value = ('h1', 0.8)
        mock_adapter_class.return_value = mock_adapter
        
        classifier = HeadingClassifier()
        classifier.model_adapter = mock_adapter
        classifier.model_loaded = True
        
        text = "Introduction"
        result = classifier.classify_block(self.sample_features, text)
        
        self.assertEqual(result.predicted_class, 'h1')
        self.assertEqual(result.confidence, 0.8)
        mock_adapter.predict.assert_called_once_with(text, self.sample_features)
    
    @patch('classifier.MobileBERTAdapter')
    def test_classify_block_low_confidence_fallback(self, mock_adapter_class):
        """Test fallback when model confidence is too low."""
        # Setup mock with low confidence
        mock_adapter = Mock()
        mock_adapter.predict.return_value = ('h1', 0.3)  # Below threshold
        mock_adapter_class.return_value = mock_adapter
        
        classifier = HeadingClassifier()
        classifier.model_adapter = mock_adapter
        classifier.model_loaded = True
        
        text = "1.1 Introduction"
        result = classifier.classify_block(self.sample_features, text)
        
        # Should use fallback classification
        self.assertIsInstance(result, ClassificationResult)
        # Fallback should classify numbered heading as h2
        self.assertEqual(result.predicted_class, 'h2')
    
    def test_load_model_success(self):
        """Test successful model loading."""
        with patch.object(self.classifier.model_adapter, 'load_model') as mock_load:
            model_path = "test/path"
            self.classifier.load_model(model_path)
            
            mock_load.assert_called_once_with(model_path)
            self.assertTrue(self.classifier.model_loaded)
    
    def test_load_model_failure(self):
        """Test model loading failure."""
        with patch.object(self.classifier.model_adapter, 'load_model', 
                         side_effect=Exception("Model not found")):
            model_path = "invalid/path"
            self.classifier.load_model(model_path)
            
            self.assertFalse(self.classifier.model_loaded)
    
    def test_predict_heading_level(self):
        """Test heading level prediction."""
        block = ProcessedBlock(
            text="Chapter 1",
            page_number=1,
            features=self.sample_features,
            original_block=None
        )
        
        result = self.classifier.predict_heading_level(block)
        self.assertIn(result, ['title', 'h1', 'h2', 'h3', 'text'])


class TestMobileBERTAdapter(unittest.TestCase):
    """Test cases for the MobileBERTAdapter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.adapter = MobileBERTAdapter()
        self.sample_features = FeatureVector(
            font_size_ratio=1.5,
            is_bold=True,
            is_italic=False,
            position_x=0.1,
            position_y=0.2,
            text_length=25,
            capitalization_score=0.3,
            whitespace_ratio=0.1
        )
    
    def test_init(self):
        """Test adapter initialization."""
        self.assertIsNone(self.adapter.model)
        self.assertIsNone(self.adapter.tokenizer)
        self.assertEqual(self.adapter.device, torch.device('cpu'))
        self.assertEqual(self.adapter.class_labels, ['text', 'title', 'h1', 'h2', 'h3'])
    
    @patch('classifier.AutoTokenizer')
    @patch('classifier.AutoModelForSequenceClassification')
    def test_load_model_success(self, mock_model_class, mock_tokenizer_class):
        """Test successful model loading."""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Create temporary directory structure
        with patch('pathlib.Path.exists', return_value=True):
            model_path = "test/model/path"
            self.adapter.load_model(model_path)
            
            # Verify calls
            mock_tokenizer_class.from_pretrained.assert_called_once()
            mock_model_class.from_pretrained.assert_called_once()
            mock_model.to.assert_called_once_with(self.adapter.device)
            mock_model.eval.assert_called_once()
            
            self.assertEqual(self.adapter.tokenizer, mock_tokenizer)
            self.assertEqual(self.adapter.model, mock_model)
    
    def test_load_model_path_not_exists(self):
        """Test model loading with non-existent path."""
        with self.assertRaises(FileNotFoundError):
            self.adapter.load_model("nonexistent/path")
    
    @patch('classifier.AutoTokenizer')
    @patch('classifier.AutoModelForSequenceClassification')
    def test_load_model_failure(self, mock_model_class, mock_tokenizer_class):
        """Test model loading failure."""
        mock_tokenizer_class.from_pretrained.side_effect = Exception("Loading failed")
        
        with patch('pathlib.Path.exists', return_value=True):
            with self.assertRaises(Exception):
                self.adapter.load_model("test/path")
    
    def test_predict_without_model(self):
        """Test prediction without loaded model."""
        with self.assertRaises(RuntimeError):
            self.adapter.predict("test text", self.sample_features)
    
    @patch('classifier.torch.no_grad')
    def test_predict_with_model(self, mock_no_grad):
        """Test prediction with loaded model."""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_model = Mock()
        
        # Mock tokenizer output
        mock_inputs = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        mock_tokenizer.return_value = mock_inputs
        
        # Mock model output
        mock_outputs = Mock()
        mock_logits = torch.tensor([[0.1, 0.8, 0.05, 0.03, 0.02]])  # h1 has highest score
        mock_outputs.logits = mock_logits
        mock_model.return_value = mock_outputs
        
        # Set up adapter
        self.adapter.tokenizer = mock_tokenizer
        self.adapter.model = mock_model
        
        # Test prediction
        text = "Introduction"
        predicted_class, confidence = self.adapter.predict(text, self.sample_features)
        
        # Verify results
        self.assertEqual(predicted_class, 'title')  # Index 1 in class_labels
        self.assertGreater(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        
        # Verify calls
        mock_tokenizer.assert_called_once()
        mock_model.assert_called_once()
    
    def test_enhance_text_with_features(self):
        """Test text enhancement with features."""
        text = "Introduction"
        
        # Test with various feature combinations
        features_large_bold = FeatureVector(
            font_size_ratio=2.0,  # Large font
            is_bold=True,
            is_italic=False,
            position_x=0.1,
            position_y=0.1,  # Top position
            text_length=20,  # Short text
            capitalization_score=0.9,  # All caps
            whitespace_ratio=0.1
        )
        
        enhanced = self.adapter._enhance_text_with_features(text, features_large_bold)
        
        # Should contain feature hints
        self.assertIn("[LARGE_FONT]", enhanced)
        self.assertIn("[BOLD]", enhanced)
        self.assertIn("[TOP_POSITION]", enhanced)
        self.assertIn("[SHORT_TEXT]", enhanced)
        self.assertIn("[ALL_CAPS]", enhanced)
        self.assertIn(text, enhanced)
    
    def test_enhance_text_no_features(self):
        """Test text enhancement with minimal features."""
        text = "Regular text"
        features_minimal = FeatureVector(
            font_size_ratio=1.0,
            is_bold=False,
            is_italic=False,
            position_x=0.5,
            position_y=0.5,
            text_length=100,
            capitalization_score=0.1,
            whitespace_ratio=0.2
        )
        
        enhanced = self.adapter._enhance_text_with_features(text, features_minimal)
        self.assertEqual(enhanced, text)  # No hints added


class TestFallbackRuleBasedClassifier(unittest.TestCase):
    """Test cases for the FallbackRuleBasedClassifier class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.classifier = FallbackRuleBasedClassifier()
    
    def test_init(self):
        """Test classifier initialization."""
        self.assertIsInstance(self.classifier.title_patterns, list)
        self.assertIsInstance(self.classifier.heading_patterns, list)
        self.assertGreater(len(self.classifier.title_patterns), 0)
        self.assertGreater(len(self.classifier.heading_patterns), 0)
    
    def test_classify_title_by_pattern(self):
        """Test title classification by pattern matching."""
        features = FeatureVector(
            font_size_ratio=1.0,
            is_bold=False,
            is_italic=False,
            position_x=0.1,
            position_y=0.1,
            text_length=50,
            capitalization_score=1.0,  # All caps
            whitespace_ratio=0.1
        )
        
        text = "DOCUMENT TITLE HERE"
        predicted_class, confidence = self.classifier.classify(text, features)
        
        self.assertEqual(predicted_class, 'title')
        self.assertGreater(confidence, 0.0)
    
    def test_classify_heading_by_font_size(self):
        """Test heading classification by font size."""
        # Test H1 classification with large font but not title-like
        features_h1 = FeatureVector(
            font_size_ratio=1.8,  # Large but not title-large
            is_bold=True,
            is_italic=False,
            position_x=0.1,
            position_y=0.4,  # Middle of page
            text_length=30,
            capitalization_score=0.3,  # Not all caps
            whitespace_ratio=0.1
        )
        
        text = "Main Heading"
        predicted_class, confidence = self.classifier.classify(text, features_h1)
        
        self.assertEqual(predicted_class, 'h2')  # Font size 1.8 falls in h2 range (1.6-2.0)
        self.assertGreater(confidence, 0.0)
    
    def test_classify_numbered_heading(self):
        """Test numbered heading classification."""
        features = FeatureVector(
            font_size_ratio=1.2,
            is_bold=False,
            is_italic=False,
            position_x=0.1,
            position_y=0.3,
            text_length=40,
            capitalization_score=0.2,
            whitespace_ratio=0.1
        )
        
        # Test different numbering patterns
        test_cases = [
            ("1. Introduction", 'h1'),
            ("1.1 Overview", 'h2'),
            ("1.1.1 Details", 'h3'),
            ("Chapter 1", 'h2'),
        ]
        
        for text, expected_level in test_cases:
            with self.subTest(text=text):
                predicted_class, confidence = self.classifier.classify(text, features)
                self.assertEqual(predicted_class, expected_level)
                self.assertGreater(confidence, 0.0)
    
    def test_classify_regular_text(self):
        """Test regular text classification."""
        features = FeatureVector(
            font_size_ratio=1.0,  # Normal size
            is_bold=False,
            is_italic=False,
            position_x=0.1,
            position_y=0.5,
            text_length=200,  # Long text
            capitalization_score=0.1,
            whitespace_ratio=0.15
        )
        
        text = "This is a regular paragraph with normal formatting and length."
        predicted_class, confidence = self.classifier.classify(text, features)
        
        self.assertEqual(predicted_class, 'text')
        self.assertGreater(confidence, 0.0)
    
    def test_is_title_patterns(self):
        """Test title pattern matching."""
        features = FeatureVector(
            font_size_ratio=1.0,
            is_bold=False,
            is_italic=False,
            position_x=0.1,
            position_y=0.1,
            text_length=50,
            capitalization_score=0.5,
            whitespace_ratio=0.1
        )
        
        # Test various title patterns
        title_texts = [
            "ANNUAL REPORT 2023",
            "TITLE: Document Overview",
            "SUBJECT: Important Information"
        ]
        
        for text in title_texts:
            with self.subTest(text=text):
                result = self.classifier._is_title(text, features)
                self.assertTrue(result)
    
    def test_is_numbered_heading(self):
        """Test numbered heading detection."""
        numbered_headings = [
            "1.1 Introduction",
            "2.3.4 Details",
            "Chapter 5",
            "Section 2"
        ]
        
        for text in numbered_headings:
            with self.subTest(text=text):
                result = self.classifier._is_numbered_heading(text)
                self.assertTrue(result)
        
        # Test non-numbered text
        non_numbered = [
            "Regular text",
            "Introduction without number",
            "Some content here"
        ]
        
        for text in non_numbered:
            with self.subTest(text=text):
                result = self.classifier._is_numbered_heading(text)
                self.assertFalse(result)
    
    def test_determine_heading_level_from_numbering(self):
        """Test heading level determination from numbering."""
        test_cases = [
            ("1 Introduction", 'h1'),
            ("1.1 Overview", 'h2'),
            ("1.1.1 Details", 'h3'),
            ("2.3.4.5 Deep nesting", 'h3'),  # Deep nesting defaults to h3
        ]
        
        for text, expected_level in test_cases:
            with self.subTest(text=text):
                result = self.classifier._determine_heading_level_from_numbering(text)
                self.assertEqual(result, expected_level)


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    unittest.main()