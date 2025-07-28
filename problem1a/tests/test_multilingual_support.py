#!/usr/bin/env python3
"""
Comprehensive test suite for multilingual support and accuracy improvements.

This test suite verifies that the PDF structure extractor can handle:
- Multiple languages and scripts
- Various Unicode characters and encodings
- Improved classification accuracy
- Enhanced text normalization
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch
sys.path.insert(0, os.path.dirname(__file__))

from preprocessor import TextPreprocessor, TextNormalizer
from feature_extractor import FeatureExtractor, ContentAnalyzer
from classifier import HeadingClassifier, FallbackRuleBasedClassifier
from pdf_parser import PDFParser
from models import TextBlock, ProcessedBlock, FeatureVector


class TestMultilingualSupport(unittest.TestCase):
    """Test multilingual text processing capabilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = TextPreprocessor()
        self.normalizer = TextNormalizer({})
        self.content_analyzer = ContentAnalyzer({})
        self.classifier = FallbackRuleBasedClassifier()
        self.pdf_parser = PDFParser()
    
    def test_unicode_normalization_comprehensive(self):
        """Test comprehensive Unicode normalization."""
        test_cases = [
            # Basic multilingual text
            ("English text", "English text"),
            ("Français avec accents", "Français avec accents"),
            ("Español con ñ", "Español con ñ"),
            ("Deutsch mit Umlauten: äöü", "Deutsch mit Umlauten: äöü"),
            ("中文测试", "中文测试"),
            ("العربية النص", "العربية النص"),
            ("русский текст", "русский текст"),
            ("日本語テスト", "日本語テスト"),
            ("한국어 테스트", "한국어 테스트"),
            
            # Unicode space characters
            ("Text\u00a0with\u00a0NBSP", "Text with NBSP"),
            ("Text\u2000with\u2001various\u2002spaces", "Text with various spaces"),
            ("Text\u3000with\u3000ideographic\u3000space", "Text with ideographic space"),
            
            # Directional marks (should be removed)
            ("Text\u200ewith\u200fmarks", "Textwithmarks"),
            ("RTL\u202aembedding\u202ctest", "RTLembeddingtest"),
            
            # Zero-width characters (should be removed)
            ("Text\u200bwith\u200czero\u200dwidth", "Textwithzerowidth"),
            ("Soft\u00adhyphen\u00adtest", "Softhyphentest"),
            
            # Quotation marks normalization
            ('"Smart quotes"', '"Smart quotes"'),
            ("'Single quotes'", "'Single quotes'"),
            ('„German quotes"', '"German quotes"'),
            
            # Line separators
            ("Line\u2028separator\u2029paragraph", "Line\nseparator\n\nparagraph"),
            
            # Empty and None cases
            ("", ""),
            ("   ", ""),
        ]
        
        for input_text, expected in test_cases:
            with self.subTest(input_text=input_text):
                result = self.normalizer._normalize_unicode(input_text)
                self.assertEqual(result, expected, 
                    f"Failed for input: '{input_text}' -> got '{result}', expected '{expected}'")
    
    def test_multilingual_capitalization_score(self):
        """Test capitalization score calculation for different scripts."""
        test_cases = [
            # Latin scripts
            ("ENGLISH ALL CAPS", 1.0),
            ("English Mixed Case", 0.2),  # 2 out of 10 letters are uppercase
            ("english lowercase", 0.0),
            
            # Scripts without case distinction (should return 0.0)
            ("中文没有大小写", 0.0),
            ("العربية لا تحتوي", 0.0),
            ("日本語にはケース", 0.0),
            ("한국어는 대소문자", 0.0),
            
            # Mixed scripts
            ("English 中文 Mixed", 0.125),  # 1 out of 8 case-sensitive letters
            ("FRANÇAIS AVEC 中文", 0.8),  # 8 out of 10 case-sensitive letters
            
            # Numbers and punctuation (should be ignored)
            ("123 !@# $%^", 0.0),
            ("Test123!@#", 0.25),  # 1 out of 4 letters
        ]
        
        for text, expected in test_cases:
            with self.subTest(text=text):
                result = self.content_analyzer._calculate_capitalization_score(text)
                self.assertAlmostEqual(result, expected, places=2,
                    msg=f"Failed for '{text}': got {result}, expected {expected}")
    
    def test_multilingual_heading_patterns(self):
        """Test heading pattern recognition for multiple languages."""
        test_cases = [
            # English patterns
            ("1. Introduction", True, "h1"),
            ("1.1 Overview", True, "h2"),
            ("Chapter 1", True, "h1"),
            ("Section 2.1", True, "h2"),
            ("Appendix A", True, "h2"),
            
            # Spanish patterns
            ("Capítulo 1", True, "h1"),
            ("1. Introducción", True, "h1"),
            ("Apéndice B", True, "h2"),
            
            # French patterns
            ("Chapitre 1", True, "h1"),
            ("1. Introduction", True, "h1"),
            ("Annexe C", True, "h2"),
            
            # German patterns
            ("Kapitel 1", True, "h1"),
            ("1. Einleitung", True, "h1"),
            ("Anhang D", True, "h2"),
            
            # Italian patterns
            ("Capitolo 1", True, "h1"),
            ("1. Introduzione", True, "h1"),
            ("Appendice E", True, "h2"),
            
            # Russian patterns
            ("Глава 1", True, "h1"),
            ("1. Введение", True, "h1"),
            ("Приложение Ф", True, "h2"),
            
            # Chinese patterns
            ("第一章", True, "h1"),
            ("第1章", True, "h1"),
            ("一、", True, "h1"),
            
            # Arabic patterns
            ("الفصل 1", True, "h1"),
            ("القسم 2", True, "h2"),
            
            # Japanese patterns
            ("第1章", True, "h1"),
            ("一、", True, "h1"),
            
            # Korean patterns
            ("제 1 장", True, "h1"),
            ("제1절", True, "h2"),
            
            # Roman numerals
            ("I. Introduction", True, "h1"),
            ("II. Methodology", True, "h1"),
            ("III. Results", True, "h1"),
            
            # Letter headings
            ("A. First Section", True, "h2"),
            ("B. Second Section", True, "h2"),
            
            # Parenthesized patterns
            ("(1) First item", True, "h2"),
            ("(a) Sub item", True, "h2"),
            ("a) Another sub item", True, "h2"),
            
            # Non-heading patterns
            ("Regular text", False, "text"),
            ("This is a sentence.", False, "text"),
            ("Not a heading pattern", False, "text"),
        ]
        
        # Create a mock feature vector for testing
        mock_features = FeatureVector(
            font_size_ratio=1.0,
            is_bold=False,
            is_italic=False,
            position_x=0.0,
            position_y=0.0,
            text_length=0,
            capitalization_score=0.0,
            whitespace_ratio=0.0
        )
        
        for text, should_be_heading, expected_level in test_cases:
            with self.subTest(text=text):
                is_numbered = self.classifier._is_numbered_heading(text)
                
                if should_be_heading and is_numbered:
                    level = self.classifier._determine_heading_level_from_numbering(text)
                    # For numbered headings, we expect the determined level
                    self.assertTrue(level in ['h1', 'h2', 'h3'], 
                        f"'{text}' should produce a valid heading level, got '{level}'")
                elif should_be_heading:
                    # For non-numbered headings, test full classification
                    predicted_class, confidence = self.classifier.classify(text, mock_features)
                    self.assertNotEqual(predicted_class, "text", 
                        f"'{text}' should be classified as a heading, got '{predicted_class}'")
                else:
                    # Should not be recognized as a numbered heading
                    self.assertFalse(is_numbered, 
                        f"'{text}' should not be recognized as a numbered heading")
    
    def test_multilingual_title_patterns(self):
        """Test title pattern recognition for multiple languages."""
        test_cases = [
            # English title patterns
            ("TITLE: Research Paper", True),
            ("SUBJECT: Important Topic", True),
            ("TOPIC: Discussion", True),
            ("Abstract: This paper discusses...", True),
            ("Introduction: This section covers...", True),
            ("Conclusion: In summary...", True),
            
            # Spanish title patterns
            ("TÍTULO: Documento de Investigación", True),
            ("Resumen: Este artículo discute...", True),
            ("Introducción: Esta sección cubre...", True),
            ("Conclusión: En resumen...", True),
            
            # French title patterns
            ("TITRE: Document de Recherche", True),
            ("Résumé: Cet article discute...", True),
            ("Introduction: Cette section couvre...", True),
            ("Conclusion: En résumé...", True),
            
            # German title patterns
            ("TITEL: Forschungsdokument", True),
            ("Zusammenfassung: Dieses Papier diskutiert...", True),
            ("Einleitung: Dieser Abschnitt behandelt...", True),
            ("Schlussfolgerung: Zusammenfassend...", True),
            
            # Italian title patterns
            ("TITOLO: Documento di Ricerca", True),
            ("Riassunto: Questo articolo discute...", True),
            ("Introduzione: Questa sezione copre...", True),
            ("Conclusione: In sintesi...", True),
            
            # Russian title patterns
            ("ЗАГОЛОВОК: Исследовательский документ", True),
            ("Аннотация: Эта статья обсуждает...", True),
            ("Введение: Этот раздел охватывает...", True),
            ("Заключение: В заключение...", True),
            
            # Chinese title patterns
            ("标题: 研究论文", True),
            ("摘要: 本文讨论...", True),
            ("介绍: 本节涵盖...", True),
            ("结论: 总之...", True),
            
            # Arabic title patterns
            ("العنوان: ورقة بحثية", True),
            ("الملخص: تناقش هذه الورقة...", True),
            ("مقدمة: يغطي هذا القسم...", True),
            ("خاتمة: في الختام...", True),
            
            # Non-title patterns
            ("Regular paragraph text", False),
            ("This is not a title", False),
            ("Just some content", False),
        ]
        
        for text, should_be_title in test_cases:
            with self.subTest(text=text):
                # Test pattern matching
                is_title_pattern = False
                for i, pattern in enumerate(self.classifier.title_patterns):
                    if i == 0:  # First pattern (all caps) is case-sensitive
                        if re.match(pattern, text):
                            is_title_pattern = True
                            break
                    else:  # Other patterns are case-insensitive
                        if re.match(pattern, text, re.IGNORECASE):
                            is_title_pattern = True
                            break
                
                if should_be_title:
                    self.assertTrue(is_title_pattern, 
                        f"'{text}' should match a title pattern")
                else:
                    self.assertFalse(is_title_pattern, 
                        f"'{text}' should not match a title pattern")
    
    def test_pdf_parser_multilingual_encoding(self):
        """Test PDF parser's multilingual text encoding normalization."""
        test_cases = [
            # Basic multilingual strings
            ("English text", "English text"),
            ("Français", "Français"),
            ("Español", "Español"),
            ("Deutsch", "Deutsch"),
            ("中文", "中文"),
            ("العربية", "العربية"),
            ("русский", "русский"),
            ("日本語", "日本語"),
            ("한국어", "한국어"),
            
            # Unicode issues
            ("Text\ufeffwith BOM", "Textwith BOM"),
            ("Text\u00a0with NBSP", "Text with NBSP"),
            ("Text\u2000\u2001\u2002various spaces", "Text various spaces"),
            
            # Directional marks
            ("Text\u200ewith\u200fmarks", "Textwithmarks"),
            
            # Zero-width characters
            ("Text\u200bwith\u200czero\u200dwidth", "Textwithzerowidth"),
            
            # Quotation marks
            ('"Smart quotes"', '"Smart quotes"'),
            ("'Single quotes'", "'Single quotes'"),
            
            # Empty cases
            ("", ""),
            ("   ", ""),
        ]
        
        for input_text, expected in test_cases:
            with self.subTest(input_text=input_text):
                result = self.pdf_parser._normalize_text_encoding(input_text)
                self.assertEqual(result, expected,
                    f"Failed for input: '{input_text}' -> got '{result}', expected '{expected}'")
    
    def test_enhanced_text_preprocessing(self):
        """Test enhanced text preprocessing with multilingual content."""
        # Create test text blocks with multilingual content
        test_blocks = [
            TextBlock("English Heading", 1, (0, 0, 100, 20), 16.0, "Arial", 16),
            TextBlock("Título en Español", 1, (0, 25, 120, 45), 14.0, "Arial", 16),
            TextBlock("Titre en Français", 1, (0, 50, 110, 70), 14.0, "Arial", 16),
            TextBlock("中文标题", 1, (0, 75, 80, 95), 15.0, "SimSun", 16),
            TextBlock("العنوان العربي", 1, (0, 100, 90, 120), 14.0, "Arial", 16),
            TextBlock("Regular\u00a0text\u2000with\u2001spaces", 1, (0, 125, 200, 145), 12.0, "Arial", 0),
            TextBlock("Text\u200bwith\u200czero\u200dwidth", 1, (0, 150, 180, 170), 12.0, "Arial", 0),
        ]
        
        # Process the blocks
        processed_blocks = self.preprocessor.preprocess_blocks(test_blocks)
        
        # Verify processing
        self.assertGreater(len(processed_blocks), 0, "Should have processed blocks")
        
        # Check that Unicode normalization was applied
        for block in processed_blocks:
            # Should not contain problematic Unicode characters
            self.assertNotIn('\u00a0', block.text, "Should not contain NBSP")
            self.assertNotIn('\u2000', block.text, "Should not contain en quad")
            self.assertNotIn('\u200b', block.text, "Should not contain zero-width space")
            self.assertNotIn('\u200c', block.text, "Should not contain zero-width non-joiner")
            self.assertNotIn('\u200d', block.text, "Should not contain zero-width joiner")
            
            # Should contain normalized text
            self.assertGreater(len(block.text.strip()), 0, "Should have non-empty text")
    
    def test_feature_extraction_multilingual(self):
        """Test feature extraction with multilingual content."""
        # Create a feature extractor
        extractor = FeatureExtractor()
        
        # Create test blocks with different languages
        test_cases = [
            ("English Title", 16.0, True),  # English, large font, bold
            ("Título Español", 14.0, True),  # Spanish, medium font, bold
            ("中文标题", 15.0, True),  # Chinese, medium-large font, bold
            ("العنوان العربي", 14.0, True),  # Arabic, medium font, bold
            ("Regular English text", 12.0, False),  # English, normal font, not bold
            ("Texto regular en español", 12.0, False),  # Spanish, normal font, not bold
        ]
        
        processed_blocks = []
        for text, font_size, is_bold in test_cases:
            font_flags = 16 if is_bold else 0
            text_block = TextBlock(text, 1, (0, 0, 100, 20), font_size, "Arial", font_flags)
            
            # Create a basic processed block
            basic_features = FeatureVector(
                font_size_ratio=font_size / 12.0,
                is_bold=is_bold,
                is_italic=False,
                position_x=0.0,
                position_y=0.0,
                text_length=len(text),
                capitalization_score=0.0,
                whitespace_ratio=0.0
            )
            
            processed_block = ProcessedBlock(
                text=text,
                page_number=1,
                features=basic_features,
                original_block=text_block
            )
            processed_blocks.append(processed_block)
        
        # Initialize document stats
        extractor.initialize_document_stats(processed_blocks)
        
        # Extract features for each block
        for block in processed_blocks:
            features = extractor.extract_features(block)
            
            # Verify feature extraction worked
            self.assertIsInstance(features, FeatureVector)
            self.assertGreaterEqual(features.font_size_ratio, 0.0)
            self.assertGreaterEqual(features.text_length, 0)
            self.assertGreaterEqual(features.capitalization_score, 0.0)
            self.assertLessEqual(features.capitalization_score, 1.0)
    
    def test_classification_accuracy_improvements(self):
        """Test improved classification accuracy with enhanced rules."""
        classifier = HeadingClassifier()
        
        # Test cases with expected classifications
        test_cases = [
            # Clear headings with multilingual content
            ("1. Introduction", "h1", 0.6),
            ("1.1 Overview", "h2", 0.6),
            ("Chapter 1: Getting Started", "h1", 0.6),
            ("Capítulo 1: Introducción", "h1", 0.6),
            ("第一章 概述", "h1", 0.6),
            ("الفصل 1: مقدمة", "h1", 0.6),
            
            # Title patterns
            ("TITLE: Research Paper", "title", 0.6),
            ("TÍTULO: Documento de Investigación", "title", 0.6),
            ("标题: 研究论文", "title", 0.6),
            
            # Regular text
            ("This is regular paragraph text that should not be classified as a heading.", "text", 0.8),
            ("Este es texto regular que no debería ser clasificado como encabezado.", "text", 0.8),
            ("这是常规段落文本，不应被归类为标题。", "text", 0.8),
        ]
        
        for text, expected_class, min_confidence in test_cases:
            with self.subTest(text=text):
                # Create mock features
                features = FeatureVector(
                    font_size_ratio=1.5 if expected_class != "text" else 1.0,
                    is_bold=expected_class != "text",
                    is_italic=False,
                    position_x=0.0,
                    position_y=0.1,
                    text_length=len(text),
                    capitalization_score=0.5 if expected_class == "title" else 0.2,
                    whitespace_ratio=0.1
                )
                
                # Test fallback classifier directly
                predicted_class, confidence = classifier.fallback_classifier.classify(text, features)
                
                # Verify classification
                if expected_class != "text":
                    self.assertNotEqual(predicted_class, "text", 
                        f"'{text}' should not be classified as regular text")
                    self.assertGreaterEqual(confidence, min_confidence,
                        f"Confidence for '{text}' should be at least {min_confidence}")


class TestAccuracyImprovements(unittest.TestCase):
    """Test accuracy improvements in classification."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.classifier = FallbackRuleBasedClassifier()
    
    def test_improved_heading_level_detection(self):
        """Test improved heading level detection accuracy."""
        test_cases = [
            # Single-level numbering should be h1
            ("1. Introduction", "h1"),
            ("2. Methodology", "h1"),
            ("3. Results", "h1"),
            
            # Two-level numbering should be h2
            ("1.1 Overview", "h2"),
            ("2.1 Data Collection", "h2"),
            ("3.1 Analysis", "h2"),
            
            # Three-level numbering should be h3
            ("1.1.1 Details", "h3"),
            ("2.1.1 Specifics", "h3"),
            ("3.1.1 Implementation", "h3"),
            
            # Roman numerals should be h1
            ("I. Introduction", "h1"),
            ("II. Methods", "h1"),
            ("III. Results", "h1"),
            
            # Letter headings should be h2
            ("A. First Section", "h2"),
            ("B. Second Section", "h2"),
            
            # Parenthesized should be h2
            ("(1) First Item", "h2"),
            ("(a) Sub Item", "h2"),
        ]
        
        for text, expected_level in test_cases:
            with self.subTest(text=text):
                if self.classifier._is_numbered_heading(text):
                    level = self.classifier._determine_heading_level_from_numbering(text)
                    self.assertEqual(level, expected_level,
                        f"'{text}' should be classified as '{expected_level}', got '{level}'")
    
    def test_enhanced_pattern_matching(self):
        """Test enhanced pattern matching for better accuracy."""
        # Test that enhanced patterns catch more cases
        enhanced_patterns = [
            # Multilingual chapter patterns
            "Capítulo 1",
            "Chapitre 1", 
            "Kapitel 1",
            "Capitolo 1",
            "Глава 1",
            "章节 1",
            "فصل 1",
            
            # Multilingual section patterns
            "Sección 1.1",
            "Section 1.1",
            "Abschnitt 1.1",
            "Sezione 1.1",
            "Раздел 1.1",
            "القسم 1.1",
            
            # Appendix patterns
            "Apéndice A",
            "Annexe A",
            "Anhang A", 
            "Appendice A",
            "Приложение А",
            "附录 A",
            "ملحق أ",
        ]
        
        for pattern in enhanced_patterns:
            with self.subTest(pattern=pattern):
                is_heading = self.classifier._is_numbered_heading(pattern)
                self.assertTrue(is_heading, f"'{pattern}' should be recognized as a heading")


if __name__ == '__main__':
    # Import re module for pattern testing
    import re
    
    # Run the tests
    unittest.main(verbosity=2)