#!/usr/bin/env python3
"""
Simple test to verify multilingual support improvements.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.pdf_extractor.core.preprocessor import TextNormalizer
from src.pdf_extractor.core.classifier import FallbackRuleBasedClassifier
from src.pdf_extractor.core.pdf_parser import PDFParser
from src.pdf_extractor.models.models import FeatureVector


def test_enhanced_unicode_normalization():
    """Test enhanced Unicode normalization."""
    print("🌍 Testing Enhanced Unicode Normalization:")
    print("-" * 50)
    
    normalizer = TextNormalizer({})
    
    test_cases = [
        # Enhanced space character handling
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
        ("Text\u201cwith\u201dsmart\u201equotes", 'Text"with"smart"quotes'),
        ("Text\u2018with\u2019single\u201aquotes", "Text'with'single'quotes"),
    ]
    
    for input_text, expected in test_cases:
        result = normalizer._normalize_unicode(input_text)
        status = "✅" if result == expected else "❌"
        print(f"{status} '{input_text}' -> '{result}'")
        if result != expected:
            print(f"   Expected: '{expected}'")


def test_multilingual_heading_patterns():
    """Test multilingual heading pattern recognition."""
    print("\n🔤 Testing Multilingual Heading Patterns:")
    print("-" * 50)
    
    classifier = FallbackRuleBasedClassifier()
    
    test_cases = [
        # English patterns
        ("1. Introduction", True),
        ("Chapter 1", True),
        ("Section 2.1", True),
        ("Appendix A", True),
        
        # Spanish patterns
        ("Capítulo 1", True),
        ("1. Introducción", True),
        ("Apéndice B", True),
        
        # French patterns
        ("Chapitre 1", True),
        ("1. Introduction", True),
        ("Annexe C", True),
        
        # German patterns
        ("Kapitel 1", True),
        ("1. Einleitung", True),
        ("Anhang D", True),
        
        # Chinese patterns
        ("第一章", True),
        ("第1章", True),
        ("一、", True),
        
        # Arabic patterns
        ("الفصل 1", True),
        ("القسم 2", True),
        
        # Korean patterns
        ("제 1 장", True),
        ("제1절", True),
        
        # Non-heading patterns
        ("Regular text", False),
        ("This is a sentence.", False),
    ]
    
    for text, should_be_heading in test_cases:
        is_heading = classifier._is_numbered_heading(text)
        status = "✅" if is_heading == should_be_heading else "❌"
        print(f"{status} '{text}' -> {'Heading' if is_heading else 'Not heading'}")


def test_multilingual_title_patterns():
    """Test multilingual title pattern recognition."""
    print("\n📋 Testing Multilingual Title Patterns:")
    print("-" * 50)
    
    classifier = FallbackRuleBasedClassifier()
    
    test_cases = [
        # English title patterns
        ("TITLE: Research Paper", True),
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
        
        # German title patterns
        ("TITEL: Forschungsdokument", True),
        ("Zusammenfassung: Dieses Papier diskutiert...", True),
        
        # Chinese title patterns
        ("标题: 研究论文", True),
        ("摘要: 本文讨论...", True),
        
        # Arabic title patterns
        ("العنوان: ورقة بحثية", True),
        ("الملخص: تناقش هذه الورقة...", True),
        
        # Non-title patterns
        ("Regular paragraph text", False),
        ("This is not a title", False),
    ]
    
    for text, should_be_title in test_cases:
        # Test pattern matching
        is_title_pattern = False
        for i, pattern in enumerate(classifier.title_patterns):
            import re
            if i == 0:  # First pattern (all caps) is case-sensitive
                if re.match(pattern, text):
                    is_title_pattern = True
                    break
            else:  # Other patterns are case-insensitive
                if re.match(pattern, text, re.IGNORECASE):
                    is_title_pattern = True
                    break
        
        status = "✅" if is_title_pattern == should_be_title else "❌"
        print(f"{status} '{text}' -> {'Title pattern' if is_title_pattern else 'Not title pattern'}")


def test_pdf_parser_enhancements():
    """Test PDF parser multilingual enhancements."""
    print("\n📄 Testing PDF Parser Multilingual Enhancements:")
    print("-" * 50)
    
    parser = PDFParser()
    
    test_cases = [
        # Basic multilingual strings
        ("English text", "English text"),
        ("Français", "Français"),
        ("中文", "中文"),
        ("العربية", "العربية"),
        ("русский", "русский"),
        
        # Enhanced Unicode handling
        ("Text\ufeffwith BOM", "Textwith BOM"),
        ("Text\u00a0with NBSP", "Text with NBSP"),
        ("Text\u2000\u2001various spaces", "Text various spaces"),
        ("Text\u200ewith\u200fmarks", "Textwithmarks"),
        ("Text\u200bwith\u200czero\u200dwidth", "Textwithzerowidth"),
        ("Text\u201cwith\u201dsmart\u201equotes", 'Text"with"smart"quotes'),
    ]
    
    for input_text, expected in test_cases:
        result = parser._normalize_text_encoding(input_text)
        status = "✅" if result == expected else "❌"
        print(f"{status} '{input_text}' -> '{result}'")
        if result != expected:
            print(f"   Expected: '{expected}'")


def test_classification_accuracy():
    """Test improved classification accuracy."""
    print("\n🎯 Testing Classification Accuracy Improvements:")
    print("-" * 50)
    
    classifier = FallbackRuleBasedClassifier()
    
    # Create mock features for testing
    def create_features(font_size_ratio=1.0, is_bold=False, position_y=0.5, text_length=50, cap_score=0.2):
        return FeatureVector(
            font_size_ratio=font_size_ratio,
            is_bold=is_bold,
            is_italic=False,
            position_x=0.0,
            position_y=position_y,
            text_length=text_length,
            capitalization_score=cap_score,
            whitespace_ratio=0.1
        )
    
    test_cases = [
        # Clear headings with multilingual content
        ("1. Introduction", "heading", create_features(1.5, True, 0.1, 20)),
        ("1.1 Overview", "heading", create_features(1.3, True, 0.2, 15)),
        ("Capítulo 1: Introducción", "heading", create_features(1.5, True, 0.1, 25)),
        ("第一章 概述", "heading", create_features(1.4, True, 0.1, 10)),
        
        # Title patterns
        ("TITLE: Research Paper", "title", create_features(2.0, True, 0.05, 30, 0.8)),
        ("标题: 研究论文", "title", create_features(1.8, True, 0.05, 15, 0.3)),
        
        # Regular text
        ("This is regular paragraph text.", "text", create_features(1.0, False, 0.5, 50, 0.1)),
        ("这是常规段落文本。", "text", create_features(1.0, False, 0.5, 20, 0.0)),
    ]
    
    for text, expected_type, features in test_cases:
        predicted_class, confidence = classifier.classify(text, features)
        
        # Check if classification is reasonable
        if expected_type == "heading":
            is_correct = predicted_class in ["h1", "h2", "h3"]
        elif expected_type == "title":
            is_correct = predicted_class == "title"
        else:  # text
            is_correct = predicted_class == "text"
        
        status = "✅" if is_correct else "❌"
        print(f"{status} '{text}' -> {predicted_class} (confidence: {confidence:.2f})")


if __name__ == "__main__":
    test_enhanced_unicode_normalization()
    test_multilingual_heading_patterns()
    test_multilingual_title_patterns()
    test_pdf_parser_enhancements()
    test_classification_accuracy()
    
    print("\n🎉 Multilingual support testing complete!")