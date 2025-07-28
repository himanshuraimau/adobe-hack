#!/usr/bin/env python3
"""
Test script to verify multilingual text encoding handling.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from pdf_parser import PDFParser

def test_encoding_normalization():
    """Test the text encoding normalization function."""
    parser = PDFParser()
    
    test_cases = [
        ("English text", "English text"),
        ("Français", "Français"),
        ("Español", "Español"),
        ("Deutsch", "Deutsch"),
        ("中文", "中文"),
        ("العربية", "العربية"),
        ("русский", "русский"),
        ("  whitespace  ", "whitespace"),
        ("", ""),
        (None, ""),
    ]
    
    print("🌍 Testing multilingual text encoding normalization:")
    print("-" * 60)
    
    for i, (input_text, expected_contains) in enumerate(test_cases, 1):
        try:
            result = parser._normalize_text_encoding(input_text)
            status = "✅" if expected_contains in result or (not expected_contains and not result) else "❌"
            print(f"{status} Test {i}: '{input_text}' -> '{result}'")
        except Exception as e:
            print(f"❌ Test {i}: Error with '{input_text}': {e}")
    
    print("\n🔧 Testing font flag detection:")
    print("-" * 40)
    
    flag_tests = [
        (0, "Normal", False, False),
        (16, "Bold", True, False),
        (2, "Italic", False, True),
        (18, "Bold+Italic", True, True),
    ]
    
    for flags, description, expected_bold, expected_italic in flag_tests:
        is_bold = parser._is_bold_font(flags)
        is_italic = parser._is_italic_font(flags)
        
        bold_status = "✅" if is_bold == expected_bold else "❌"
        italic_status = "✅" if is_italic == expected_italic else "❌"
        
        print(f"{bold_status} {italic_status} {description} (flags={flags}): "
              f"bold={is_bold}, italic={is_italic}")

if __name__ == "__main__":
    test_encoding_normalization()