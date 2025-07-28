"""
Integration test for preprocessing with PDF parser.

This test verifies that the preprocessing pipeline works correctly
with actual PDF parsing output.
"""

import sys
import os
import logging

# Add the current directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pdf_extractor.core.pdf_parser import PDFParser
from src.pdf_extractor.core.preprocessor import TextPreprocessor
from pathlib import Path


def test_preprocessing_integration():
    """Test preprocessing with actual PDF data."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Find a test PDF file
    input_dir = Path(__file__).parent / "input"
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in input directory for testing")
        return
    
    pdf_file = pdf_files[0]
    print(f"Testing with PDF: {pdf_file.name}")
    
    # Parse PDF
    parser = PDFParser()
    try:
        text_blocks = parser.parse_document(str(pdf_file))
        print(f"Extracted {len(text_blocks)} text blocks from PDF")
        
        # Show sample of raw blocks
        print("\nSample raw text blocks:")
        for i, block in enumerate(text_blocks[:5]):
            print(f"Block {i+1}: '{block.text[:50]}...' (font: {block.font_size}, page: {block.page_number})")
        
        # Preprocess blocks
        preprocessor = TextPreprocessor()
        processed_blocks = preprocessor.preprocess_blocks(text_blocks)
        print(f"\nProcessed to {len(processed_blocks)} blocks")
        
        # Show sample of processed blocks
        print("\nSample processed text blocks:")
        for i, block in enumerate(processed_blocks[:5]):
            print(f"Block {i+1}: '{block.text[:50]}...'")
            print(f"  Features: font_ratio={block.features.font_size_ratio:.2f}, "
                  f"bold={block.features.is_bold}, "
                  f"cap_score={block.features.capitalization_score:.2f}")
        
        # Verify that processing preserved important information
        assert len(processed_blocks) > 0, "No blocks after preprocessing"
        
        # Check that features are calculated
        for block in processed_blocks[:3]:
            assert block.features is not None, "Missing features"
            assert block.features.font_size_ratio > 0, "Invalid font size ratio"
            assert 0 <= block.features.capitalization_score <= 1, "Invalid capitalization score"
            assert 0 <= block.features.whitespace_ratio <= 1, "Invalid whitespace ratio"
        
        print("\n✓ Integration test passed successfully!")
        
    except Exception as e:
        print(f"Integration test failed: {e}")
        raise
    finally:
        parser.close()


if __name__ == '__main__':
    test_preprocessing_integration()