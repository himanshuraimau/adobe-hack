#!/usr/bin/env python3
"""
Manual test script to verify PDF parsing functionality with sample PDFs.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.pdf_extractor.core.pdf_parser import PDFParser
from pathlib import Path

def test_sample_pdfs():
    """Test parsing with the provided sample PDFs."""
    input_dir = Path(__file__).parent / "input"
    sample_pdfs = [
        "E0CCG5S239.pdf",
        "TOPJUMP-PARTY-INVITATION-20161003-V01.pdf"
    ]
    
    for pdf_name in sample_pdfs:
        pdf_path = input_dir / pdf_name
        if not pdf_path.exists():
            print(f"❌ Sample PDF {pdf_name} not found")
            continue
        
        print(f"\n📄 Testing {pdf_name}:")
        print("-" * 50)
        
        try:
            with PDFParser() as parser:
                # Parse the document
                blocks = parser.parse_document(str(pdf_path))
                
                # Get metadata
                metadata = parser.get_document_metadata()
                
                print(f"✅ Successfully parsed {len(blocks)} text blocks")
                print(f"📊 Document has {metadata.get('page_count', 'unknown')} pages")
                
                # Show first few blocks
                print("\n🔍 First 5 text blocks:")
                for i, block in enumerate(blocks[:5]):
                    print(f"  {i+1}. Page {block.page_number}: '{block.text[:50]}...' "
                          f"(font: {block.font_name}, size: {block.font_size:.1f})")
                
                # Show font information
                font_sizes = [block.font_size for block in blocks]
                unique_fonts = set(block.font_name for block in blocks)
                
                print(f"\n📝 Font analysis:")
                print(f"  - Font size range: {min(font_sizes):.1f} - {max(font_sizes):.1f}")
                print(f"  - Average font size: {sum(font_sizes)/len(font_sizes):.1f}")
                print(f"  - Unique fonts: {len(unique_fonts)}")
                print(f"  - Font names: {', '.join(list(unique_fonts)[:3])}...")
                
        except Exception as e:
            print(f"❌ Error parsing {pdf_name}: {e}")

if __name__ == "__main__":
    test_sample_pdfs()