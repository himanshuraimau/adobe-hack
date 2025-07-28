#!/usr/bin/env python3
"""Main entry point for the PDF Analysis System."""

import os
import sys
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pdf_analysis.pdf_analyzer import PDFAnalyzer


def main():
    """Main function to process PDF collections."""
    print("PDF Analysis System - Starting...")
    
    # Test the PDF analyzer with a sample file
    analyzer = PDFAnalyzer()
    
    # Find a sample PDF to test
    sample_collections = ["Collection 1", "Collection 2", "Collection 3"]
    
    for collection in sample_collections:
        collection_path = Path(collection)
        if collection_path.exists():
            pdfs_path = collection_path / "PDFs"
            if pdfs_path.exists():
                pdf_files = list(pdfs_path.glob("*.pdf"))
                if pdf_files:
                    print(f"\nTesting with {pdf_files[0]}")
                    try:
                        sections = analyzer.get_section_content(str(pdf_files[0]))
                        print(f"Extracted {len(sections)} sections:")
                        for i, section in enumerate(sections[:3]):  # Show first 3
                            print(f"  {i+1}. {section.section_title} (Page {section.page_number})")
                            print(f"     Content preview: {section.content[:100]}...")
                        break
                    except Exception as e:
                        print(f"Error processing {pdf_files[0]}: {e}")
                        continue
    
    print("\nPDF Analysis System - Task 1 Complete!")


if __name__ == "__main__":
    main()