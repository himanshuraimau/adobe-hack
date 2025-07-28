#!/usr/bin/env python3
"""Integration test for processing all collections."""

import logging
import sys
import os

# Add src to path for imports
sys.path.insert(0, 'src')

from pdf_analysis.collection_processor import CollectionProcessor

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Test processing all collections
    processor = CollectionProcessor()
    
    print("Testing process_all_collections method...")
    try:
        # Process all collections
        results = processor.process_all_collections()
        
        print(f"\nProcessed {len(results)} collections:")
        for result in results:
            print(f"  - {result.collection_path}: {'SUCCESS' if result.success else 'FAILED'}")
            if result.success:
                print(f"    Documents: {result.documents_processed}, Sections: {result.sections_processed}")
            else:
                print(f"    Error: {result.error_message}")
        
        successful = sum(1 for r in results if r.success)
        print(f"\nSummary: {successful}/{len(results)} collections processed successfully")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()