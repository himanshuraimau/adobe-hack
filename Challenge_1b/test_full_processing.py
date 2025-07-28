#!/usr/bin/env python3
"""Test script for full collection processing."""

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
    
    # Test full collection processing
    processor = CollectionProcessor()
    
    print("Testing full collection processing...")
    try:
        # Process just the first collection for testing
        collections = processor.discover_collections()
        if not collections:
            print("No collections found")
            return
        
        # Process first collection
        result = processor.process_collection(collections[0])
        
        print(f"\nProcessing result for {result.collection_path}:")
        print(f"  - Success: {result.success}")
        if result.success:
            print(f"  - Documents processed: {result.documents_processed}")
            print(f"  - Sections processed: {result.sections_processed}")
        else:
            print(f"  - Error: {result.error_message}")
        
        print("\nFull collection processing test completed!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()