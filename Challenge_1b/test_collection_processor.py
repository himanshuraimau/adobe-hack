#!/usr/bin/env python3
"""Test script for CollectionProcessor implementation."""

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
    
    # Test collection discovery
    processor = CollectionProcessor()
    
    print("Testing collection discovery...")
    try:
        collections = processor.discover_collections()
        print(f"Found collections: {collections}")
        
        if not collections:
            print("No collections found - this is expected if no Collection directories exist")
            return
        
        # Test loading input config from first collection
        print(f"\nTesting input config loading from {collections[0]}...")
        config = processor._load_input_config(collections[0])
        print(f"Config loaded successfully:")
        print(f"  - Challenge: {config.challenge_info.get('test_case_name', 'Unknown')}")
        print(f"  - Persona: {config.persona.get('role', 'Unknown')}")
        print(f"  - Task: {config.job_to_be_done.get('task', 'Unknown')}")
        print(f"  - Documents: {len(config.documents)}")
        
        # Test PDF file discovery
        print(f"\nTesting PDF file discovery...")
        document_filenames = [doc['filename'] for doc in config.documents]
        pdf_paths = processor._find_pdf_files(collections[0], document_filenames)
        print(f"Found PDF files: {len(pdf_paths)}")
        for pdf_path in pdf_paths[:3]:  # Show first 3
            print(f"  - {pdf_path}")
        if len(pdf_paths) > 3:
            print(f"  ... and {len(pdf_paths) - 3} more")
        
        print("\nCollectionProcessor implementation test completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()