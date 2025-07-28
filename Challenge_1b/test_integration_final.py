#!/usr/bin/env python3
"""Integration test for the complete PDF Analysis System."""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pdf_analysis.collection_processor import CollectionProcessor


def test_integration():
    """Test the complete integration of all components."""
    print("Running integration test...")
    
    # Test with existing Collection 1
    processor = CollectionProcessor(batch_size=2)  # Small batch size for testing
    
    try:
        # Test collection discovery
        collections = processor.discover_collections(".")
        print(f"✓ Discovered {len(collections)} collections")
        
        # Test processing a single collection
        if collections:
            result = processor.process_collection(collections[0])
            if result.success:
                print(f"✓ Successfully processed {collections[0]}")
                print(f"  - Documents: {result.documents_processed}")
                print(f"  - Sections: {result.sections_processed}")
                
                # Verify output file exists
                output_file = os.path.join(collections[0], "challenge1b_output.json")
                if os.path.exists(output_file):
                    print("✓ Output file created successfully")
                    
                    # Verify JSON structure
                    with open(output_file, 'r') as f:
                        data = json.load(f)
                    
                    required_keys = ["metadata", "extracted_sections", "subsection_analysis"]
                    if all(key in data for key in required_keys):
                        print("✓ Output JSON has correct structure")
                    else:
                        print("✗ Output JSON missing required keys")
                        return False
                else:
                    print("✗ Output file not created")
                    return False
            else:
                print(f"✗ Failed to process collection: {result.error_message}")
                return False
        
        print("✓ Integration test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False


def test_error_handling():
    """Test error handling capabilities."""
    print("\nTesting error handling...")
    
    processor = CollectionProcessor()
    
    # Test with non-existent collection
    try:
        result = processor.process_collection("/nonexistent/collection")
        if not result.success:
            print("✓ Correctly handled non-existent collection")
        else:
            print("✗ Should have failed for non-existent collection")
            return False
    except Exception as e:
        print(f"✗ Unexpected exception: {e}")
        return False
    
    print("✓ Error handling test passed!")
    return True


def test_batch_processing():
    """Test batch processing functionality."""
    print("\nTesting batch processing...")
    
    # Test with different batch sizes
    for batch_size in [1, 3, 10]:
        processor = CollectionProcessor(batch_size=batch_size)
        collections = processor.discover_collections(".")
        
        if collections:
            result = processor.process_collection(collections[0])
            if result.success:
                print(f"✓ Batch size {batch_size} works correctly")
            else:
                print(f"✗ Batch size {batch_size} failed: {result.error_message}")
                return False
    
    print("✓ Batch processing test passed!")
    return True


if __name__ == "__main__":
    print("PDF Analysis System - Integration Tests")
    print("=" * 50)
    
    success = True
    
    # Run all tests
    success &= test_integration()
    success &= test_error_handling()
    success &= test_batch_processing()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ All integration tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed!")
        sys.exit(1)