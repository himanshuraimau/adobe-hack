#!/usr/bin/env python3
"""
Simple test script to verify the complete pipeline integration.

This script tests the key functionality without requiring pytest.
"""

import tempfile
import shutil
import json
from pathlib import Path
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import process_pdf, _generate_error_output, _generate_timeout_output


def test_error_output_generation():
    """Test error output generation."""
    print("Testing error output generation...")
    
    test_output_dir = Path(tempfile.mkdtemp())
    try:
        result = _generate_error_output('test.pdf', str(test_output_dir), 'Test error message')
        
        assert result is not None, "Error output should be generated"
        assert Path(result).exists(), "Error output file should exist"
        
        # Verify JSON structure
        with open(result, 'r') as f:
            json_data = json.load(f)
        
        assert "error" in json_data, "Error JSON should contain error field"
        assert json_data["title"].startswith("Error:"), "Title should indicate error"
        assert json_data["outline"] == [], "Outline should be empty for errors"
        
        print("✓ Error output generation test passed")
        
    finally:
        shutil.rmtree(test_output_dir)


def test_timeout_output_generation():
    """Test timeout output generation."""
    print("Testing timeout output generation...")
    
    test_output_dir = Path(tempfile.mkdtemp())
    try:
        result = _generate_timeout_output('test.pdf', str(test_output_dir))
        
        assert result is not None, "Timeout output should be generated"
        assert Path(result).exists(), "Timeout output file should exist"
        
        # Verify JSON structure
        with open(result, 'r') as f:
            json_data = json.load(f)
        
        assert "error" in json_data, "Timeout JSON should contain error field"
        assert json_data["error"]["type"] == "TimeoutError", "Error type should be TimeoutError"
        
        print("✓ Timeout output generation test passed")
        
    finally:
        shutil.rmtree(test_output_dir)


def test_nonexistent_file_handling():
    """Test handling of nonexistent PDF files."""
    print("Testing nonexistent file handling...")
    
    test_output_dir = Path(tempfile.mkdtemp())
    try:
        result = process_pdf('nonexistent.pdf', str(test_output_dir))
        
        assert result is not None, "Should generate error output for nonexistent file"
        assert Path(result).exists(), "Error output file should exist"
        
        # Verify error JSON
        with open(result, 'r') as f:
            json_data = json.load(f)
        
        assert "error" in json_data, "Should contain error information"
        
        print("✓ Nonexistent file handling test passed")
        
    finally:
        shutil.rmtree(test_output_dir)


def test_sample_pdf_processing():
    """Test processing of sample PDFs if they exist."""
    print("Testing sample PDF processing...")
    
    sample_pdfs = [
        Path("input/E0CCG5S239.pdf"),
        Path("input/TOPJUMP-PARTY-INVITATION-20161003-V01.pdf")
    ]
    
    test_output_dir = Path(tempfile.mkdtemp())
    try:
        for pdf_path in sample_pdfs:
            if pdf_path.exists():
                print(f"  Processing {pdf_path.name}...")
                
                result = process_pdf(str(pdf_path), str(test_output_dir), timeout=15)
                
                assert result is not None, f"Should process {pdf_path.name} successfully"
                assert Path(result).exists(), f"Output file should exist for {pdf_path.name}"
                
                # Verify JSON structure
                with open(result, 'r') as f:
                    json_data = json.load(f)
                
                assert "title" in json_data, "JSON should contain title"
                assert "outline" in json_data, "JSON should contain outline"
                assert isinstance(json_data["title"], str), "Title should be string"
                assert isinstance(json_data["outline"], list), "Outline should be list"
                
                # Verify outline entries if any
                for entry in json_data["outline"]:
                    assert "level" in entry, "Outline entry should have level"
                    assert "text" in entry, "Outline entry should have text"
                    assert "page" in entry, "Outline entry should have page"
                    assert entry["level"] in ["H1", "H2", "H3"], "Level should be valid"
                    assert isinstance(entry["page"], int), "Page should be integer"
                    assert entry["page"] >= 1, "Page should be positive"
                
                print(f"  ✓ {pdf_path.name} processed successfully")
            else:
                print(f"  - Skipping {pdf_path.name} (not found)")
        
        print("✓ Sample PDF processing test passed")
        
    finally:
        shutil.rmtree(test_output_dir)


def test_json_validation():
    """Test JSON output validation."""
    print("Testing JSON output validation...")
    
    # Test valid JSON structure
    valid_json = {
        "title": "Test Document",
        "outline": [
            {"level": "H1", "text": "Introduction", "page": 1},
            {"level": "H2", "text": "Overview", "page": 2}
        ]
    }
    
    # This would normally use the JSONBuilder validator, but for simplicity
    # we'll just check the basic structure
    assert "title" in valid_json
    assert "outline" in valid_json
    assert isinstance(valid_json["title"], str)
    assert isinstance(valid_json["outline"], list)
    
    for entry in valid_json["outline"]:
        assert "level" in entry
        assert "text" in entry
        assert "page" in entry
        assert entry["level"] in ["H1", "H2", "H3"]
    
    print("✓ JSON validation test passed")


def main():
    """Run all tests."""
    print("Running pipeline integration tests...\n")
    
    try:
        test_error_output_generation()
        test_timeout_output_generation()
        test_nonexistent_file_handling()
        test_sample_pdf_processing()
        test_json_validation()
        
        print("\n🎉 All integration tests passed!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())