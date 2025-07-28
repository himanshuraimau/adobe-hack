"""
Integration tests for the complete PDF processing pipeline.

This module tests the end-to-end processing pipeline using the provided sample PDFs
to ensure all components work together correctly.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import logging

from main import process_pdf, main, _generate_error_output, _generate_timeout_output
from src.pdf_extractor.config.config import config
from src.pdf_extractor.models.models import DocumentStructure, Heading


class TestPipelineIntegration:
    """Test the complete PDF processing pipeline."""
    
    def setup_method(self):
        """Set up test environment."""
        self.test_output_dir = Path(tempfile.mkdtemp())
        self.sample_pdfs_dir = Path("input")
        
        # Ensure sample PDFs exist
        self.sample_pdf1 = self.sample_pdfs_dir / "E0CCG5S239.pdf"
        self.sample_pdf2 = self.sample_pdfs_dir / "TOPJUMP-PARTY-INVITATION-20161003-V01.pdf"
        
        # Set up logging for tests
        logging.basicConfig(level=logging.INFO)
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.test_output_dir.exists():
            shutil.rmtree(self.test_output_dir)
    
    def test_process_single_pdf_success(self):
        """Test successful processing of a single PDF."""
        if not self.sample_pdf1.exists():
            pytest.skip(f"Sample PDF not found: {self.sample_pdf1}")
        
        # Process the PDF
        result_path = process_pdf(
            str(self.sample_pdf1), 
            str(self.test_output_dir),
            timeout=15  # Give extra time for testing
        )
        
        # Verify result
        assert result_path is not None
        assert Path(result_path).exists()
        
        # Verify JSON structure
        with open(result_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        self._validate_json_structure(json_data)
    
    def test_process_second_pdf_success(self):
        """Test successful processing of the second sample PDF."""
        if not self.sample_pdf2.exists():
            pytest.skip(f"Sample PDF not found: {self.sample_pdf2}")
        
        # Process the PDF
        result_path = process_pdf(
            str(self.sample_pdf2), 
            str(self.test_output_dir),
            timeout=15  # Give extra time for testing
        )
        
        # Verify result
        assert result_path is not None
        assert Path(result_path).exists()
        
        # Verify JSON structure
        with open(result_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        self._validate_json_structure(json_data)
    
    def test_process_nonexistent_pdf(self):
        """Test processing of non-existent PDF file."""
        nonexistent_pdf = "nonexistent.pdf"
        
        result_path = process_pdf(
            nonexistent_pdf,
            str(self.test_output_dir)
        )
        
        # Should generate error output
        assert result_path is not None
        assert Path(result_path).exists()
        
        # Verify error JSON structure
        with open(result_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        assert "error" in json_data
        assert json_data["title"].startswith("Error:")
        assert json_data["outline"] == []
    
    def test_process_pdf_timeout(self):
        """Test processing timeout handling."""
        if not self.sample_pdf1.exists():
            pytest.skip(f"Sample PDF not found: {self.sample_pdf1}")
        
        # Process with very short timeout
        result_path = process_pdf(
            str(self.sample_pdf1),
            str(self.test_output_dir),
            timeout=1  # Very short timeout
        )
        
        # Should generate timeout output
        assert result_path is not None
        assert Path(result_path).exists()
        
        # Verify timeout JSON structure
        with open(result_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # Should be either successful (if processing was fast) or timeout error
        if "error" in json_data:
            assert json_data["error"]["type"] == "TimeoutError"
            assert json_data["title"].startswith("Error:")
            assert json_data["outline"] == []
    
    def test_main_function_single_file(self):
        """Test main function with single PDF file."""
        if not self.sample_pdf1.exists():
            pytest.skip(f"Sample PDF not found: {self.sample_pdf1}")
        
        # Mock sys.argv
        test_args = [
            "main.py",
            str(self.sample_pdf1),
            "-o", str(self.test_output_dir),
            "--timeout", "15"
        ]
        
        with patch('sys.argv', test_args):
            try:
                main()
            except SystemExit as e:
                # Should exit with code 0 on success
                assert e.code == 0 or e.code is None
        
        # Verify output file was created
        expected_output = self.test_output_dir / f"{self.sample_pdf1.stem}.json"
        assert expected_output.exists()
    
    def test_main_function_directory(self):
        """Test main function with directory input."""
        if not (self.sample_pdf1.exists() or self.sample_pdf2.exists()):
            pytest.skip("No sample PDFs found")
        
        # Mock sys.argv
        test_args = [
            "main.py",
            str(self.sample_pdfs_dir),
            "-o", str(self.test_output_dir),
            "--timeout", "15"
        ]
        
        with patch('sys.argv', test_args):
            try:
                main()
            except SystemExit as e:
                # Should exit with code 0 on success
                assert e.code == 0 or e.code is None
        
        # Verify output files were created
        json_files = list(self.test_output_dir.glob("*.json"))
        assert len(json_files) > 0
    
    def test_error_output_generation(self):
        """Test error output generation."""
        error_message = "Test error message"
        pdf_path = "test.pdf"
        
        result_path = _generate_error_output(pdf_path, str(self.test_output_dir), error_message)
        
        assert result_path is not None
        assert Path(result_path).exists()
        
        # Verify error JSON structure
        with open(result_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        assert json_data["title"].startswith("Error:")
        assert json_data["outline"] == []
        assert json_data["error"]["message"] == error_message
        assert json_data["error"]["type"] == "ProcessingError"
    
    def test_timeout_output_generation(self):
        """Test timeout output generation."""
        pdf_path = "test.pdf"
        
        result_path = _generate_timeout_output(pdf_path, str(self.test_output_dir))
        
        assert result_path is not None
        assert Path(result_path).exists()
        
        # Verify timeout JSON structure
        with open(result_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        assert json_data["title"].startswith("Error:")
        assert json_data["outline"] == []
        assert json_data["error"]["type"] == "TimeoutError"
    
    def test_pipeline_with_mock_components(self):
        """Test pipeline with mocked components to verify integration."""
        # Create mock document structure
        mock_structure = DocumentStructure(
            title="Test Document",
            headings=[
                Heading(level="H1", text="Introduction", page=1, confidence=0.9),
                Heading(level="H2", text="Overview", page=1, confidence=0.8),
                Heading(level="H1", text="Conclusion", page=2, confidence=0.9)
            ],
            metadata={"total_blocks": 10, "heading_blocks": 3}
        )
        
        # Mock the structure analyzer to return our test structure
        with patch('main.StructureAnalyzer') as mock_analyzer_class:
            mock_analyzer = MagicMock()
            mock_analyzer.analyze_structure.return_value = mock_structure
            mock_analyzer_class.return_value = mock_analyzer
            
            # Create a temporary test PDF (empty file for this test)
            test_pdf = self.test_output_dir / "test.pdf"
            test_pdf.write_bytes(b"dummy pdf content")
            
            # Process the mock PDF
            result_path = process_pdf(
                str(test_pdf),
                str(self.test_output_dir)
            )
            
            # Verify the result
            assert result_path is not None
            assert Path(result_path).exists()
            
            # Verify JSON content matches our mock structure
            with open(result_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            assert json_data["title"] == "Test Document"
            assert len(json_data["outline"]) == 3
            assert json_data["outline"][0]["level"] == "H1"
            assert json_data["outline"][0]["text"] == "Introduction"
            assert json_data["outline"][0]["page"] == 1
    
    def test_performance_within_timeout(self):
        """Test that processing completes within the timeout limit."""
        if not self.sample_pdf1.exists():
            pytest.skip(f"Sample PDF not found: {self.sample_pdf1}")
        
        import time
        
        start_time = time.time()
        result_path = process_pdf(
            str(self.sample_pdf1),
            str(self.test_output_dir),
            timeout=10
        )
        processing_time = time.time() - start_time
        
        # Should complete within timeout
        assert processing_time < 10
        assert result_path is not None
    
    def _validate_json_structure(self, json_data):
        """Validate that JSON data matches the required structure."""
        # Check required top-level keys
        assert "title" in json_data
        assert "outline" in json_data
        
        # Validate title
        assert isinstance(json_data["title"], str)
        assert len(json_data["title"]) > 0
        
        # Validate outline
        assert isinstance(json_data["outline"], list)
        
        # Validate each outline entry
        for entry in json_data["outline"]:
            assert isinstance(entry, dict)
            assert "level" in entry
            assert "text" in entry
            assert "page" in entry
            
            # Validate level
            assert entry["level"] in ["H1", "H2", "H3"]
            
            # Validate text
            assert isinstance(entry["text"], str)
            assert len(entry["text"].strip()) > 0
            
            # Validate page
            assert isinstance(entry["page"], int)
            assert entry["page"] >= 1


class TestPipelineErrorHandling:
    """Test error handling in the pipeline."""
    
    def setup_method(self):
        """Set up test environment."""
        self.test_output_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.test_output_dir.exists():
            shutil.rmtree(self.test_output_dir)
    
    def test_corrupted_pdf_handling(self):
        """Test handling of corrupted PDF files."""
        # Create a fake corrupted PDF
        corrupted_pdf = self.test_output_dir / "corrupted.pdf"
        corrupted_pdf.write_text("This is not a valid PDF file")
        
        result_path = process_pdf(
            str(corrupted_pdf),
            str(self.test_output_dir)
        )
        
        # Should generate error output
        assert result_path is not None
        assert Path(result_path).exists()
        
        # Verify error JSON
        with open(result_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        assert "error" in json_data
        assert json_data["title"].startswith("Error:")
    
    def test_empty_pdf_handling(self):
        """Test handling of empty PDF files."""
        # Create an empty file
        empty_pdf = self.test_output_dir / "empty.pdf"
        empty_pdf.write_bytes(b"")
        
        result_path = process_pdf(
            str(empty_pdf),
            str(self.test_output_dir)
        )
        
        # Should generate error output
        assert result_path is not None
        assert Path(result_path).exists()
        
        # Verify error JSON
        with open(result_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        assert "error" in json_data
    
    def test_model_loading_failure(self):
        """Test handling when model loading fails."""
        if not Path("input/E0CCG5S239.pdf").exists():
            pytest.skip("Sample PDF not found")
        
        # Mock the classifier to fail model loading
        with patch('main.HeadingClassifier') as mock_classifier_class:
            mock_classifier = MagicMock()
            mock_classifier.load_model.side_effect = Exception("Model loading failed")
            mock_classifier_class.return_value = mock_classifier
            
            result_path = process_pdf(
                "input/E0CCG5S239.pdf",
                str(self.test_output_dir)
            )
            
            # Should still generate some output (fallback classification)
            assert result_path is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])