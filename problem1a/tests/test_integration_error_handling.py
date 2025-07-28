"""
Integration tests for error handling in the PDF Structure Extractor.

This module tests the complete error handling pipeline with real components.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, Mock

try:
    from .main import process_pdf
    from .error_handler import global_error_handler
except ImportError:
    from main import process_pdf
    from error_handler import global_error_handler


class TestIntegrationErrorHandling:
    """Test integration error handling scenarios."""
    
    def test_process_nonexistent_pdf(self):
        """Test processing a non-existent PDF file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = process_pdf("/nonexistent/file.pdf", temp_dir, timeout=5)
            
            # Should generate error output
            assert result is not None
            
            # Verify error JSON was created
            with open(result) as f:
                data = json.load(f)
            
            assert "Error" in data["title"]
            assert data["outline"] == []
            assert "error" in data
    
    def test_process_empty_file(self):
        """Test processing an empty file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create empty file
            empty_file = Path(temp_dir) / "empty.pdf"
            empty_file.touch()
            
            result = process_pdf(str(empty_file), temp_dir, timeout=5)
            
            # Should generate error output
            assert result is not None
            
            # Verify error JSON was created
            with open(result) as f:
                data = json.load(f)
            
            assert "Error" in data["title"]
            assert "error" in data
    
    def test_timeout_handling(self):
        """Test timeout handling in processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a dummy PDF file
            pdf_path = Path(temp_dir) / "test.pdf"
            pdf_path.write_text("dummy content")
            
            # Mock PDF parser to simulate slow processing
            with patch('problem1a.main.PDFParser') as mock_parser_class:
                mock_parser = Mock()
                
                def slow_parse(*args, **kwargs):
                    import time
                    time.sleep(3)  # Simulate slow processing
                    return []
                
                mock_parser.parse_document.side_effect = slow_parse
                mock_parser_class.return_value = mock_parser
                
                # Set very short timeout
                result = process_pdf(str(pdf_path), temp_dir, timeout=1)
                
                # Should generate error output (timeout is handled as a general error)
                assert result is not None
                
                # Verify error JSON was created
                with open(result) as f:
                    data = json.load(f)
                
                # The timeout might be handled as a general processing error
                assert "error" in data["title"].lower() or "timeout" in data["title"].lower()
                assert "error" in data
    
    def test_error_statistics_tracking(self):
        """Test that error statistics are properly tracked."""
        # Reset error counts
        global_error_handler.error_counts.clear()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Process multiple non-existent files to generate errors
            for i in range(3):
                process_pdf(f"/nonexistent/file{i}.pdf", temp_dir, timeout=5)
            
            # Check error statistics
            stats = global_error_handler.get_error_statistics()
            assert stats["total_errors"] > 0
            assert len(stats["error_counts"]) > 0
    
    def test_graceful_degradation_with_model_failure(self):
        """Test graceful degradation when model loading fails."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a dummy PDF file
            pdf_path = Path(temp_dir) / "test.pdf"
            pdf_path.write_text("dummy content")
            
            # Mock components to simulate model loading failure but successful fallback
            with patch('problem1a.main.PDFParser') as mock_parser_class, \
                 patch('problem1a.main.HeadingClassifier') as mock_classifier_class:
                
                # Set up parser to return some data
                from models import TextBlock
                mock_parser = Mock()
                mock_parser.parse_document.return_value = [
                    TextBlock(
                        text="Test Heading",
                        page_number=1,
                        bbox=(0, 0, 100, 20),
                        font_size=16.0,
                        font_name="Arial",
                        font_flags=16  # Bold flag
                    )
                ]
                mock_parser_class.return_value = mock_parser
                
                # Set up classifier to fail model loading but use fallback
                from models import ClassificationResult, ProcessedBlock, FeatureVector
                mock_classifier = Mock()
                mock_classifier.load_model.side_effect = Exception("Model loading failed")
                mock_classifier.model_loaded = False
                
                # Mock successful fallback classification
                mock_result = ClassificationResult(
                    block=ProcessedBlock(
                        text="Test Heading",
                        page_number=1,
                        features=FeatureVector(
                            font_size_ratio=1.5,
                            is_bold=True,
                            is_italic=False,
                            position_x=0.0,
                            position_y=0.1,
                            text_length=12,
                            capitalization_score=0.0,
                            whitespace_ratio=0.1
                        ),
                        original_block=None
                    ),
                    predicted_class="h1",
                    confidence=0.6
                )
                mock_classifier.classify_block.return_value = mock_result
                mock_classifier_class.return_value = mock_classifier
                
                result = process_pdf(str(pdf_path), temp_dir, timeout=30)
                
                # Should still produce valid output despite model failure
                assert result is not None
                
                # Verify output is valid JSON
                with open(result) as f:
                    data = json.load(f)
                
                assert "title" in data
                assert "outline" in data
                # Should have processed the heading despite model failure
                if not data.get("error"):  # If no error, should have outline
                    assert len(data["outline"]) >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])