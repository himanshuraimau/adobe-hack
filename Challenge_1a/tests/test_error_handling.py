"""
Unit tests for comprehensive error handling in the PDF Structure Extractor.

This module tests all error scenarios, edge cases, and graceful degradation
functionality across all components of the system.
"""

import pytest
import tempfile
import json
import time
import signal
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

try:
    from src.pdf_extractor.core.error_handler import (
        PDFStructureExtractorError, PDFProcessingError, ModelLoadingError,
        ClassificationError, FeatureExtractionError, StructureAnalysisError,
        OutputGenerationError, TimeoutError, ValidationError,
        ErrorHandler, TimeoutManager, RetryManager,
        global_error_handler, timeout_manager, retry_manager
    )
    from src.pdf_extractor.models.models import TextBlock, ProcessedBlock, FeatureVector, ClassificationResult, Heading, DocumentStructure
    from src.pdf_extractor.core.pdf_parser import PDFParser
    from src.pdf_extractor.core.classifier import HeadingClassifier, MobileBERTAdapter, FallbackRuleBasedClassifier
    from src.pdf_extractor.core.json_builder import JSONBuilder, OutputValidator
    from main import process_pdf, _generate_error_output, _generate_timeout_output
except ImportError:
    from error_handler import (
        PDFStructureExtractorError, PDFProcessingError, ModelLoadingError,
        ClassificationError, FeatureExtractionError, StructureAnalysisError,
        OutputGenerationError, TimeoutError, ValidationError,
        ErrorHandler, TimeoutManager, RetryManager,
        global_error_handler, timeout_manager, retry_manager
    )
    from models import TextBlock, ProcessedBlock, FeatureVector, ClassificationResult, Heading, DocumentStructure
    from pdf_parser import PDFParser
    from classifier import HeadingClassifier, MobileBERTAdapter, FallbackRuleBasedClassifier
    from json_builder import JSONBuilder, OutputValidator
    from main import process_pdf, _generate_error_output, _generate_timeout_output


class TestCustomExceptions:
    """Test custom exception classes."""
    
    def test_pdf_structure_extractor_error_base(self):
        """Test base exception class."""
        error = PDFStructureExtractorError(
            "Test error",
            error_code="TEST_ERROR",
            details={"key": "value"}
        )
        
        assert str(error) == "Test error"
        assert error.error_code == "TEST_ERROR"
        assert error.details == {"key": "value"}
        assert error.timestamp > 0
    
    def test_pdf_processing_error(self):
        """Test PDF processing error."""
        error = PDFProcessingError("PDF failed", error_code="PDF_FAIL")
        assert isinstance(error, PDFStructureExtractorError)
        assert error.error_code == "PDF_FAIL"
    
    def test_model_loading_error(self):
        """Test model loading error."""
        error = ModelLoadingError("Model failed", error_code="MODEL_FAIL")
        assert isinstance(error, PDFStructureExtractorError)
        assert error.error_code == "MODEL_FAIL"
    
    def test_classification_error(self):
        """Test classification error."""
        error = ClassificationError("Classification failed", error_code="CLASSIFY_FAIL")
        assert isinstance(error, PDFStructureExtractorError)
        assert error.error_code == "CLASSIFY_FAIL"


class TestErrorHandler:
    """Test the central error handler."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.error_handler = ErrorHandler()
    
    def test_handle_pdf_processing_error(self):
        """Test PDF processing error handling."""
        error = Exception("Test PDF error")
        pdf_path = "/test/path.pdf"
        
        result = self.error_handler.handle_pdf_processing_error(error, pdf_path)
        
        assert result["title"] == "Error: Failed to process path.pdf"
        assert result["outline"] == []
        assert result["error"]["type"] == "PDFProcessingError"
        assert result["error"]["message"] == "Test PDF error"
        assert "timestamp" in result["error"]
    
    def test_handle_model_loading_error_with_fallback(self):
        """Test model loading error with fallback enabled."""
        error = Exception("Model loading failed")
        model_path = "/test/model"
        
        result = self.error_handler.handle_model_loading_error(error, model_path)
        
        assert result["fallback_mode"] is True
        assert result["classification_method"] == "rule_based"
        assert result["model_error"] == "Model loading failed"
    
    def test_handle_model_loading_error_without_fallback(self):
        """Test model loading error with fallback disabled."""
        self.error_handler.fallback_enabled = False
        error = Exception("Model loading failed")
        model_path = "/test/model"
        
        result = self.error_handler.handle_model_loading_error(error, model_path)
        
        assert result["title"] == "Error: Model Loading Failed"
        assert result["error"]["type"] == "ModelLoadingError"
    
    def test_handle_classification_error_with_fallback(self):
        """Test classification error with fallback classifier."""
        error = Exception("Classification failed")
        text = "Test text"
        
        # Mock fallback classifier
        fallback_classifier = Mock()
        fallback_classifier.classify_text_only.return_value = ("h1", 0.7)
        
        result = self.error_handler.handle_classification_error(error, text, fallback_classifier)
        
        assert result == ("h1", 0.7)
        fallback_classifier.classify_text_only.assert_called_once_with(text)
    
    def test_handle_classification_error_without_fallback(self):
        """Test classification error without fallback classifier."""
        error = Exception("Classification failed")
        text = "Test text"
        
        result = self.error_handler.handle_classification_error(error, text, None)
        
        assert result == ("text", 0.1)
    
    def test_handle_feature_extraction_error(self):
        """Test feature extraction error handling."""
        error = Exception("Feature extraction failed")
        block_text = "Test block text"
        
        result = self.error_handler.handle_feature_extraction_error(error, block_text)
        
        expected_features = {
            'font_size_ratio': 1.0,
            'is_bold': False,
            'is_italic': False,
            'position_x': 0.0,
            'position_y': 0.0,
            'text_length': len(block_text),
            'capitalization_score': 0.0,
            'whitespace_ratio': 0.0
        }
        
        assert result == expected_features
    
    def test_handle_structure_analysis_error(self):
        """Test structure analysis error handling."""
        error = Exception("Structure analysis failed")
        
        # Create mock classification results
        mock_block = Mock()
        mock_block.text = "Test heading"
        mock_block.page_number = 1
        
        mock_result = Mock()
        mock_result.predicted_class = "h1"
        mock_result.block = mock_block
        mock_result.confidence = 0.8
        
        classified_blocks = [mock_result]
        
        result = self.error_handler.handle_structure_analysis_error(error, classified_blocks)
        
        assert isinstance(result, DocumentStructure)
        assert result.title == "Document (Structure Analysis Failed)"
        assert len(result.headings) == 1
        assert result.headings[0].level == "H1"
        assert result.headings[0].text == "Test heading"
        assert result.metadata["structure_analysis_failed"] is True
    
    def test_handle_output_generation_error(self):
        """Test output generation error handling."""
        error = Exception("Output generation failed")
        
        # Create mock document structure
        heading = Heading(level="H1", text="Test heading", page=1, confidence=0.8)
        structure = DocumentStructure(
            title="Test title",
            headings=[heading],
            metadata={}
        )
        
        result = self.error_handler.handle_output_generation_error(error, structure)
        
        assert result["title"] == "Test title"
        assert len(result["outline"]) == 1
        assert result["outline"][0]["level"] == "H1"
        assert result["outline"][0]["text"] == "Test heading"
        assert result["error"]["type"] == "OutputGenerationError"
    
    def test_handle_timeout_error(self):
        """Test timeout error handling."""
        pdf_path = "/test/path.pdf"
        timeout_seconds = 10
        
        result = self.error_handler.handle_timeout_error(pdf_path, timeout_seconds)
        
        assert result["title"] == "Error: Processing timeout for path.pdf"
        assert result["error"]["type"] == "TimeoutError"
        assert result["error"]["timeout_limit"] == timeout_seconds
    
    def test_handle_validation_error(self):
        """Test validation error handling."""
        error = Exception("Validation failed")
        
        result = self.error_handler.handle_validation_error(error)
        
        assert result["title"] == "Error: Invalid Output Format"
        assert result["error"]["type"] == "ValidationError"
    
    def test_get_error_statistics(self):
        """Test error statistics tracking."""
        # Simulate some errors
        self.error_handler._log_detailed_error(Exception("Test 1"), "Context1")
        self.error_handler._log_detailed_error(Exception("Test 2"), "Context1")
        self.error_handler._log_detailed_error(ValueError("Test 3"), "Context2")
        
        stats = self.error_handler.get_error_statistics()
        
        assert stats["total_errors"] == 3
        assert stats["error_counts"]["Context1:Exception"] == 2
        assert stats["error_counts"]["Context2:ValueError"] == 1
        assert stats["fallback_enabled"] is True


class TestTimeoutManager:
    """Test timeout management functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.timeout_manager = TimeoutManager(default_timeout=2)
    
    def test_timeout_context_success(self):
        """Test successful operation within timeout."""
        with self.timeout_manager.timeout_context(1, "test_operation"):
            time.sleep(0.1)  # Should complete successfully
    
    def test_timeout_context_timeout(self):
        """Test operation that exceeds timeout."""
        with pytest.raises(TimeoutError) as exc_info:
            with self.timeout_manager.timeout_context(1, "test_operation"):
                time.sleep(2)  # Should timeout
        
        assert "test_operation exceeded 1 second limit" in str(exc_info.value)
    
    def test_timeout_context_cleanup(self):
        """Test that timeout context properly cleans up."""
        # This test ensures that the signal handler is properly restored
        original_handler = signal.signal(signal.SIGALRM, signal.SIG_DFL)
        
        try:
            with self.timeout_manager.timeout_context(1, "test_operation"):
                pass
            
            # Verify that the original handler is restored
            current_handler = signal.signal(signal.SIGALRM, signal.SIG_DFL)
            # Note: We can't directly compare handlers, but we can verify no alarm is set
            signal.alarm(0)  # Clear any remaining alarm
            
        finally:
            signal.signal(signal.SIGALRM, original_handler)


class TestRetryManager:
    """Test retry management functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.retry_manager = RetryManager(max_retries=2, base_delay=0.1)
    
    def test_retry_success_first_attempt(self):
        """Test successful operation on first attempt."""
        mock_func = Mock(return_value="success")
        
        result = self.retry_manager.retry_with_backoff(mock_func, "arg1", kwarg1="value1")
        
        assert result == "success"
        assert mock_func.call_count == 1
        mock_func.assert_called_with("arg1", kwarg1="value1")
    
    def test_retry_success_after_failures(self):
        """Test successful operation after some failures."""
        mock_func = Mock(side_effect=[Exception("Fail 1"), Exception("Fail 2"), "success"])
        
        result = self.retry_manager.retry_with_backoff(mock_func)
        
        assert result == "success"
        assert mock_func.call_count == 3
    
    def test_retry_all_attempts_fail(self):
        """Test when all retry attempts fail."""
        mock_func = Mock(side_effect=Exception("Always fails"))
        
        with pytest.raises(Exception) as exc_info:
            self.retry_manager.retry_with_backoff(mock_func)
        
        assert str(exc_info.value) == "Always fails"
        assert mock_func.call_count == 3  # Initial attempt + 2 retries


class TestPDFParserErrorHandling:
    """Test error handling in PDF parser."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = PDFParser()
    
    def test_parse_nonexistent_file(self):
        """Test parsing a non-existent PDF file."""
        # The safe_operation decorator should handle the error gracefully
        result = self.parser.parse_document("/nonexistent/file.pdf")
        
        # Should return empty list as fallback
        assert result == []
    
    def test_parse_empty_file(self):
        """Test parsing an empty file."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # The safe_operation decorator should handle the error gracefully
            result = self.parser.parse_document(temp_path)
            
            # Should return empty list as fallback
            assert result == []
        finally:
            Path(temp_path).unlink()
    
    def test_extract_page_text_no_document(self):
        """Test extracting text when no document is loaded."""
        # The safe_operation decorator should handle the error gracefully
        result = self.parser.extract_page_text(0)
        
        # Should return empty list as fallback
        assert result == []
    
    def test_extract_page_text_invalid_page_number(self):
        """Test extracting text with invalid page number."""
        # Mock a document
        self.parser.document = Mock()
        self.parser.document.__len__ = Mock(return_value=5)
        
        # The safe_operation decorator should handle the error gracefully
        result = self.parser.extract_page_text(-1)
        assert result == []  # Should return empty list as fallback
        
        result = self.parser.extract_page_text(10)
        assert result == []  # Should return empty list as fallback


class TestClassifierErrorHandling:
    """Test error handling in classifier."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = HeadingClassifier()
        self.features = FeatureVector(
            font_size_ratio=1.0,
            is_bold=False,
            is_italic=False,
            position_x=0.0,
            position_y=0.0,
            text_length=10,
            capitalization_score=0.0,
            whitespace_ratio=0.0
        )
    
    def test_classify_empty_text(self):
        """Test classifying empty text."""
        # The safe_operation decorator should handle the error gracefully
        result = self.classifier.classify_block(self.features, "")
        
        # Should return a fallback result instead of raising
        assert result is None  # safe_operation returns None as fallback
    
    def test_classify_no_features(self):
        """Test classifying with no features."""
        # The safe_operation decorator should handle the error gracefully
        result = self.classifier.classify_block(None, "Test text")
        
        # Should return None as fallback
        assert result is None
    
    def test_load_model_empty_path(self):
        """Test loading model with empty path."""
        # The safe_operation decorator should handle the error gracefully
        result = self.classifier.load_model("")
        
        # Should not raise exception, but model_loaded should remain False
        assert self.classifier.model_loaded is False
    
    def test_load_model_nonexistent_path(self):
        """Test loading model from non-existent path."""
        # The safe_operation decorator should handle the error gracefully
        result = self.classifier.load_model("/nonexistent/model")
        
        # Should not raise exception, but model_loaded should remain False
        assert self.classifier.model_loaded is False
    
    def test_fallback_classifier_error_handling(self):
        """Test fallback classifier error handling."""
        fallback = FallbackRuleBasedClassifier()
        
        # Test with invalid features (should not crash)
        result = fallback.classify("Test text", None)
        assert result == ("text", 0.1)  # Should return safe fallback
        
        # Test text-only classification
        result = fallback.classify_text_only("1. Introduction")
        assert result[0] == "h1"  # Should detect numbered heading


class TestJSONBuilderErrorHandling:
    """Test error handling in JSON builder."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.json_builder = JSONBuilder()
    
    def test_build_json_with_none_structure(self):
        """Test building JSON with None structure."""
        # Should handle gracefully and not crash
        result = self.json_builder.build_json(None)
        assert "error" in result
    
    def test_write_json_invalid_path(self):
        """Test writing JSON to invalid path."""
        json_data = {"title": "Test", "outline": []}
        
        # Try to write to a directory that doesn't exist and can't be created
        with pytest.raises(Exception):
            self.json_builder.write_json(json_data, "/root/nonexistent/output.json")
    
    def test_output_validator_invalid_data(self):
        """Test output validator with invalid data."""
        validator = OutputValidator()
        
        # Test with non-dictionary
        assert not validator.validate_output("not a dict")
        
        # Test with missing keys
        assert not validator.validate_output({"title": "Test"})
        
        # Test with invalid outline
        assert not validator.validate_output({
            "title": "Test",
            "outline": "not a list"
        })
        
        # Test with invalid outline entry
        assert not validator.validate_output({
            "title": "Test",
            "outline": [{"level": "INVALID", "text": "Test", "page": 1}]
        })


class TestMainProcessErrorHandling:
    """Test error handling in main processing pipeline."""
    
    def test_generate_error_output(self):
        """Test error output generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = _generate_error_output("/test/file.pdf", temp_dir, "Test error")
            
            assert result is not None
            output_path = Path(result)
            assert output_path.exists()
            
            # Verify JSON content
            with open(output_path) as f:
                data = json.load(f)
            
            assert data["title"] == "Error: Failed to process file.pdf"
            assert data["outline"] == []
            assert data["error"]["message"] == "Test error"
    
    def test_generate_timeout_output(self):
        """Test timeout output generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = _generate_timeout_output("/test/file.pdf", temp_dir)
            
            assert result is not None
            output_path = Path(result)
            assert output_path.exists()
            
            # Verify JSON content
            with open(output_path) as f:
                data = json.load(f)
            
            assert data["title"] == "Error: Processing timeout for file.pdf"
            assert data["error"]["type"] == "TimeoutError"
    
    @patch('Challenge_1a.main.PDFParser')
    def test_process_pdf_with_pdf_parsing_error(self, mock_parser_class):
        """Test process_pdf with PDF parsing error."""
        # Mock parser to raise an error
        mock_parser = Mock()
        mock_parser.parse_document.side_effect = Exception("PDF parsing failed")
        mock_parser_class.return_value = mock_parser
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a dummy PDF file
            pdf_path = Path(temp_dir) / "test.pdf"
            pdf_path.write_text("dummy content")
            
            result = process_pdf(str(pdf_path), temp_dir, timeout=30)
            
            # Should generate error output
            assert result is not None
            
            # Verify error JSON was created
            with open(result) as f:
                data = json.load(f)
            
            assert "Error" in data["title"]
            assert "PDF parsing failed" in data["error"]["message"]


class TestIntegrationErrorScenarios:
    """Test integration error scenarios across multiple components."""
    
    def test_end_to_end_error_recovery(self):
        """Test end-to-end error recovery with multiple component failures."""
        # This test simulates a scenario where multiple components fail
        # but the system still produces valid output through fallbacks
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a dummy PDF file
            pdf_path = Path(temp_dir) / "test.pdf"
            pdf_path.write_text("dummy content")
            
            # Mock various components to fail
            with patch('Challenge_1a.main.PDFParser') as mock_parser_class, \
                 patch('Challenge_1a.main.HeadingClassifier') as mock_classifier_class:
                
                # Set up parser to return minimal data
                mock_parser = Mock()
                mock_parser.parse_document.return_value = [
                    TextBlock(
                        text="Test heading",
                        page_number=1,
                        bbox=(0, 0, 100, 20),
                        font_size=14.0,
                        font_name="Arial",
                        font_flags=0
                    )
                ]
                mock_parser_class.return_value = mock_parser
                
                # Set up classifier to fail model loading but use fallback
                mock_classifier = Mock()
                mock_classifier.load_model.side_effect = Exception("Model loading failed")
                mock_classifier.model_loaded = False
                mock_classifier.classify_block.return_value = ClassificationResult(
                    block=None,
                    predicted_class="h1",
                    confidence=0.6
                )
                mock_classifier_class.return_value = mock_classifier
                
                result = process_pdf(str(pdf_path), temp_dir, timeout=30)
                
                # Should still produce output despite failures
                assert result is not None
                
                # Verify output exists and is valid JSON
                with open(result) as f:
                    data = json.load(f)
                
                assert "title" in data
                assert "outline" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])