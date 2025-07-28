"""
Unit tests for JSON output generation system.

This module tests the JSONBuilder, OutputValidator, and ErrorHandler classes
to ensure proper JSON formatting, validation, and error handling.
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from .json_builder import JSONBuilder, OutputValidator, ErrorHandler
from .models import DocumentStructure, Heading


class TestJSONBuilder:
    """Test cases for JSONBuilder class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.json_builder = JSONBuilder()
    
    def test_build_json_with_title_and_headings(self):
        """Test building JSON with title and multiple headings."""
        # Create test document structure
        headings = [
            Heading(level="H1", text="Introduction", page=1, confidence=0.9),
            Heading(level="H2", text="Background", page=2, confidence=0.8),
            Heading(level="H3", text="Related Work", page=3, confidence=0.7),
            Heading(level="H1", text="Methodology", page=5, confidence=0.85)
        ]
        
        structure = DocumentStructure(
            title="Test Document",
            headings=headings,
            metadata={"pages": 10}
        )
        
        # Build JSON
        result = self.json_builder.build_json(structure)
        
        # Verify structure
        assert result["title"] == "Test Document"
        assert len(result["outline"]) == 4
        
        # Verify first heading
        assert result["outline"][0]["level"] == "H1"
        assert result["outline"][0]["text"] == "Introduction"
        assert result["outline"][0]["page"] == 1
        
        # Verify nested heading
        assert result["outline"][1]["level"] == "H2"
        assert result["outline"][1]["text"] == "Background"
        assert result["outline"][1]["page"] == 2
    
    def test_build_json_no_title(self):
        """Test building JSON when no title is provided."""
        headings = [
            Heading(level="H1", text="First Heading", page=1, confidence=0.9)
        ]
        
        structure = DocumentStructure(
            title=None,
            headings=headings,
            metadata={}
        )
        
        result = self.json_builder.build_json(structure)
        
        assert result["title"] == "Untitled Document"
        assert len(result["outline"]) == 1
    
    def test_build_json_empty_title(self):
        """Test building JSON with empty title."""
        headings = [
            Heading(level="H1", text="First Heading", page=1, confidence=0.9)
        ]
        
        structure = DocumentStructure(
            title="",
            headings=headings,
            metadata={}
        )
        
        result = self.json_builder.build_json(structure)
        
        assert result["title"] == "Untitled Document"
    
    def test_build_json_no_headings(self):
        """Test building JSON with no headings."""
        structure = DocumentStructure(
            title="Empty Document",
            headings=[],
            metadata={}
        )
        
        result = self.json_builder.build_json(structure)
        
        assert result["title"] == "Empty Document"
        assert result["outline"] == []
    
    def test_build_json_excludes_title_level_headings(self):
        """Test that title-level headings are excluded from outline."""
        headings = [
            Heading(level="title", text="Document Title", page=1, confidence=0.95),
            Heading(level="H1", text="First Section", page=1, confidence=0.9),
            Heading(level="H2", text="Subsection", page=2, confidence=0.8)
        ]
        
        structure = DocumentStructure(
            title="Test Document",
            headings=headings,
            metadata={}
        )
        
        result = self.json_builder.build_json(structure)
        
        # Should only include H1 and H2, not title
        assert len(result["outline"]) == 2
        assert result["outline"][0]["level"] == "H1"
        assert result["outline"][1]["level"] == "H2"
    
    def test_build_json_strips_whitespace(self):
        """Test that heading text is properly stripped of whitespace."""
        headings = [
            Heading(level="H1", text="  Heading with spaces  ", page=1, confidence=0.9)
        ]
        
        structure = DocumentStructure(
            title="Test Document",
            headings=headings,
            metadata={}
        )
        
        result = self.json_builder.build_json(structure)
        
        assert result["outline"][0]["text"] == "Heading with spaces"
    
    @patch('problem1a.json_builder.logger')
    def test_build_json_handles_exception(self, mock_logger):
        """Test that exceptions during JSON building are handled."""
        # Create a structure that will cause an exception
        structure = Mock()
        structure.title = "Test"
        structure.headings = None  # This will cause an error when iterating
        
        result = self.json_builder.build_json(structure)
        
        # Should return error JSON
        assert "Error: Processing Failed" in result["title"]
        assert result["outline"] == []
        assert "error" in result
        mock_logger.error.assert_called()
    
    def test_write_json_creates_directory(self):
        """Test that write_json creates output directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "subdir" / "output.json"
            
            json_data = {"title": "Test", "outline": []}
            
            self.json_builder.write_json(json_data, str(output_path))
            
            # Verify file was created
            assert output_path.exists()
            
            # Verify content
            with open(output_path, 'r') as f:
                loaded_data = json.load(f)
            
            assert loaded_data == json_data
    
    def test_write_json_proper_formatting(self):
        """Test that JSON is written with proper formatting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "output.json"
            
            json_data = {
                "title": "Test Document",
                "outline": [
                    {"level": "H1", "text": "Heading", "page": 1}
                ]
            }
            
            self.json_builder.write_json(json_data, str(output_path))
            
            # Read raw content to check formatting
            with open(output_path, 'r') as f:
                content = f.read()
            
            # Should be indented (not minified)
            assert "  " in content
            assert "\n" in content
    
    @patch('problem1a.json_builder.logger')
    def test_write_json_handles_write_error(self, mock_logger):
        """Test that write errors are handled gracefully."""
        # Try to write to an invalid path (use a path that will definitely fail)
        invalid_path = "/root/invalid/path/output.json"  # This should fail due to permissions
        json_data = {"title": "Test", "outline": []}
        
        # Should not raise exception, but will log error
        try:
            self.json_builder.write_json(json_data, invalid_path)
        except Exception:
            # If it still raises an exception, that's expected for this test
            pass
        
        # Should log error
        mock_logger.error.assert_called()
    
    def test_process_and_write_integration(self):
        """Test the integrated process_and_write method."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "output.json"
            
            headings = [
                Heading(level="H1", text="Introduction", page=1, confidence=0.9)
            ]
            
            structure = DocumentStructure(
                title="Integration Test",
                headings=headings,
                metadata={}
            )
            
            result = self.json_builder.process_and_write(structure, str(output_path))
            
            # Verify return value
            assert result["title"] == "Integration Test"
            assert len(result["outline"]) == 1
            
            # Verify file was written
            assert output_path.exists()
            
            # Verify file content matches return value
            with open(output_path, 'r') as f:
                file_data = json.load(f)
            
            assert file_data == result


class TestOutputValidator:
    """Test cases for OutputValidator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = OutputValidator()
    
    def test_validate_valid_output(self):
        """Test validation of valid JSON output."""
        valid_json = {
            "title": "Test Document",
            "outline": [
                {"level": "H1", "text": "Introduction", "page": 1},
                {"level": "H2", "text": "Background", "page": 2},
                {"level": "H3", "text": "Details", "page": 3}
            ]
        }
        
        assert self.validator.validate_output(valid_json) is True
    
    def test_validate_empty_outline(self):
        """Test validation with empty outline."""
        valid_json = {
            "title": "Empty Document",
            "outline": []
        }
        
        assert self.validator.validate_output(valid_json) is True
    
    def test_validate_not_dictionary(self):
        """Test validation fails for non-dictionary input."""
        invalid_json = "not a dictionary"
        
        assert self.validator.validate_output(invalid_json) is False
    
    def test_validate_missing_title(self):
        """Test validation fails when title is missing."""
        invalid_json = {
            "outline": []
        }
        
        assert self.validator.validate_output(invalid_json) is False
    
    def test_validate_missing_outline(self):
        """Test validation fails when outline is missing."""
        invalid_json = {
            "title": "Test"
        }
        
        assert self.validator.validate_output(invalid_json) is False
    
    def test_validate_title_not_string(self):
        """Test validation fails when title is not a string."""
        invalid_json = {
            "title": 123,
            "outline": []
        }
        
        assert self.validator.validate_output(invalid_json) is False
    
    def test_validate_outline_not_list(self):
        """Test validation fails when outline is not a list."""
        invalid_json = {
            "title": "Test",
            "outline": "not a list"
        }
        
        assert self.validator.validate_output(invalid_json) is False
    
    def test_validate_outline_entry_not_dict(self):
        """Test validation fails when outline entry is not a dictionary."""
        invalid_json = {
            "title": "Test",
            "outline": ["not a dict"]
        }
        
        assert self.validator.validate_output(invalid_json) is False
    
    def test_validate_outline_entry_missing_keys(self):
        """Test validation fails when outline entry is missing required keys."""
        invalid_json = {
            "title": "Test",
            "outline": [
                {"level": "H1", "text": "Missing page"}
            ]
        }
        
        assert self.validator.validate_output(invalid_json) is False
    
    def test_validate_invalid_level(self):
        """Test validation fails for invalid heading levels."""
        invalid_json = {
            "title": "Test",
            "outline": [
                {"level": "H4", "text": "Invalid level", "page": 1}
            ]
        }
        
        assert self.validator.validate_output(invalid_json) is False
    
    def test_validate_empty_text(self):
        """Test validation fails for empty text."""
        invalid_json = {
            "title": "Test",
            "outline": [
                {"level": "H1", "text": "", "page": 1}
            ]
        }
        
        assert self.validator.validate_output(invalid_json) is False
    
    def test_validate_whitespace_only_text(self):
        """Test validation fails for whitespace-only text."""
        invalid_json = {
            "title": "Test",
            "outline": [
                {"level": "H1", "text": "   ", "page": 1}
            ]
        }
        
        assert self.validator.validate_output(invalid_json) is False
    
    def test_validate_invalid_page_number(self):
        """Test validation fails for invalid page numbers."""
        invalid_json = {
            "title": "Test",
            "outline": [
                {"level": "H1", "text": "Test", "page": 0}
            ]
        }
        
        assert self.validator.validate_output(invalid_json) is False
        
        invalid_json["outline"][0]["page"] = -1
        assert self.validator.validate_output(invalid_json) is False
        
        invalid_json["outline"][0]["page"] = "not a number"
        assert self.validator.validate_output(invalid_json) is False
    
    @patch('problem1a.json_builder.logger')
    def test_validate_handles_exception(self, mock_logger):
        """Test that validation handles exceptions gracefully."""
        # Create an object that will cause an exception during validation
        # Use a dict-like object that raises an exception when accessed
        class ProblematicDict(dict):
            def __getitem__(self, key):
                raise Exception("Test exception")
        
        problematic_json = ProblematicDict()
        
        result = self.validator.validate_output(problematic_json)
        
        assert result is False
        mock_logger.error.assert_called()


class TestErrorHandler:
    """Test cases for ErrorHandler class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.error_handler = ErrorHandler()
    
    def test_handle_errors_general_exception(self):
        """Test handling of general exceptions."""
        test_error = ValueError("Test error message")
        
        result = self.error_handler.handle_errors(test_error)
        
        assert result["title"] == "Error: Processing Failed"
        assert result["outline"] == []
        assert result["error"]["type"] == "ValueError"
        assert result["error"]["message"] == "Test error message"
        assert "details" in result["error"]
    
    def test_handle_validation_error(self):
        """Test handling of validation errors."""
        result = self.error_handler.handle_validation_error()
        
        assert result["title"] == "Error: Invalid Output Format"
        assert result["outline"] == []
        assert result["error"]["type"] == "ValidationError"
        assert "validation" in result["error"]["message"]
    
    def test_handle_write_error(self):
        """Test handling of write errors."""
        write_error = IOError("Permission denied")
        
        result = self.error_handler.handle_write_error(write_error)
        
        assert result["title"] == "Error: File Write Failed"
        assert result["outline"] == []
        assert result["error"]["type"] == "WriteError"
        assert result["error"]["message"] == "Permission denied"
    
    def test_handle_no_title_found(self):
        """Test handling when no title is found."""
        result = self.error_handler.handle_no_title_found()
        
        assert result["title"] == "Untitled Document"
        assert result["outline"] == []
        assert result["error"]["type"] == "NoTitleFound"
    
    def test_handle_no_headings_detected(self):
        """Test handling when no headings are detected."""
        result = self.error_handler.handle_no_headings_detected()
        
        assert result["title"] == "Document"
        assert result["outline"] == []
        assert result["error"]["type"] == "NoHeadingsDetected"
    
    @patch('problem1a.json_builder.logger')
    def test_error_logging(self, mock_logger):
        """Test that errors are properly logged."""
        test_error = RuntimeError("Test runtime error")
        
        self.error_handler.handle_errors(test_error)
        
        mock_logger.error.assert_called_with(f"Handling error: {test_error}")


class TestJSONBuilderIntegration:
    """Integration tests for the complete JSON building system."""
    
    def test_complete_workflow_success(self):
        """Test complete workflow from structure to file output."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_output.json"
            
            # Create comprehensive test structure
            headings = [
                Heading(level="H1", text="Chapter 1: Introduction", page=1, confidence=0.95),
                Heading(level="H2", text="1.1 Overview", page=1, confidence=0.9),
                Heading(level="H2", text="1.2 Objectives", page=2, confidence=0.88),
                Heading(level="H3", text="1.2.1 Primary Goals", page=2, confidence=0.85),
                Heading(level="H1", text="Chapter 2: Methodology", page=5, confidence=0.92),
                Heading(level="H2", text="2.1 Approach", page=5, confidence=0.87)
            ]
            
            structure = DocumentStructure(
                title="Research Paper: Advanced PDF Processing",
                headings=headings,
                metadata={"total_pages": 20, "language": "en"}
            )
            
            # Process and write
            json_builder = JSONBuilder()
            result = json_builder.process_and_write(structure, str(output_path))
            
            # Verify result structure
            assert result["title"] == "Research Paper: Advanced PDF Processing"
            assert len(result["outline"]) == 6
            
            # Verify hierarchy is preserved
            assert result["outline"][0]["level"] == "H1"
            assert result["outline"][0]["text"] == "Chapter 1: Introduction"
            assert result["outline"][1]["level"] == "H2"
            assert result["outline"][1]["text"] == "1.1 Overview"
            
            # Verify file was written correctly
            assert output_path.exists()
            
            with open(output_path, 'r') as f:
                file_content = json.load(f)
            
            assert file_content == result
    
    def test_error_recovery_workflow(self):
        """Test error recovery in the complete workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "error_test.json"
            
            # Create structure that will pass validation but test error handling
            structure = DocumentStructure(
                title=None,  # This should trigger fallback
                headings=[],  # Empty headings
                metadata={}
            )
            
            json_builder = JSONBuilder()
            result = json_builder.process_and_write(structure, str(output_path))
            
            # Should handle gracefully
            assert result["title"] == "Untitled Document"
            assert result["outline"] == []
            
            # File should still be written
            assert output_path.exists()