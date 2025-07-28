"""
JSON output generation module for formatting document structure.

This module formats the extracted document structure into the required JSON format
with proper validation and error handling.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
try:
    from .models import DocumentStructure, Heading
    from .config import config
except ImportError:
    from models import DocumentStructure, Heading
    from config import config


logger = logging.getLogger(__name__)


class JSONBuilder:
    """Main JSON formatting logic for output generation."""
    
    def __init__(self):
        self.config = config.get_output_config()
        self.validator = OutputValidator()
        self.error_handler = ErrorHandler()
    
    def build_json(self, structure: DocumentStructure) -> Dict[str, Any]:
        """
        Build JSON output from document structure.
        
        Args:
            structure: DocumentStructure to format
            
        Returns:
            Dictionary in required JSON format
        """
        try:
            # Handle case where no title is found
            title = structure.title if structure.title else "Untitled Document"
            
            # Build outline from headings, excluding title-level headings from outline
            outline = []
            for heading in structure.headings:
                if heading.level != "title":  # Only include H1, H2, H3 in outline
                    outline_entry = {
                        "level": heading.level,
                        "text": heading.text.strip(),
                        "page": heading.page
                    }
                    outline.append(outline_entry)
            
            # Create the final JSON structure
            json_data = {
                "title": title,
                "outline": outline
            }
            
            # Validate the output
            if not self.validator.validate_output(json_data):
                logger.warning("Generated JSON failed validation, using error handler")
                return self.error_handler.handle_validation_error()
            
            return json_data
            
        except Exception as e:
            logger.error(f"Error building JSON: {e}")
            return self.error_handler.handle_errors(e)
    
    def write_json(self, json_data: Dict[str, Any], output_path: str) -> None:
        """
        Write JSON data to file.
        
        Args:
            json_data: Dictionary to write as JSON
            output_path: Path to output file
        """
        try:
            output_file = Path(output_path)
            
            # Ensure output directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write JSON with proper formatting
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(
                    json_data, 
                    f, 
                    indent=self.config.get('indent', 2),
                    ensure_ascii=self.config.get('ensure_ascii', False),
                    sort_keys=self.config.get('sort_keys', False)
                )
            
            logger.info(f"JSON output written to {output_path}")
            
        except Exception as e:
            logger.error(f"Error writing JSON to {output_path}: {e}")
            # Try to write error JSON instead
            error_json = self.error_handler.handle_write_error(e)
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(error_json, f, indent=2)
            except Exception as write_error:
                logger.critical(f"Failed to write error JSON: {write_error}")
                raise
    
    def process_and_write(self, structure: DocumentStructure, output_path: str) -> Dict[str, Any]:
        """
        Process document structure and write JSON output in one step.
        
        Args:
            structure: DocumentStructure to process
            output_path: Path to output file
            
        Returns:
            Generated JSON data
        """
        json_data = self.build_json(structure)
        self.write_json(json_data, output_path)
        return json_data


class OutputValidator:
    """Validates output format against specification."""
    
    def validate_output(self, json_data: Dict[str, Any]) -> bool:
        """
        Validate JSON output format.
        
        Args:
            json_data: Dictionary to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required top-level keys
            if not isinstance(json_data, dict):
                logger.error("JSON data is not a dictionary")
                return False
            
            required_keys = {"title", "outline"}
            if not required_keys.issubset(json_data.keys()):
                missing_keys = required_keys - json_data.keys()
                logger.error(f"Missing required keys: {missing_keys}")
                return False
            
            # Validate title
            if not isinstance(json_data["title"], str):
                logger.error("Title must be a string")
                return False
            
            # Validate outline
            outline = json_data["outline"]
            if not isinstance(outline, list):
                logger.error("Outline must be a list")
                return False
            
            # Validate each outline entry
            for i, entry in enumerate(outline):
                if not self._validate_outline_entry(entry, i):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            return False
    
    def _validate_outline_entry(self, entry: Dict[str, Any], index: int) -> bool:
        """
        Validate a single outline entry.
        
        Args:
            entry: Outline entry to validate
            index: Index of entry for error reporting
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(entry, dict):
            logger.error(f"Outline entry {index} is not a dictionary")
            return False
        
        # Check required keys
        required_keys = {"level", "text", "page"}
        if not required_keys.issubset(entry.keys()):
            missing_keys = required_keys - entry.keys()
            logger.error(f"Outline entry {index} missing keys: {missing_keys}")
            return False
        
        # Validate level
        valid_levels = {"H1", "H2", "H3"}
        if entry["level"] not in valid_levels:
            logger.error(f"Outline entry {index} has invalid level: {entry['level']}")
            return False
        
        # Validate text
        if not isinstance(entry["text"], str) or not entry["text"].strip():
            logger.error(f"Outline entry {index} has invalid text")
            return False
        
        # Validate page
        if not isinstance(entry["page"], int) or entry["page"] < 1:
            logger.error(f"Outline entry {index} has invalid page number: {entry['page']}")
            return False
        
        return True


class ErrorHandler:
    """Handles error cases in JSON generation."""
    
    def handle_errors(self, error: Exception) -> Dict[str, Any]:
        """
        Handle errors and generate fallback JSON output.
        
        Args:
            error: Exception that occurred
            
        Returns:
            Error JSON response
        """
        logger.error(f"Handling error: {error}")
        
        return {
            "title": "Error: Processing Failed",
            "outline": [],
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "details": "An error occurred during document processing"
            }
        }
    
    def handle_validation_error(self) -> Dict[str, Any]:
        """
        Handle validation errors.
        
        Returns:
            Error JSON response for validation failures
        """
        return {
            "title": "Error: Invalid Output Format",
            "outline": [],
            "error": {
                "type": "ValidationError",
                "message": "Generated output failed format validation",
                "details": "The processed document structure could not be formatted correctly"
            }
        }
    
    def handle_write_error(self, error: Exception) -> Dict[str, Any]:
        """
        Handle file writing errors.
        
        Args:
            error: Exception that occurred during writing
            
        Returns:
            Error JSON response for write failures
        """
        return {
            "title": "Error: File Write Failed",
            "outline": [],
            "error": {
                "type": "WriteError",
                "message": str(error),
                "details": "Failed to write output file"
            }
        }
    
    def handle_no_title_found(self) -> Dict[str, Any]:
        """
        Handle case where no title is detected.
        
        Returns:
            JSON response with default title
        """
        return {
            "title": "Untitled Document",
            "outline": [],
            "error": {
                "type": "NoTitleFound",
                "message": "No document title could be detected",
                "details": "Using default title"
            }
        }
    
    def handle_no_headings_detected(self) -> Dict[str, Any]:
        """
        Handle case where no headings are detected.
        
        Returns:
            JSON response with empty outline
        """
        return {
            "title": "Document",
            "outline": [],
            "error": {
                "type": "NoHeadingsDetected",
                "message": "No headings could be detected in the document",
                "details": "The document may not contain structured headings"
            }
        }