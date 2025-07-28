"""
Configuration management system for the PDF Structure Extractor.

This module provides centralized configuration for all components of the system,
including model paths, processing parameters, and output settings.
"""

import os
from pathlib import Path
from typing import Dict, Any


class Config:
    """Central configuration class for the PDF Structure Extractor."""
    
    def __init__(self):
        # Base paths - go up to project root from src/pdf_extractor/config/
        self.base_dir = Path(__file__).parent.parent.parent.parent
        self.input_dir = self.base_dir / "data" / "input"
        self.output_dir = self.base_dir / "data" / "output"
        self.models_dir = self.base_dir / "models"
        
        # Model configuration
        self.mobilebert_model_path = self.models_dir / "local_mobilebert"
        
        # Processing parameters
        self.max_pages = 50
        self.processing_timeout = 10  # seconds
        self.max_model_size_mb = 200
        
        # PDF parsing configuration
        self.pdf_config = {
            "extract_images": False,
            "extract_tables": False,
            "preserve_layout": True,
            "encoding": "utf-8"
        }
        
        # Text preprocessing configuration
        self.preprocessing_config = {
            "normalize_whitespace": True,
            "remove_empty_blocks": True,
            "min_text_length": 3,
            "preserve_formatting": True
        }
        
        # Feature extraction configuration
        self.feature_config = {
            "font_size_threshold": 1.2,  # Ratio for heading detection
            "position_weight": 0.3,
            "content_weight": 0.4,
            "format_weight": 0.3
        }
        
        # Classification configuration
        self.classification_config = {
            "confidence_threshold": 0.5,
            "use_fallback_rules": True,
            "max_sequence_length": 512
        }
        
        # Structure analysis configuration
        self.structure_config = {
            "title_detection_methods": ["first_heading", "largest_font", "metadata"],
            "hierarchy_tolerance": 0.1,
            "min_heading_confidence": 0.3
        }
        
        # Output configuration
        self.output_config = {
            "indent": 2,
            "ensure_ascii": False,
            "sort_keys": False
        }
        
        # Logging configuration
        self.logging_config = {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": None,  # Set to file path for file logging
            "max_file_size": 10 * 1024 * 1024,  # 10MB
            "backup_count": 5,
            "error_log_file": self.base_dir / "logs" / "errors.log",
            "debug_log_file": self.base_dir / "logs" / "debug.log"
        }
    
    def get_pdf_config(self) -> Dict[str, Any]:
        """Get PDF parsing configuration."""
        return self.pdf_config.copy()
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Get text preprocessing configuration."""
        return self.preprocessing_config.copy()
    
    def get_feature_config(self) -> Dict[str, Any]:
        """Get feature extraction configuration."""
        return self.feature_config.copy()
    
    def get_classification_config(self) -> Dict[str, Any]:
        """Get classification configuration."""
        return self.classification_config.copy()
    
    def get_structure_config(self) -> Dict[str, Any]:
        """Get structure analysis configuration."""
        return self.structure_config.copy()
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration."""
        return self.output_config.copy()
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.logging_config.copy()
    
    def ensure_directories(self):
        """Ensure all required directories exist."""
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)


# Global configuration instance
config = Config()