"""PDF Structure Extractor - Main Package

This package provides functionality to extract structured outlines from PDF documents
using machine learning and rule-based approaches.
"""

# Import models (no external dependencies)
from .models.models import (
    TextBlock,
    ProcessedBlock,
    FeatureVector,
    ClassificationResult,
    Heading,
    DocumentStructure
)

# Import config (no external dependencies)
from .config.config import Config

# Core modules with external dependencies - import on demand
def get_pdf_parser():
    from .core.pdf_parser import PDFParser
    return PDFParser

def get_text_preprocessor():
    from .core.preprocessor import TextPreprocessor
    return TextPreprocessor

def get_feature_extractor():
    from .core.feature_extractor import FeatureExtractor
    return FeatureExtractor

def get_heading_classifier():
    from .core.classifier import HeadingClassifier
    return HeadingClassifier

def get_structure_analyzer():
    from .core.structure_analyzer import StructureAnalyzer
    return StructureAnalyzer

def get_json_builder():
    from .core.json_builder import JSONBuilder
    return JSONBuilder

# For backward compatibility, try to import core modules
try:
    from .core.pdf_parser import PDFParser
    from .core.preprocessor import TextPreprocessor
    from .core.feature_extractor import FeatureExtractor
    from .core.classifier import HeadingClassifier
    from .core.structure_analyzer import StructureAnalyzer
    from .core.json_builder import JSONBuilder
    
    # If successful, add to __all__
    __all__ = [
        "TextBlock", "ProcessedBlock", "FeatureVector", "ClassificationResult", 
        "Heading", "DocumentStructure", "Config",
        "PDFParser", "TextPreprocessor", "FeatureExtractor", 
        "HeadingClassifier", "StructureAnalyzer", "JSONBuilder"
    ]
except ImportError:
    # If dependencies not available, only export models and config
    __all__ = [
        "TextBlock", "ProcessedBlock", "FeatureVector", "ClassificationResult", 
        "Heading", "DocumentStructure", "Config",
        "get_pdf_parser", "get_text_preprocessor", "get_feature_extractor",
        "get_heading_classifier", "get_structure_analyzer", "get_json_builder"
    ]

__version__ = "0.1.0"
__author__ = "Adobe PDF Processing Challenge"