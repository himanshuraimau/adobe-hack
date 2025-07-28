"""PDF Structure Extractor - Main Package

This package provides functionality to extract structured outlines from PDF documents
using machine learning and rule-based approaches.
"""

from .models.models import (
    TextBlock,
    ProcessedBlock,
    FeatureVector,
    ClassificationResult,
    Heading,
    DocumentStructure
)

from .core.pdf_parser import PDFParser
from .core.preprocessor import TextPreprocessor
from .core.feature_extractor import FeatureExtractor
from .core.classifier import HeadingClassifier
from .core.structure_analyzer import StructureAnalyzer
from .core.json_builder import JSONBuilder
from .config.config import Config

__version__ = "0.1.0"
__author__ = "Adobe PDF Processing Challenge"

__all__ = [
    "TextBlock",
    "ProcessedBlock", 
    "FeatureVector",
    "ClassificationResult",
    "Heading",
    "DocumentStructure",
    "PDFParser",
    "TextPreprocessor",
    "FeatureExtractor",
    "HeadingClassifier",
    "StructureAnalyzer",
    "JSONBuilder",
    "Config"
]