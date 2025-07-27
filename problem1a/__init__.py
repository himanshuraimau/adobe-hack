"""
PDF Structure Extractor package.

This package provides functionality to extract structured outlines from PDF documents,
identifying titles and hierarchical headings with their corresponding page numbers.
"""

__version__ = "0.1.0"
__author__ = "PDF Structure Extractor"

from .models import (
    TextBlock,
    ProcessedBlock,
    FeatureVector,
    ClassificationResult,
    Heading,
    DocumentStructure
)

from .config import config

__all__ = [
    "TextBlock",
    "ProcessedBlock", 
    "FeatureVector",
    "ClassificationResult",
    "Heading",
    "DocumentStructure",
    "config"
]