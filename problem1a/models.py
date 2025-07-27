"""
Core data models for the PDF Structure Extractor.

This module defines the data structures used throughout the PDF processing pipeline,
from raw text extraction to final JSON output generation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class TextBlock:
    """Represents a text block extracted from a PDF with formatting information."""
    text: str
    page_number: int
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    font_size: float
    font_name: str
    font_flags: int  # Font style flags (bold, italic, etc.)


@dataclass
class ProcessedBlock:
    """Represents a text block after preprocessing with extracted features."""
    text: str
    page_number: int
    features: 'FeatureVector'
    original_block: TextBlock


@dataclass
class FeatureVector:
    """Feature vector for heading classification."""
    font_size_ratio: float  # Relative to document average
    is_bold: bool
    is_italic: bool
    position_x: float  # Normalized position on page
    position_y: float  # Normalized position on page
    text_length: int
    capitalization_score: float  # Ratio of uppercase characters
    whitespace_ratio: float  # Ratio of whitespace to total characters


@dataclass
class ClassificationResult:
    """Result of heading classification for a text block."""
    block: ProcessedBlock
    predicted_class: str  # 'title', 'h1', 'h2', 'h3', 'text'
    confidence: float


@dataclass
class Heading:
    """Represents a detected heading with its hierarchical level."""
    level: str  # 'title', 'H1', 'H2', 'H3'
    text: str
    page: int
    confidence: float


@dataclass
class DocumentStructure:
    """Complete document structure with title and hierarchical headings."""
    title: Optional[str]
    headings: List[Heading]
    metadata: Dict[str, Any]