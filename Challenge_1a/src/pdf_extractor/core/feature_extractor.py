"""
Feature extraction module for generating classification features from processed text blocks.

This module implements comprehensive feature extraction for heading classification,
including font analysis, position analysis, and content analysis.
"""

import re
import math
import logging
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter, defaultdict

try:
    from ..models.models import TextBlock, ProcessedBlock, FeatureVector
    from ..config.config import config
except ImportError:
    # Handle relative imports when running as script
    from src.pdf_extractor.models.models import TextBlock, ProcessedBlock, FeatureVector
    from src.pdf_extractor.config.config import config


logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Main feature extraction logic that generates classification features from processed text blocks."""
    
    def __init__(self):
        self.config = config.get_feature_config()
        self.font_analyzer = FontAnalyzer(self.config)
        self.position_analyzer = PositionAnalyzer(self.config)
        self.content_analyzer = ContentAnalyzer(self.config)
        
        # Document-level statistics for normalization
        self.doc_stats = {}
        self.page_dimensions = {}
    
    def extract_features(self, block: ProcessedBlock) -> FeatureVector:
        """
        Extract comprehensive features from a processed text block.
        
        Args:
            block: ProcessedBlock object to extract features from
            
        Returns:
            FeatureVector with all extracted features
        """
        # Get font features
        font_features = self.font_analyzer.analyze_font_characteristics(block)
        
        # Get position features
        position_features = self.position_analyzer.analyze_position(block)
        
        # Get content features
        content_features = self.content_analyzer.analyze_content(block)
        
        # Combine all features into a comprehensive feature vector
        feature_vector = FeatureVector(
            font_size_ratio=font_features.get('font_size_ratio', 1.0),
            is_bold=font_features.get('is_bold', False),
            is_italic=font_features.get('is_italic', False),
            position_x=position_features.get('normalized_x', 0.0),
            position_y=position_features.get('normalized_y', 0.0),
            text_length=content_features.get('text_length', 0),
            capitalization_score=content_features.get('capitalization_score', 0.0),
            whitespace_ratio=content_features.get('whitespace_ratio', 0.0)
        )
        
        # Add extended features as attributes for advanced classification
        feature_vector.font_weight_score = font_features.get('font_weight_score', 0.0)
        feature_vector.font_style_score = font_features.get('font_style_score', 0.0)
        feature_vector.relative_font_size = font_features.get('relative_font_size', 1.0)
        
        feature_vector.page_position_score = position_features.get('page_position_score', 0.0)
        feature_vector.alignment_score = position_features.get('alignment_score', 0.0)
        feature_vector.whitespace_above = position_features.get('whitespace_above', 0.0)
        feature_vector.whitespace_below = position_features.get('whitespace_below', 0.0)
        feature_vector.indentation_level = position_features.get('indentation_level', 0.0)
        
        feature_vector.word_count = content_features.get('word_count', 0)
        feature_vector.sentence_count = content_features.get('sentence_count', 0)
        feature_vector.punctuation_density = content_features.get('punctuation_density', 0.0)
        feature_vector.numeric_content_ratio = content_features.get('numeric_content_ratio', 0.0)
        feature_vector.special_char_ratio = content_features.get('special_char_ratio', 0.0)
        feature_vector.title_case_score = content_features.get('title_case_score', 0.0)
        feature_vector.all_caps_score = content_features.get('all_caps_score', 0.0)
        
        return feature_vector
    
    def initialize_document_stats(self, blocks: List[ProcessedBlock]):
        """
        Initialize document-level statistics for feature normalization with memory optimization.
        
        Args:
            blocks: List of all ProcessedBlock objects in the document
        """
        if not blocks:
            return
        
        logger.info(f"Initializing document statistics from {len(blocks)} blocks")
        
        # Calculate font statistics efficiently
        font_sizes = []
        font_names = []
        
        # Process in batches to manage memory
        batch_size = 500
        for i in range(0, len(blocks), batch_size):
            batch = blocks[i:i + batch_size]
            batch_font_sizes = [block.original_block.font_size for block in batch if block.original_block.font_size > 0]
            batch_font_names = [block.original_block.font_name for block in batch]
            
            font_sizes.extend(batch_font_sizes)
            font_names.extend(batch_font_names)
        
        self.doc_stats = {
            'avg_font_size': sum(font_sizes) / len(font_sizes) if font_sizes else 12.0,
            'max_font_size': max(font_sizes) if font_sizes else 12.0,
            'min_font_size': min(font_sizes) if font_sizes else 12.0,
            'median_font_size': sorted(font_sizes)[len(font_sizes)//2] if font_sizes else 12.0,
            'font_size_std': self._calculate_std(font_sizes) if len(font_sizes) > 1 else 0.0,
            'common_fonts': Counter(font_names).most_common(5),
            'total_blocks': len(blocks),
            'total_pages': max(block.page_number for block in blocks) if blocks else 1
        }
        
        # Clear temporary lists to free memory
        del font_sizes, font_names
        
        # Calculate page dimensions for position normalization
        self._calculate_page_dimensions(blocks)
        
        # Update analyzers with document statistics
        self.font_analyzer.set_document_stats(self.doc_stats)
        self.position_analyzer.set_document_stats(self.doc_stats, self.page_dimensions)
        self.content_analyzer.set_document_stats(self.doc_stats)
        
        logger.debug(f"Document stats: avg_font={self.doc_stats['avg_font_size']:.1f}, "
                    f"pages={self.doc_stats['total_pages']}, blocks={self.doc_stats['total_blocks']}")
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation of a list of values."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)
    
    def _calculate_page_dimensions(self, blocks: List[ProcessedBlock]):
        """Calculate page dimensions from text block positions."""
        page_bounds = defaultdict(lambda: {'min_x': float('inf'), 'max_x': 0, 
                                          'min_y': float('inf'), 'max_y': 0})
        
        for block in blocks:
            page_num = block.page_number
            bbox = block.original_block.bbox
            
            page_bounds[page_num]['min_x'] = min(page_bounds[page_num]['min_x'], bbox[0])
            page_bounds[page_num]['max_x'] = max(page_bounds[page_num]['max_x'], bbox[2])
            page_bounds[page_num]['min_y'] = min(page_bounds[page_num]['min_y'], bbox[1])
            page_bounds[page_num]['max_y'] = max(page_bounds[page_num]['max_y'], bbox[3])
        
        # Convert to page dimensions
        for page_num, bounds in page_bounds.items():
            if bounds['min_x'] != float('inf'):
                self.page_dimensions[page_num] = {
                    'width': bounds['max_x'] - bounds['min_x'],
                    'height': bounds['max_y'] - bounds['min_y'],
                    'left_margin': bounds['min_x'],
                    'top_margin': bounds['min_y']
                }


class FontAnalyzer:
    """Analyzes font characteristics for heading classification."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.doc_stats = {}
    
    def set_document_stats(self, doc_stats: Dict):
        """Set document-level statistics for normalization."""
        self.doc_stats = doc_stats
    
    def analyze_font_characteristics(self, block: ProcessedBlock) -> Dict[str, float]:
        """
        Analyze font characteristics of a text block.
        
        Args:
            block: ProcessedBlock to analyze
            
        Returns:
            Dictionary of font-related features
        """
        original_block = block.original_block
        
        # Basic font features
        font_size = original_block.font_size
        font_flags = original_block.font_flags
        font_name = original_block.font_name
        
        # Calculate font size ratio relative to document average
        avg_font_size = self.doc_stats.get('avg_font_size', 12.0)
        font_size_ratio = font_size / avg_font_size if avg_font_size > 0 else 1.0
        
        # Calculate relative font size (percentile in document)
        max_font_size = self.doc_stats.get('max_font_size', font_size)
        min_font_size = self.doc_stats.get('min_font_size', font_size)
        if max_font_size > min_font_size:
            relative_font_size = (font_size - min_font_size) / (max_font_size - min_font_size)
        else:
            relative_font_size = 0.5
        
        # Font style analysis
        is_bold = self._is_bold_font(font_flags)
        is_italic = self._is_italic_font(font_flags)
        
        # Font weight score (combination of size and style)
        font_weight_score = self._calculate_font_weight_score(font_size_ratio, is_bold)
        
        # Font style score (italic, decorative fonts, etc.)
        font_style_score = self._calculate_font_style_score(font_name, is_italic)
        
        return {
            'font_size_ratio': font_size_ratio,
            'relative_font_size': relative_font_size,
            'is_bold': is_bold,
            'is_italic': is_italic,
            'font_weight_score': font_weight_score,
            'font_style_score': font_style_score
        }
    
    def _is_bold_font(self, font_flags: int) -> bool:
        """Check if font is bold based on flags."""
        return bool(font_flags & 2**4)  # Bold flag
    
    def _is_italic_font(self, font_flags: int) -> bool:
        """Check if font is italic based on flags."""
        return bool(font_flags & 2**1)  # Italic flag
    
    def _calculate_font_weight_score(self, font_size_ratio: float, is_bold: bool) -> float:
        """Calculate a composite font weight score."""
        weight_score = font_size_ratio
        if is_bold:
            weight_score *= 1.5  # Boost for bold text
        return min(weight_score, 3.0)  # Cap at 3.0
    
    def _calculate_font_style_score(self, font_name: str, is_italic: bool) -> float:
        """Calculate font style score based on font name and style."""
        style_score = 0.0
        
        # Check for decorative or heading fonts
        heading_font_indicators = ['heading', 'title', 'header', 'bold', 'black', 'heavy']
        font_name_lower = font_name.lower()
        
        for indicator in heading_font_indicators:
            if indicator in font_name_lower:
                style_score += 0.3
        
        # Italic text might be less likely to be a heading
        if is_italic:
            style_score -= 0.1
        
        return max(0.0, min(style_score, 1.0))


class PositionAnalyzer:
    """Analyzes spatial positioning and layout characteristics."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.doc_stats = {}
        self.page_dimensions = {}
    
    def set_document_stats(self, doc_stats: Dict, page_dimensions: Dict):
        """Set document-level statistics and page dimensions."""
        self.doc_stats = doc_stats
        self.page_dimensions = page_dimensions
    
    def analyze_position(self, block: ProcessedBlock) -> Dict[str, float]:
        """
        Analyze position and layout characteristics of a text block.
        
        Args:
            block: ProcessedBlock to analyze
            
        Returns:
            Dictionary of position-related features
        """
        original_block = block.original_block
        bbox = original_block.bbox
        page_num = original_block.page_number
        
        # Get page dimensions for normalization
        page_dims = self.page_dimensions.get(page_num, {'width': 612, 'height': 792, 'left_margin': 0, 'top_margin': 0})
        
        # Normalize position coordinates
        normalized_x = (bbox[0] - page_dims['left_margin']) / page_dims['width'] if page_dims['width'] > 0 else 0.0
        normalized_y = (bbox[1] - page_dims['top_margin']) / page_dims['height'] if page_dims['height'] > 0 else 0.0
        
        # Calculate position-based features
        page_position_score = self._calculate_page_position_score(normalized_x, normalized_y)
        alignment_score = self._calculate_alignment_score(normalized_x, bbox)
        indentation_level = self._calculate_indentation_level(normalized_x)
        
        # Whitespace analysis (simplified - would need more context for full implementation)
        whitespace_above = self._estimate_whitespace_above(bbox, page_dims)
        whitespace_below = self._estimate_whitespace_below(bbox, page_dims)
        
        return {
            'normalized_x': normalized_x,
            'normalized_y': normalized_y,
            'page_position_score': page_position_score,
            'alignment_score': alignment_score,
            'indentation_level': indentation_level,
            'whitespace_above': whitespace_above,
            'whitespace_below': whitespace_below
        }
    
    def _calculate_page_position_score(self, normalized_x: float, normalized_y: float) -> float:
        """Calculate score based on position on page (top positions score higher for headings)."""
        # Top of page scores higher
        y_score = 1.0 - normalized_y
        
        # Left alignment scores higher for headings
        x_score = 1.0 - abs(normalized_x - 0.0)  # Distance from left edge
        
        # Combine scores with emphasis on vertical position
        return 0.7 * y_score + 0.3 * x_score
    
    def _calculate_alignment_score(self, normalized_x: float, bbox: Tuple[float, float, float, float]) -> float:
        """Calculate alignment score (left-aligned text scores higher for headings)."""
        # Left alignment (close to 0) scores higher
        left_alignment_score = 1.0 - min(normalized_x, 1.0)
        
        # Center alignment detection
        center_position = 0.5
        center_alignment_score = 1.0 - abs(normalized_x - center_position) * 2
        
        # Return the higher of left or center alignment scores
        return max(left_alignment_score, center_alignment_score * 0.8)  # Slight preference for left
    
    def _calculate_indentation_level(self, normalized_x: float) -> float:
        """Calculate indentation level (0 = no indent, higher = more indented)."""
        # Simple indentation calculation based on distance from left margin
        return normalized_x
    
    def _estimate_whitespace_above(self, bbox: Tuple[float, float, float, float], page_dims: Dict) -> float:
        """Estimate whitespace above the text block."""
        # Simplified estimation - in full implementation would compare with other blocks
        if page_dims['height'] <= 0:
            return 0.0
        relative_y = (bbox[1] - page_dims['top_margin']) / page_dims['height']
        return max(0.0, relative_y)  # Higher values = more space above
    
    def _estimate_whitespace_below(self, bbox: Tuple[float, float, float, float], page_dims: Dict) -> float:
        """Estimate whitespace below the text block."""
        # Simplified estimation - in full implementation would compare with other blocks
        if page_dims['height'] <= 0:
            return 0.0
        relative_y_bottom = (bbox[3] - page_dims['top_margin']) / page_dims['height']
        return max(0.0, 1.0 - relative_y_bottom)  # Higher values = more space below


class ContentAnalyzer:
    """Analyzes text content patterns for heading classification."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.doc_stats = {}
    
    def set_document_stats(self, doc_stats: Dict):
        """Set document-level statistics."""
        self.doc_stats = doc_stats
    
    def analyze_content(self, block: ProcessedBlock) -> Dict[str, float]:
        """
        Analyze text content characteristics of a text block.
        
        Args:
            block: ProcessedBlock to analyze
            
        Returns:
            Dictionary of content-related features
        """
        text = block.text.strip()
        
        # Basic text statistics
        text_length = len(text)
        word_count = len(text.split()) if text else 0
        sentence_count = self._count_sentences(text)
        
        # Character analysis
        capitalization_score = self._calculate_capitalization_score(text)
        whitespace_ratio = self._calculate_whitespace_ratio(text)
        punctuation_density = self._calculate_punctuation_density(text)
        numeric_content_ratio = self._calculate_numeric_content_ratio(text)
        special_char_ratio = self._calculate_special_char_ratio(text)
        
        # Advanced text analysis
        title_case_score = self._calculate_title_case_score(text)
        all_caps_score = self._calculate_all_caps_score(text)
        
        return {
            'text_length': text_length,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'capitalization_score': capitalization_score,
            'whitespace_ratio': whitespace_ratio,
            'punctuation_density': punctuation_density,
            'numeric_content_ratio': numeric_content_ratio,
            'special_char_ratio': special_char_ratio,
            'title_case_score': title_case_score,
            'all_caps_score': all_caps_score
        }
    
    def _count_sentences(self, text: str) -> int:
        """Count the number of sentences in text."""
        if not text:
            return 0
        
        # Simple sentence counting based on sentence-ending punctuation
        sentence_endings = re.findall(r'[.!?]+', text)
        return len(sentence_endings)
    
    def _calculate_capitalization_score(self, text: str) -> float:
        """Calculate the ratio of uppercase characters in text with multilingual support."""
        if not text:
            return 0.0
        
        # Get all alphabetic characters, including non-Latin scripts
        letters = []
        for char in text:
            if char.isalpha():
                # Check if character has case (some scripts don't have upper/lower case)
                if char.upper() != char.lower():
                    letters.append(char)
        
        if not letters:
            return 0.0
        
        uppercase_count = sum(1 for c in letters if c.isupper())
        return uppercase_count / len(letters)
    
    def _calculate_whitespace_ratio(self, text: str) -> float:
        """Calculate the ratio of whitespace characters in text."""
        if not text:
            return 0.0
        
        whitespace_count = sum(1 for c in text if c.isspace())
        return whitespace_count / len(text)
    
    def _calculate_punctuation_density(self, text: str) -> float:
        """Calculate the density of punctuation characters."""
        if not text:
            return 0.0
        
        punctuation_chars = '.,;:!?()[]{}"\'-'
        punctuation_count = sum(1 for c in text if c in punctuation_chars)
        return punctuation_count / len(text)
    
    def _calculate_numeric_content_ratio(self, text: str) -> float:
        """Calculate the ratio of numeric characters in text."""
        if not text:
            return 0.0
        
        numeric_count = sum(1 for c in text if c.isdigit())
        return numeric_count / len(text)
    
    def _calculate_special_char_ratio(self, text: str) -> float:
        """Calculate the ratio of special characters (non-alphanumeric, non-whitespace)."""
        if not text:
            return 0.0
        
        special_count = sum(1 for c in text if not (c.isalnum() or c.isspace()))
        return special_count / len(text)
    
    def _calculate_title_case_score(self, text: str) -> float:
        """Calculate how well text matches title case pattern."""
        if not text:
            return 0.0
        
        words = text.split()
        if not words:
            return 0.0
        
        # First check if it's all caps - if so, it's not title case
        if all(word.isupper() for word in words if word.isalpha()):
            return 0.0
        
        # Check if each word starts with uppercase (ignoring common articles/prepositions)
        title_case_words = 0
        ignore_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        for i, word in enumerate(words):
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if i == 0 or clean_word not in ignore_words:  # First word or not an ignore word
                if word and word[0].isupper() and not word.isupper():  # First letter upper, but not all caps
                    title_case_words += 1
            else:  # Ignore word - should be lowercase
                if word and word[0].islower():
                    title_case_words += 1
        
        return title_case_words / len(words)
    
    def _calculate_all_caps_score(self, text: str) -> float:
        """Calculate how much of the text is in all caps."""
        if not text:
            return 0.0
        
        words = text.split()
        if not words:
            return 0.0
        
        all_caps_words = sum(1 for word in words if word.isupper() and len(word) > 1)
        return all_caps_words / len(words)