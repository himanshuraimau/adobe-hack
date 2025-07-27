"""
Feature extraction module for generating classification features from text blocks.
"""

import re
import math
import logging
from typing import List, Dict, Tuple, Optional
from collections import Counter

try:
    from .models import ProcessedBlock, FeatureVector, TextBlock
    from .config import config
except ImportError:
    from models import ProcessedBlock, FeatureVector, TextBlock
    from config import config

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Main feature extraction logic for heading classification."""
    
    def __init__(self):
        self.config = config.get_feature_config()
        self.font_analyzer = FontAnalyzer(self.config)
        self.position_analyzer = PositionAnalyzer(self.config)
        self.content_analyzer = ContentAnalyzer(self.config)
        self.document_stats = {}
    
    def extract_features(self, block: ProcessedBlock) -> FeatureVector:
        """Extract classification features from a processed text block."""
        font_features = self.font_analyzer.analyze_font_characteristics(block)
        position_features = self.position_analyzer.analyze_position(block)
        content_features = self.content_analyzer.analyze_content(block)
        
        feature_vector = FeatureVector(
            font_size_ratio=font_features['font_size_ratio'],
            is_bold=font_features['is_bold'],
            is_italic=font_features['is_italic'],
            position_x=position_features['normalized_x'],
            position_y=position_features['normalized_y'],
            text_length=content_features['text_length'],
            capitalization_score=content_features['capitalization_score'],
            whitespace_ratio=content_features['whitespace_ratio']
        )
        
        # Add additional features
        feature_vector.font_weight_score = font_features.get('font_weight_score', 0.0)
        feature_vector.alignment_score = position_features.get('alignment_score', 0.0)
        feature_vector.page_position_score = position_features.get('page_position_score', 0.0)
        feature_vector.punctuation_score = content_features.get('punctuation_score', 0.0)
        feature_vector.word_count = content_features.get('word_count', 0)
        feature_vector.numeric_ratio = content_features.get('numeric_ratio', 0.0)
        feature_vector.special_char_ratio = content_features.get('special_char_ratio', 0.0)
        feature_vector.heading_pattern_score = content_features.get('heading_pattern_score', 0.0)
        feature_vector.length_score = content_features.get('length_score', 0.0)
        
        return feature_vector
    
    def extract_features_batch(self, blocks: List[ProcessedBlock]) -> List[FeatureVector]:
        """Extract features from multiple blocks efficiently."""
        if not blocks:
            return []
        
        self.set_document_stats(blocks)
        
        feature_vectors = []
        for block in blocks:
            features = self.extract_features(block)
            feature_vectors.append(features)
        
        logger.info(f"Extracted features for {len(blocks)} blocks")
        return feature_vectors
    
    def set_document_stats(self, blocks: List[ProcessedBlock]):
        """Set document-level statistics for feature normalization."""
        self.document_stats = self._calculate_document_stats(blocks)
        self.font_analyzer.set_document_stats(self.document_stats)
        self.position_analyzer.set_document_stats(self.document_stats)
        self.content_analyzer.set_document_stats(self.document_stats)
    
    def _calculate_document_stats(self, blocks: List[ProcessedBlock]) -> Dict:
        """Calculate document-level statistics for normalization."""
        if not blocks:
            return {}
        
        font_sizes = []
        page_numbers = []
        text_lengths = []
        x_positions = []
        y_positions = []
        
        for block in blocks:
            original = block.original_block
            font_sizes.append(original.font_size)
            page_numbers.append(original.page_number)
            text_lengths.append(len(block.text))
            
            if original.bbox:
                x_positions.append(original.bbox[0])
                y_positions.append(original.bbox[1])
        
        stats = {
            'avg_font_size': sum(font_sizes) / len(font_sizes) if font_sizes else 12.0,
            'max_font_size': max(font_sizes) if font_sizes else 12.0,
            'min_font_size': min(font_sizes) if font_sizes else 12.0,
            'median_font_size': sorted(font_sizes)[len(font_sizes)//2] if font_sizes else 12.0,
            'total_pages': max(page_numbers) if page_numbers else 1,
            'total_blocks': len(blocks),
            'avg_text_length': sum(text_lengths) / len(text_lengths) if text_lengths else 0,
            'max_text_length': max(text_lengths) if text_lengths else 0,
            'page_width': max(x_positions) - min(x_positions) if x_positions else 612,
            'page_height': max(y_positions) - min(y_positions) if y_positions else 792
        }
        
        return stats


class FontAnalyzer:
    """Analyzes font characteristics for heading detection."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.document_stats = {}
    
    def set_document_stats(self, stats: Dict):
        """Set document-level statistics for normalization."""
        self.document_stats = stats
    
    def analyze_font_characteristics(self, block: ProcessedBlock) -> Dict:
        """Analyze font characteristics of a text block."""
        original = block.original_block
        
        font_size_ratio = self._calculate_font_size_ratio(original.font_size)
        is_bold = self._is_bold_font(original.font_flags)
        is_italic = self._is_italic_font(original.font_flags)
        
        font_weight_score = self._calculate_font_weight_score(original)
        font_style_score = self._calculate_font_style_score(original)
        relative_size_score = self._calculate_relative_size_score(original.font_size)
        
        return {
            'font_size_ratio': font_size_ratio,
            'is_bold': is_bold,
            'is_italic': is_italic,
            'font_weight_score': font_weight_score,
            'font_style_score': font_style_score,
            'relative_size_score': relative_size_score
        }
    
    def _calculate_font_size_ratio(self, font_size: float) -> float:
        """Calculate font size ratio relative to document average."""
        avg_font_size = self.document_stats.get('avg_font_size', 12.0)
        return font_size / avg_font_size if avg_font_size > 0 else 1.0
    
    def _is_bold_font(self, font_flags: int) -> bool:
        """Check if font is bold based on flags."""
        return bool(font_flags & 2**4)
    
    def _is_italic_font(self, font_flags: int) -> bool:
        """Check if font is italic based on flags."""
        return bool(font_flags & 2**1)
    
    def _calculate_font_weight_score(self, block: TextBlock) -> float:
        """Calculate a comprehensive font weight score."""
        score = 0.0
        
        font_size_ratio = self._calculate_font_size_ratio(block.font_size)
        score += min(font_size_ratio * 0.5, 1.0)
        
        if self._is_bold_font(block.font_flags):
            score += 0.3
        
        font_name = block.font_name.lower()
        if any(weight in font_name for weight in ['bold', 'heavy', 'black', 'extra']):
            score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_font_style_score(self, block: TextBlock) -> float:
        """Calculate font style distinctiveness score."""
        score = 0.0
        
        if self._is_italic_font(block.font_flags):
            score += 0.2
        
        font_name = block.font_name.lower()
        if any(style in font_name for style in ['serif', 'sans', 'mono', 'script']):
            score += 0.1
        
        return score
    
    def _calculate_relative_size_score(self, font_size: float) -> float:
        """Calculate how much larger this font is compared to others."""
        max_font_size = self.document_stats.get('max_font_size', font_size)
        min_font_size = self.document_stats.get('min_font_size', font_size)
        
        if max_font_size == min_font_size:
            return 0.5
        
        return (font_size - min_font_size) / (max_font_size - min_font_size)


class PositionAnalyzer:
    """Analyzes spatial positioning for heading detection."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.document_stats = {}
    
    def set_document_stats(self, stats: Dict):
        """Set document-level statistics for normalization."""
        self.document_stats = stats
    
    def analyze_position(self, block: ProcessedBlock) -> Dict:
        """Analyze spatial position of a text block."""
        original = block.original_block
        bbox = original.bbox
        
        if not bbox:
            return {
                'normalized_x': 0.0,
                'normalized_y': 0.0,
                'alignment_score': 0.0,
                'page_position_score': 0.0,
                'whitespace_above': 0.0,
                'whitespace_below': 0.0
            }
        
        normalized_x, normalized_y = self._normalize_position(bbox)
        alignment_score = self._calculate_alignment_score(bbox)
        page_position_score = self._calculate_page_position_score(bbox, original.page_number)
        whitespace_analysis = self._analyze_whitespace_patterns(block)
        
        return {
            'normalized_x': normalized_x,
            'normalized_y': normalized_y,
            'alignment_score': alignment_score,
            'page_position_score': page_position_score,
            'whitespace_above': whitespace_analysis.get('above', 0.0),
            'whitespace_below': whitespace_analysis.get('below', 0.0)
        }
    
    def _normalize_position(self, bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """Normalize position coordinates to [0, 1] range."""
        page_width = self.document_stats.get('page_width', 612)
        page_height = self.document_stats.get('page_height', 792)
        
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        normalized_x = center_x / page_width if page_width > 0 else 0.0
        normalized_y = center_y / page_height if page_height > 0 else 0.0
        
        return normalized_x, normalized_y
    
    def _calculate_alignment_score(self, bbox: Tuple[float, float, float, float]) -> float:
        """Calculate alignment score (left, center, right alignment)."""
        page_width = self.document_stats.get('page_width', 612)
        
        left_x = bbox[0]
        right_x = bbox[2]
        center_x = (left_x + right_x) / 2
        
        left_margin = left_x
        right_margin = page_width - right_x
        center_distance = abs(center_x - page_width / 2)
        
        alignment_score = 0.0
        
        if left_margin < page_width * 0.1:
            alignment_score += 0.3
        
        if center_distance < page_width * 0.1:
            alignment_score += 0.4
        
        margin_diff = abs(left_margin - right_margin)
        if margin_diff < page_width * 0.05:
            alignment_score += 0.3
        
        return min(alignment_score, 1.0)
    
    def _calculate_page_position_score(self, bbox: Tuple[float, float, float, float], page_num: int) -> float:
        """Calculate score based on position within page and document."""
        page_height = self.document_stats.get('page_height', 792)
        total_pages = self.document_stats.get('total_pages', 1)
        
        y_position = bbox[1]
        top_score = max(0, 1 - (y_position / page_height)) if page_height > 0 else 0.5
        
        page_score = max(0, 1 - ((page_num - 1) / total_pages)) if total_pages > 1 else 1.0
        
        return (top_score * 0.6 + page_score * 0.4)
    
    def _analyze_whitespace_patterns(self, block: ProcessedBlock) -> Dict:
        """Analyze whitespace patterns around the text block."""
        text = block.text
        
        leading_whitespace = len(text) - len(text.lstrip())
        trailing_whitespace = len(text) - len(text.rstrip())
        total_length = len(text) if text else 1
        
        return {
            'above': leading_whitespace / total_length,
            'below': trailing_whitespace / total_length
        }


class ContentAnalyzer:
    """Analyzes text content patterns for heading detection."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.document_stats = {}
        
        # Common heading patterns
        self.heading_patterns = [
            r'^\d+\.?\s+',  # Numbered headings
            r'^[A-Z][A-Z\s]+$',  # ALL CAPS
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$',  # Title Case
            r'^\w+\s*:',  # Colon endings
            r'^(?:Chapter|Section|Part)\s+\d+',  # Chapter/Section markers
        ]
    
    def set_document_stats(self, stats: Dict):
        """Set document-level statistics for normalization."""
        self.document_stats = stats
    
    def analyze_content(self, block: ProcessedBlock) -> Dict:
        """Analyze content patterns of a text block."""
        text = block.text.strip()
        
        text_length = len(text)
        word_count = len(text.split()) if text else 0
        capitalization_score = self._calculate_capitalization_score(text)
        whitespace_ratio = self._calculate_whitespace_ratio(text)
        
        punctuation_score = self._calculate_punctuation_score(text)
        numeric_ratio = self._calculate_numeric_ratio(text)
        special_char_ratio = self._calculate_special_char_ratio(text)
        heading_pattern_score = self._calculate_heading_pattern_score(text)
        length_score = self._calculate_length_score(text_length)
        
        return {
            'text_length': text_length,
            'word_count': word_count,
            'capitalization_score': capitalization_score,
            'whitespace_ratio': whitespace_ratio,
            'punctuation_score': punctuation_score,
            'numeric_ratio': numeric_ratio,
            'special_char_ratio': special_char_ratio,
            'heading_pattern_score': heading_pattern_score,
            'length_score': length_score
        }
    
    def _calculate_capitalization_score(self, text: str) -> float:
        """Calculate the ratio of uppercase characters in text."""
        if not text:
            return 0.0
        
        letters = [c for c in text if c.isalpha()]
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
    
    def _calculate_punctuation_score(self, text: str) -> float:
        """Calculate punctuation patterns that indicate headings."""
        if not text:
            return 0.0
        
        score = 0.0
        
        if text.endswith(':'):
            score += 0.4
        elif text.endswith('.'):
            score -= 0.2
        elif not re.search(r'[.!?]$', text):
            score += 0.2
        
        punctuation_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
        punctuation_ratio = punctuation_count / len(text) if text else 0
        
        if punctuation_ratio < 0.1:
            score += 0.3
        elif punctuation_ratio > 0.3:
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _calculate_numeric_ratio(self, text: str) -> float:
        """Calculate the ratio of numeric characters in text."""
        if not text:
            return 0.0
        
        numeric_count = sum(1 for c in text if c.isdigit())
        return numeric_count / len(text)
    
    def _calculate_special_char_ratio(self, text: str) -> float:
        """Calculate the ratio of special characters in text."""
        if not text:
            return 0.0
        
        special_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
        return special_count / len(text)
    
    def _calculate_heading_pattern_score(self, text: str) -> float:
        """Calculate score based on common heading patterns."""
        if not text:
            return 0.0
        
        score = 0.0
        
        for pattern in self.heading_patterns:
            if re.match(pattern, text):
                score += 0.2
        
        if len(text.split()) <= 6 and text[0].isupper():
            score += 0.1
        
        if re.match(r'^\d+', text):
            score += 0.15
        
        words = text.split()
        if len(words) > 1 and all(word[0].isupper() if word else False for word in words):
            score += 0.1
        
        if text.isupper() and len(text) < 50:
            score += 0.2
        
        return min(1.0, score)
    
    def _calculate_length_score(self, text_length: int) -> float:
        """Calculate score based on text length (headings are typically shorter)."""
        avg_length = self.document_stats.get('avg_text_length', 100)
        
        if text_length < avg_length * 0.5:
            return 0.8
        elif text_length < avg_length:
            return 0.6
        elif text_length < avg_length * 1.5:
            return 0.4
        else:
            return 0.2