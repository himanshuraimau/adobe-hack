"""
Text preprocessing module for cleaning and normalizing extracted text.

This module handles text preprocessing while preserving structural formatting
information needed for heading detection and classification.
"""

import re
import unicodedata
from typing import List, Dict, Set, Tuple
import logging
from collections import defaultdict

try:
    from .models import TextBlock, ProcessedBlock, FeatureVector
    from .config import config
except ImportError:
    # Handle relative imports when running as script
    from models import TextBlock, ProcessedBlock, FeatureVector
    from config import config


logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Main preprocessing logic for cleaning and normalizing text blocks."""
    
    def __init__(self):
        self.config = config.get_preprocessing_config()
        self.normalizer = TextNormalizer(self.config)
        self.structure_preserver = StructurePreserver(self.config)
        self.grouper = TextBlockGrouper(self.config)
    
    def preprocess_blocks(self, blocks: List[TextBlock]) -> List[ProcessedBlock]:
        """
        Preprocess a list of text blocks.
        
        Args:
            blocks: List of TextBlock objects to preprocess
            
        Returns:
            List of ProcessedBlock objects with normalized text and features
        """
        if not blocks:
            return []
        
        logger.info(f"Preprocessing {len(blocks)} text blocks")
        
        # Step 1: Filter out empty or invalid blocks
        valid_blocks = self._filter_valid_blocks(blocks)
        logger.debug(f"Filtered to {len(valid_blocks)} valid blocks")
        
        # Step 2: Normalize text content
        normalized_blocks = []
        for block in valid_blocks:
            normalized_text = self.normalizer.normalize_text(block.text)
            if normalized_text:  # Only keep blocks with content after normalization
                normalized_block = TextBlock(
                    text=normalized_text,
                    page_number=block.page_number,
                    bbox=block.bbox,
                    font_size=block.font_size,
                    font_name=block.font_name,
                    font_flags=block.font_flags
                )
                normalized_blocks.append(normalized_block)
        
        logger.debug(f"Normalized to {len(normalized_blocks)} blocks")
        
        # Step 3: Group related text blocks
        grouped_blocks = self.grouper.group_related_blocks(normalized_blocks)
        logger.debug(f"Grouped into {len(grouped_blocks)} blocks")
        
        # Step 4: Preserve structure and create processed blocks
        processed_blocks = self.structure_preserver.preserve_structure(grouped_blocks)
        
        logger.info(f"Preprocessing complete: {len(processed_blocks)} processed blocks")
        return processed_blocks
    
    def _filter_valid_blocks(self, blocks: List[TextBlock]) -> List[TextBlock]:
        """Filter out empty or invalid text blocks."""
        valid_blocks = []
        min_length = self.config.get('min_text_length', 1)
        
        for block in blocks:
            # Skip empty or whitespace-only blocks
            if not block.text or not block.text.strip():
                continue
            
            # Skip blocks that are too short
            if len(block.text.strip()) < min_length:
                continue
            
            # Skip blocks with invalid formatting
            if block.font_size <= 0:
                continue
            
            valid_blocks.append(block)
        
        return valid_blocks


class TextNormalizer:
    """Handles text cleaning and normalization."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.normalize_whitespace = config.get('normalize_whitespace', True)
        self.preserve_formatting = config.get('preserve_formatting', True)
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text while preserving meaningful formatting.
        
        Args:
            text: Raw text to normalize
            
        Returns:
            Normalized text string
        """
        if not text:
            return ""
        
        # Step 1: Handle Unicode normalization for multilingual support
        normalized = self._normalize_unicode(text)
        
        # Step 2: Clean up whitespace while preserving structure
        if self.normalize_whitespace:
            normalized = self._normalize_whitespace(normalized)
        
        # Step 3: Remove unwanted characters but preserve formatting
        normalized = self._clean_text(normalized)
        
        # Step 4: Handle special cases
        normalized = self._handle_special_cases(normalized)
        
        return normalized.strip()
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters for consistent processing."""
        try:
            # Normalize to NFC form for consistent character representation
            normalized = unicodedata.normalize('NFC', text)
            
            # Handle common encoding issues
            normalized = normalized.replace('\ufeff', '')  # Remove BOM
            normalized = normalized.replace('\u00a0', ' ')  # Non-breaking space to regular space
            
            return normalized
        except Exception as e:
            logger.warning(f"Unicode normalization failed: {e}")
            return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace while preserving meaningful structure."""
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Clean up tabs first
        text = text.replace('\t', ' ')
        
        # Normalize line breaks but preserve paragraph breaks
        # First, normalize multiple newlines with optional whitespace to double newlines
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        # Then convert single line breaks to spaces (but not double newlines)
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        
        return text
    
    def _clean_text(self, text: str) -> str:
        """Remove unwanted characters while preserving formatting."""
        # Remove control characters except newlines and tabs
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\n\t')
        
        # Remove excessive punctuation repetition
        text = re.sub(r'([.!?]){3,}', r'\1\1\1', text)  # Limit to max 3 repetitions
        text = re.sub(r'([,-]){2,}', r'\1', text)       # Remove repeated commas/dashes
        
        # Clean up quotation marks
        text = re.sub(r'[\u201c\u201d\u201e]', '"', text)  # Normalize quotes
        text = re.sub(r'[\u2018\u2019\u201a]', "'", text)  # Normalize apostrophes
        
        return text
    
    def _handle_special_cases(self, text: str) -> str:
        """Handle special formatting cases."""
        # Handle bullet points and numbering - match at start of line or after whitespace
        text = re.sub(r'(^|\s)[•·▪▫◦‣⁃]\s*', r'\1• ', text, flags=re.MULTILINE)
        
        # Handle numbered lists
        text = re.sub(r'^(\d+)[.)]\s*', r'\1. ', text, flags=re.MULTILINE)
        
        # Handle headers with underlines (preserve them)
        if self.preserve_formatting:
            # Don't remove underlines that might indicate headers
            pass
        
        return text


class StructurePreserver:
    """Maintains formatting and spatial relationships between text elements."""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def preserve_structure(self, blocks: List[TextBlock]) -> List[ProcessedBlock]:
        """
        Preserve structural information during preprocessing.
        
        Args:
            blocks: List of TextBlock objects
            
        Returns:
            List of ProcessedBlock objects with preserved structure
        """
        if not blocks:
            return []
        
        processed_blocks = []
        
        # Calculate document-level statistics for feature normalization
        doc_stats = self._calculate_document_stats(blocks)
        
        for block in blocks:
            # Create basic feature vector (will be enhanced in feature extraction task)
            features = self._create_basic_features(block, doc_stats)
            
            # Create processed block
            processed_block = ProcessedBlock(
                text=block.text,
                page_number=block.page_number,
                features=features,
                original_block=block
            )
            
            processed_blocks.append(processed_block)
        
        return processed_blocks
    
    def _calculate_document_stats(self, blocks: List[TextBlock]) -> Dict:
        """Calculate document-level statistics for normalization."""
        if not blocks:
            return {}
        
        font_sizes = [block.font_size for block in blocks if block.font_size > 0]
        page_numbers = [block.page_number for block in blocks]
        
        stats = {
            'avg_font_size': sum(font_sizes) / len(font_sizes) if font_sizes else 12.0,
            'max_font_size': max(font_sizes) if font_sizes else 12.0,
            'min_font_size': min(font_sizes) if font_sizes else 12.0,
            'total_pages': max(page_numbers) if page_numbers else 1,
            'total_blocks': len(blocks)
        }
        
        return stats
    
    def _create_basic_features(self, block: TextBlock, doc_stats: Dict) -> FeatureVector:
        """Create basic feature vector for a text block."""
        # Font size ratio relative to document average
        avg_font_size = doc_stats.get('avg_font_size', 12.0)
        font_size_ratio = block.font_size / avg_font_size if avg_font_size > 0 else 1.0
        
        # Font style detection
        is_bold = self._is_bold_font(block.font_flags)
        is_italic = self._is_italic_font(block.font_flags)
        
        # Position normalization (basic - will be enhanced in feature extraction)
        bbox = block.bbox
        position_x = bbox[0] if bbox else 0.0
        position_y = bbox[1] if bbox else 0.0
        
        # Text analysis
        text_length = len(block.text)
        capitalization_score = self._calculate_capitalization_score(block.text)
        whitespace_ratio = self._calculate_whitespace_ratio(block.text)
        
        return FeatureVector(
            font_size_ratio=font_size_ratio,
            is_bold=is_bold,
            is_italic=is_italic,
            position_x=position_x,
            position_y=position_y,
            text_length=text_length,
            capitalization_score=capitalization_score,
            whitespace_ratio=whitespace_ratio
        )
    
    def _is_bold_font(self, font_flags: int) -> bool:
        """Check if font is bold based on flags."""
        return bool(font_flags & 2**4)  # Bold flag
    
    def _is_italic_font(self, font_flags: int) -> bool:
        """Check if font is italic based on flags."""
        return bool(font_flags & 2**1)  # Italic flag
    
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


class TextBlockGrouper:
    """Groups related text blocks and maintains spatial relationships."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.line_height_tolerance = 2.0  # Pixels
        self.horizontal_gap_threshold = 10.0  # Pixels
    
    def group_related_blocks(self, blocks: List[TextBlock]) -> List[TextBlock]:
        """
        Group related text blocks that should be treated as single units.
        
        Args:
            blocks: List of TextBlock objects
            
        Returns:
            List of TextBlock objects with related blocks merged
        """
        if not blocks:
            return []
        
        # Sort blocks by page, then by vertical position, then by horizontal position
        sorted_blocks = sorted(blocks, key=lambda b: (b.page_number, b.bbox[1], b.bbox[0]))
        
        grouped_blocks = []
        current_group = []
        
        for block in sorted_blocks:
            if not current_group:
                current_group = [block]
            elif self._should_group_blocks(current_group[-1], block):
                current_group.append(block)
            else:
                # Finalize current group and start new one
                if current_group:
                    merged_block = self._merge_blocks(current_group)
                    grouped_blocks.append(merged_block)
                current_group = [block]
        
        # Don't forget the last group
        if current_group:
            merged_block = self._merge_blocks(current_group)
            grouped_blocks.append(merged_block)
        
        logger.debug(f"Grouped {len(blocks)} blocks into {len(grouped_blocks)} groups")
        return grouped_blocks
    
    def _should_group_blocks(self, block1: TextBlock, block2: TextBlock) -> bool:
        """Determine if two blocks should be grouped together."""
        # Must be on the same page
        if block1.page_number != block2.page_number:
            return False
        
        # Must have similar font characteristics
        if not self._similar_font_characteristics(block1, block2):
            return False
        
        # Must be spatially close
        if not self._spatially_close(block1, block2):
            return False
        
        return True
    
    def _similar_font_characteristics(self, block1: TextBlock, block2: TextBlock) -> bool:
        """Check if two blocks have similar font characteristics."""
        # Font size should be similar (within 10% tolerance)
        size_ratio = abs(block1.font_size - block2.font_size) / max(block1.font_size, block2.font_size)
        if size_ratio > 0.1:
            return False
        
        # Font name should be the same
        if block1.font_name != block2.font_name:
            return False
        
        # Font flags should be the same
        if block1.font_flags != block2.font_flags:
            return False
        
        return True
    
    def _spatially_close(self, block1: TextBlock, block2: TextBlock) -> bool:
        """Check if two blocks are spatially close enough to group."""
        bbox1 = block1.bbox
        bbox2 = block2.bbox
        
        # Check if blocks are on the same line (similar y-coordinates)
        y_diff = abs(bbox1[1] - bbox2[1])
        if y_diff <= self.line_height_tolerance:
            # Same line - check horizontal gap
            horizontal_gap = bbox2[0] - bbox1[2]  # Left edge of block2 - right edge of block1
            return horizontal_gap <= self.horizontal_gap_threshold
        
        # Check if blocks are vertically adjacent
        vertical_gap = bbox2[1] - bbox1[3]  # Top of block2 - bottom of block1
        if 0 <= vertical_gap <= self.line_height_tolerance:
            # Check horizontal overlap
            horizontal_overlap = min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0])
            return horizontal_overlap > 0
        
        return False
    
    def _merge_blocks(self, blocks: List[TextBlock]) -> TextBlock:
        """Merge a list of related blocks into a single block."""
        if len(blocks) == 1:
            return blocks[0]
        
        # Combine text with appropriate spacing
        texts = []
        for i, block in enumerate(blocks):
            if i > 0:
                # Add space between blocks on the same line, newline for different lines
                prev_block = blocks[i-1]
                if abs(block.bbox[1] - prev_block.bbox[1]) <= self.line_height_tolerance:
                    texts.append(' ')  # Same line
                else:
                    texts.append('\n')  # Different line
            texts.append(block.text)
        
        merged_text = ''.join(texts)
        
        # Calculate merged bounding box
        min_x = min(block.bbox[0] for block in blocks)
        min_y = min(block.bbox[1] for block in blocks)
        max_x = max(block.bbox[2] for block in blocks)
        max_y = max(block.bbox[3] for block in blocks)
        merged_bbox = (min_x, min_y, max_x, max_y)
        
        # Use characteristics from the first block
        first_block = blocks[0]
        
        return TextBlock(
            text=merged_text,
            page_number=first_block.page_number,
            bbox=merged_bbox,
            font_size=first_block.font_size,
            font_name=first_block.font_name,
            font_flags=first_block.font_flags
        )