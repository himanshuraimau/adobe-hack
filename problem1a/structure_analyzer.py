"""
Structure analysis module for building document hierarchy from classification results.

This module processes classification results to determine proper hierarchical relationships
between headings, detect document titles, and handle edge cases like missing levels.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from collections import defaultdict

from models import ClassificationResult, Heading, DocumentStructure

logger = logging.getLogger(__name__)


class StructureAnalyzer:
    """Main structure analysis logic for building document hierarchy."""
    
    def __init__(self):
        self.hierarchy_builder = HierarchyBuilder()
        self.title_detector = TitleDetector()
    
    def analyze_structure(self, classified_blocks: List[ClassificationResult]) -> DocumentStructure:
        """
        Analyze classification results to build document structure.
        
        Args:
            classified_blocks: List of classification results from heading classifier
            
        Returns:
            DocumentStructure with title and hierarchical headings
        """
        try:
            # Filter out regular text blocks and convert to headings
            heading_blocks = [
                block for block in classified_blocks 
                if block.predicted_class in ['title', 'h1', 'h2', 'h3']
            ]
            
            if not heading_blocks:
                logger.warning("No headings detected in document")
                return DocumentStructure(
                    title=None,
                    headings=[],
                    metadata={'total_blocks': len(classified_blocks), 'heading_blocks': 0}
                )
            
            # Convert classification results to headings
            raw_headings = self._convert_to_headings(heading_blocks)
            
            # Detect document title
            title = self.title_detector.detect_title(heading_blocks)
            
            # Remove title from headings if it was detected
            filtered_headings = self._filter_title_from_headings(raw_headings, title)
            
            # Build proper hierarchy
            structured_headings = self.hierarchy_builder.build_hierarchy(filtered_headings)
            
            # Create metadata
            metadata = {
                'total_blocks': len(classified_blocks),
                'heading_blocks': len(heading_blocks),
                'title_detected': title is not None,
                'hierarchy_levels': self._count_hierarchy_levels(structured_headings)
            }
            
            return DocumentStructure(
                title=title,
                headings=structured_headings,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Structure analysis failed: {e}")
            # Return minimal structure on error
            return DocumentStructure(
                title=None,
                headings=[],
                metadata={'error': str(e), 'total_blocks': len(classified_blocks)}
            )
    
    def _convert_to_headings(self, classified_blocks: List[ClassificationResult]) -> List[Heading]:
        """Convert classification results to Heading objects."""
        headings = []
        
        for block in classified_blocks:
            # Convert predicted class to proper heading level format
            level = self._normalize_heading_level(block.predicted_class)
            
            heading = Heading(
                level=level,
                text=block.block.text.strip(),
                page=block.block.page_number,
                confidence=block.confidence
            )
            headings.append(heading)
        
        return headings
    
    def _normalize_heading_level(self, predicted_class: str) -> str:
        """Normalize predicted class to proper heading level format."""
        class_mapping = {
            'title': 'title',
            'h1': 'H1',
            'h2': 'H2',
            'h3': 'H3'
        }
        return class_mapping.get(predicted_class, 'H1')
    
    def _filter_title_from_headings(self, headings: List[Heading], title: Optional[str]) -> List[Heading]:
        """Remove the detected title from the headings list."""
        if not title:
            return headings
        
        # Remove headings that match the title text
        filtered = []
        for heading in headings:
            if heading.text.strip() != title.strip():
                filtered.append(heading)
            else:
                logger.debug(f"Filtered title from headings: {heading.text}")
        
        return filtered
    
    def _count_hierarchy_levels(self, headings: List[Heading]) -> Dict[str, int]:
        """Count the number of headings at each level."""
        counts = defaultdict(int)
        for heading in headings:
            counts[heading.level] += 1
        return dict(counts)


class HierarchyBuilder:
    """Builds heading hierarchy and handles missing levels."""
    
    def build_hierarchy(self, headings: List[Heading]) -> List[Heading]:
        """
        Build proper hierarchical relationships between headings.
        
        Args:
            headings: List of raw headings
            
        Returns:
            List of headings with proper hierarchical levels
        """
        if not headings:
            return []
        
        # Sort headings by page number to maintain document order
        sorted_headings = sorted(headings, key=lambda h: (h.page, h.text))
        
        # Handle missing hierarchy levels and inconsistent formatting
        normalized_headings = self._normalize_hierarchy_levels(sorted_headings)
        
        # Ensure proper hierarchy progression
        structured_headings = self._ensure_hierarchy_progression(normalized_headings)
        
        return structured_headings
    
    def _normalize_hierarchy_levels(self, headings: List[Heading]) -> List[Heading]:
        """
        Normalize heading levels based on font size, position, and content analysis.
        
        This handles cases where the classifier might have inconsistent results.
        For now, we'll keep it simple and trust the classifier results.
        """
        if not headings:
            return []
        
        # For now, just return the headings as-is
        # The hierarchy progression step will handle the actual level adjustments
        return headings
    

    
    def _ensure_hierarchy_progression(self, headings: List[Heading]) -> List[Heading]:
        """
        Ensure proper hierarchy progression (H1 -> H2 -> H3).
        
        Handles missing levels and inconsistent formatting.
        """
        if not headings:
            return []
        
        structured = []
        current_levels = {'H1': False, 'H2': False, 'H3': False}
        
        for heading in headings:
            adjusted_level = self._adjust_level_for_progression(heading.level, current_levels)
            
            # Update current levels based on adjusted level
            if adjusted_level == 'H1':
                current_levels['H1'] = True
            elif adjusted_level == 'H2':
                current_levels['H2'] = True
            elif adjusted_level == 'H3':
                current_levels['H3'] = True
            
            adjusted_heading = Heading(
                level=adjusted_level,
                text=heading.text,
                page=heading.page,
                confidence=heading.confidence
            )
            structured.append(adjusted_heading)
        
        return structured
    
    def _adjust_level_for_progression(self, level: str, current_levels: Dict[str, bool]) -> str:
        """Adjust heading level to ensure proper hierarchy progression."""
        if level == 'H1':
            return 'H1'
        elif level == 'H2':
            # H2 can appear after H1, or be promoted to H1 only if no H1 has been seen yet
            if current_levels['H1']:
                return 'H2'  # H1 exists, so H2 can remain H2
            else:
                return 'H1'  # Promote first H2 to H1 if no H1 exists yet
        elif level == 'H3':
            # H3 can only appear after H2
            if current_levels['H2']:
                return 'H3'
            elif current_levels['H1']:
                return 'H2'  # Promote to H2 if H1 exists but no H2 exists yet
            else:
                return 'H1'  # Promote to H1 if no H1 exists yet
        
        return level


class TitleDetector:
    """Detects document title using multiple heuristics."""
    
    def detect_title(self, classified_blocks: List[ClassificationResult]) -> Optional[str]:
        """
        Detect document title using multiple heuristics.
        
        Args:
            classified_blocks: List of classification results
            
        Returns:
            Detected title text or None if no title found
        """
        if not classified_blocks:
            return None
        
        # Strategy 1: Look for blocks explicitly classified as 'title'
        title_blocks = [
            block for block in classified_blocks 
            if block.predicted_class == 'title'
        ]
        
        if title_blocks:
            # Use the title block with highest confidence
            best_title = max(title_blocks, key=lambda b: b.confidence)
            return best_title.block.text.strip()
        
        # Strategy 2: Use first heading as title
        first_heading = self._find_first_heading(classified_blocks)
        if first_heading:
            return first_heading.block.text.strip()
        
        # Strategy 3: Find largest font heading on first page
        largest_font_heading = self._find_largest_font_heading(classified_blocks)
        if largest_font_heading:
            return largest_font_heading.block.text.strip()
        
        # Strategy 4: Use document metadata (if available)
        # This would require access to PDF metadata, which we don't have in this context
        
        return None
    
    def _find_first_heading(self, classified_blocks: List[ClassificationResult]) -> Optional[ClassificationResult]:
        """Find the first heading in the document."""
        heading_blocks = [
            block for block in classified_blocks 
            if block.predicted_class in ['h1', 'h2', 'h3']
        ]
        
        if not heading_blocks:
            return None
        
        # Sort by page number and position
        sorted_blocks = sorted(
            heading_blocks, 
            key=lambda b: (b.block.page_number, b.block.features.position_y)
        )
        
        return sorted_blocks[0]
    
    def _find_largest_font_heading(self, classified_blocks: List[ClassificationResult]) -> Optional[ClassificationResult]:
        """Find the heading with the largest font on the first page."""
        # Filter to first page headings only
        first_page_headings = [
            block for block in classified_blocks 
            if (block.predicted_class in ['h1', 'h2', 'h3'] and 
                block.block.page_number == 1)
        ]
        
        if not first_page_headings:
            return None
        
        # Find the one with largest font size ratio
        largest_font = max(
            first_page_headings, 
            key=lambda b: b.block.features.font_size_ratio
        )
        
        return largest_font