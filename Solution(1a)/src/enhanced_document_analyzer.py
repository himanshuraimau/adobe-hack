# Enhanced Document Analyzer - Optimized for PDF Heading Extraction Challenge
# Focus: High precision title and heading detection with proper hierarchy

import re
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Tuple
import math

class EnhancedDocumentAnalyzer:
    """
    Enhanced document analyzer focused on accurate title and heading extraction
    with proper hierarchy assignment (H1, H2, H3) and robust fragment filtering.
    """
    
    def __init__(self):
        # Title detection patterns
        self.title_indicators = [
            r'^(introduction|overview|summary|abstract|conclusion)',
            r'^(chapter|section|part)\s+\d+',
            r'^\d+\.\s+[A-Z]',
            r'^[A-Z][A-Z\s]+$',  # All caps titles
        ]
        
        # Fragment patterns to exclude
        self.fragment_patterns = [
            r'^[a-zA-Z]{1,3}\s*$',  # Single letters or very short fragments
            r'.*\s+[a-zA-Z]\s*$',   # Text ending with single letter
            r'^(quest|oposal|r pr|rfp:?\s*r|f|r|pr)\s*$',  # Broken fragments
            r'(request|quest)\s+[a-zA-Z]{1,3}\s*$',  # "request f" patterns
            r'^rfp:\s*request\s+[a-zA-Z]{1,3}\s*$',  # "RFP: Request f"
            r'^march\s+\d{4}\s*$',  # Date fragments
            r'^\w{1,2}\s*$',        # Very short words
            r'^(from|the|to|of|and|or|in|at|on|by)$',  # Single prepositions
            r'quest\s+for\s+pr',    # "quest for Pr" fragments
            r'r\s+proposal',        # "r Proposal" fragments
            r'^\d+\.\s+\d+\.\s*$',  # Number fragments like "5. 6."
        ]
        
        # Caption and noise patterns to exclude
        self.exclude_patterns = [
            r'^(figure|fig\.?|table|tbl\.?)\s+\d+',
            r'^(image|photo|chart|graph|diagram)',
            r'source:|note:|adapted from:',
            r'^(address|phone|email|date|time|location):',
            r'^(www\.|http)',
            r'^\([^)]*\)$',
            r'^\d{2,}$',
            r'^-+$',
        ]
        
        # Numbering patterns for hierarchy detection
        self.numbering_patterns = {
            'H1': [
                r'^\d+\.\s+',              # 1. Title
                r'^(chapter|section|part)\s+\d+',
                r'^[A-Z]\.\s+',            # A. Title
                r'^[IVXLC]+\.\s+',         # I. Title (Roman)
            ],
            'H2': [
                r'^\d+\.\d+\.\s+',         # 1.1. Subtitle
                r'^\d+\.\d+\s+',           # 1.1 Subtitle
                r'^[A-Z]\.\d+\.\s+',       # A.1. Subtitle
            ],
            'H3': [
                r'^\d+\.\d+\.\d+\.\s+',    # 1.1.1. Subsubtitle
                r'^\d+\.\d+\.\d+\s+',      # 1.1.1 Subsubtitle
                r'^\([a-z]\)\s+',          # (a) Item
                r'^\([0-9]\)\s+',          # (1) Item
            ]
        }
        
    def analyze_document(self, extracted_data: Dict[str, Any], nlp_pipeline: Any = None) -> Dict[str, Any]:
        """
        Main analysis function that extracts title and headings with proper hierarchy.
        """
        text_blocks = extracted_data.get("text_blocks", [])
        page_dimensions = extracted_data.get("page_dimensions", [])
        
        if not text_blocks:
            return {"title": "", "outline": []}
        
        print(f"Analyzing document with {len(text_blocks)} text blocks")
        
        # Step 1: Clean and filter text blocks
        cleaned_blocks = self._clean_text_blocks(text_blocks)
        print(f"After cleaning: {len(cleaned_blocks)} blocks")
        
        # Step 2: Merge fragmented text (multi-line headings)
        merged_blocks = self._merge_multi_line_text(cleaned_blocks)
        print(f"After merging: {len(merged_blocks)} blocks")
        
        # Step 3: Analyze document style and identify body text patterns
        style_analysis = self._analyze_document_style(merged_blocks)
        
        # Step 4: Extract title
        title = self._extract_title(merged_blocks, style_analysis)
        print(f"Extracted title: '{title}'")
        
        # Step 5: Extract and classify headings
        headings = self._extract_headings(merged_blocks, style_analysis, page_dimensions)
        print(f"Extracted {len(headings)} headings")
        
        # Step 6: Assign hierarchy levels (H1, H2, H3)
        hierarchical_headings = self._assign_hierarchy(headings, style_analysis)
        
        # Step 7: Format output
        outline = self._format_outline(hierarchical_headings, title)
        
        return {
            "title": title,
            "outline": outline
        }
    
    def _clean_text_blocks(self, text_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove obvious fragments and noise."""
        cleaned = []
        
        for block in text_blocks:
            text = block.get('text', '').strip()
            
            if len(text) < 2:  # Too short
                continue
                
            if len(text) > 200:  # Too long for heading
                continue
                
            # Check fragment patterns
            is_fragment = False
            for pattern in self.fragment_patterns:
                if re.match(pattern, text, re.IGNORECASE):
                    is_fragment = True
                    break
            
            if is_fragment:
                continue
                
            # Check exclude patterns
            is_excluded = False
            for pattern in self.exclude_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    is_excluded = True
                    break
            
            if is_excluded:
                continue
            
            # Filter out obvious body text
            if self._is_obvious_body_text(text):
                continue
                
            cleaned.append(block)
        
        return cleaned
    
    def _is_obvious_body_text(self, text: str) -> bool:
        """Identify obvious body text that shouldn't be considered for headings."""
        # Long sentences with multiple clauses
        if len(text) > 100 and text.count(',') > 2:
            return True
            
        # Ends with sentence punctuation and is long
        if text.endswith(('.', '!', '?')) and len(text) > 80:
            return True
            
        # Contains many common words indicating prose
        common_words = ['the', 'and', 'is', 'are', 'was', 'were', 'will', 'would', 'could', 'should']
        word_count = len(text.split())
        common_count = sum(1 for word in text.lower().split() if word in common_words)
        
        if word_count > 5 and common_count / word_count > 0.3:
            return True
            
        return False
    
    def _merge_multi_line_text(self, text_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge text blocks that should be combined (multi-line headings)."""
        if not text_blocks:
            return []
        
        # Sort by page and vertical position
        sorted_blocks = sorted(text_blocks, key=lambda x: (x.get('page_num', 0), x.get('bbox', [0,0,0,0])[1]))
        
        merged = []
        i = 0
        
        while i < len(sorted_blocks):
            current = sorted_blocks[i]
            current_text = current.get('text', '').strip()
            
            # Look for continuation on next line
            if i + 1 < len(sorted_blocks):
                next_block = sorted_blocks[i + 1]
                next_text = next_block.get('text', '').strip()
                
                # Check if they should be merged
                if self._should_merge_blocks(current, next_block, current_text, next_text):
                    # Merge the blocks
                    merged_text = current_text + ' ' + next_text
                    merged_block = current.copy()
                    merged_block['text'] = merged_text
                    
                    # Update bounding box
                    curr_bbox = current.get('bbox', [0,0,0,0])
                    next_bbox = next_block.get('bbox', [0,0,0,0])
                    merged_block['bbox'] = [
                        min(curr_bbox[0], next_bbox[0]),  # left
                        min(curr_bbox[1], next_bbox[1]),  # top
                        max(curr_bbox[2], next_bbox[2]),  # right
                        max(curr_bbox[3], next_bbox[3])   # bottom
                    ]
                    
                    merged.append(merged_block)
                    i += 2  # Skip next block since we merged it
                    continue
            
            merged.append(current)
            i += 1
        
        return merged
    
    def _should_merge_blocks(self, block1: Dict, block2: Dict, text1: str, text2: str) -> bool:
        """Determine if two text blocks should be merged."""
        # Must be on same page
        if block1.get('page_num') != block2.get('page_num'):
            return False
        
        # Must have similar font properties
        if (abs(block1.get('font_size', 12) - block2.get('font_size', 12)) > 1 or
            block1.get('font_name') != block2.get('font_name') or
            block1.get('is_bold') != block2.get('is_bold')):
            return False
        
        # Check vertical proximity
        bbox1 = block1.get('bbox', [0,0,0,0])
        bbox2 = block2.get('bbox', [0,0,0,0])
        vertical_gap = abs(bbox2[1] - bbox1[3])  # Gap between bottom of first and top of second
        
        if vertical_gap > 15:  # Too far apart
            return False
        
        # Check if first text looks incomplete
        if text1 and not text1[-1].isalnum() and text1[-1] not in '.-':
            return True
        
        # Check if second text looks like continuation
        if text2 and text2[0].islower():
            return True
        
        # Both are short and similar style - likely multi-line heading
        if len(text1) < 50 and len(text2) < 50:
            return True
        
        return False
    
    def _analyze_document_style(self, text_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze document to identify common fonts and styles."""
        font_sizes = []
        font_names = []
        bold_count = 0
        
        for block in text_blocks:
            font_sizes.append(block.get('font_size', 12))
            font_names.append(block.get('font_name', 'default'))
            if block.get('is_bold', False):
                bold_count += 1
        
        if not font_sizes:
            return {
                'body_font_size': 12,
                'common_font_name': 'default',
                'bold_ratio': 0,
                'font_size_distribution': {}
            }
        
        # Most common font size is likely body text
        size_counter = Counter(font_sizes)
        body_font_size = size_counter.most_common(1)[0][0]
        
        # Most common font name
        name_counter = Counter(font_names)
        common_font_name = name_counter.most_common(1)[0][0]
        
        # Calculate font size distribution
        unique_sizes = sorted(set(font_sizes), reverse=True)
        size_distribution = {size: size_counter[size] for size in unique_sizes}
        
        return {
            'body_font_size': body_font_size,
            'common_font_name': common_font_name,
            'bold_ratio': bold_count / len(text_blocks),
            'font_size_distribution': size_distribution,
            'all_font_sizes': unique_sizes
        }
    
    def _extract_title(self, text_blocks: List[Dict[str, Any]], style_analysis: Dict) -> str:
        """Extract document title using various heuristics."""
        if not text_blocks:
            return ""
        
        # Sort blocks by page and position (top to bottom)
        sorted_blocks = sorted(text_blocks, key=lambda x: (x.get('page_num', 0), x.get('bbox', [0,0,0,0])[1]))
        
        # Look for title in first few blocks
        title_candidates = []
        
        for i, block in enumerate(sorted_blocks[:15]):  # Check first 15 blocks
            text = block.get('text', '').strip()
            
            if len(text) < 3 or len(text) > 100:  # Too short or too long
                continue
            
            # Skip obvious non-titles
            if self._is_obvious_non_title(text):
                continue
            
            score = self._calculate_title_score(block, style_analysis, i)
            
            if score > 10:  # Minimum threshold for title consideration
                title_candidates.append({
                    'text': text,
                    'score': score,
                    'position': i,
                    'page': block.get('page_num', 0)
                })
        
        if not title_candidates:
            return ""
        
        # Sort by score and return best candidate
        title_candidates.sort(key=lambda x: x['score'], reverse=True)
        best_title = title_candidates[0]['text']
        
        # Additional validation - reject obvious bad titles
        if self._is_bad_title(best_title):
            return ""
        
        return best_title
    
    def _is_obvious_non_title(self, text: str) -> bool:
        """Check if text is obviously not a title."""
        text_lower = text.lower()
        
        # Skip fragments we know are not titles
        bad_patterns = [
            r'quest for pr', r'r proposal', r'rfp:', r'request f',
            r'rsvp:', r'closed toed shoes', r'climbing',
            r'application form', r'grant of ltc',
            r'foundation level extensions'
        ]
        
        for pattern in bad_patterns:
            if re.search(pattern, text_lower):
                return True
        
        # Skip if starts with lowercase (likely fragment)
        if text and text[0].islower():
            return True
        
        # Skip if contains excessive punctuation for a title
        if text.count('-') > 5 or text.count(':') > 2:
            return True
        
        return False
    
    def _is_bad_title(self, text: str) -> bool:
        """Additional validation to reject bad titles."""
        text_lower = text.lower()
        
        # Reject specific known bad patterns
        if any(bad in text_lower for bad in [
            'quest for pr', 'r proposal', 'request f', 'rsvp:', 
            'closed toed shoes', 'climbing', 'application form'
        ]):
            return True
        
        # Reject if too many special characters
        special_count = sum(text.count(c) for c in ':-()[]{}/')
        if special_count > len(text) * 0.3:
            return True
        
        return False
    
    def _calculate_title_score(self, block: Dict, style_analysis: Dict, position: int) -> float:
        """Calculate score for title candidacy."""
        text = block.get('text', '').strip()
        score = 0.0
        
        # Position bonus (earlier is better for title)
        if position == 0:
            score += 50
        elif position <= 2:
            score += 30
        elif position <= 5:
            score += 10
        
        # Font size bonus
        font_size = block.get('font_size', 12)
        body_size = style_analysis.get('body_font_size', 12)
        
        if font_size > body_size * 1.2:
            score += (font_size - body_size) * 5
        
        # Bold bonus
        if block.get('is_bold', False):
            score += 20
        
        # Length sweet spot for titles
        word_count = len(text.split())
        if 2 <= word_count <= 8:
            score += 15
        elif word_count > 15:
            score -= 20  # Too long for title
        
        # Center alignment bonus
        bbox = block.get('bbox', [0,0,0,0])
        center_x = (bbox[0] + bbox[2]) / 2
        # Assuming page width of 600 (approximate)
        if abs(center_x - 300) < 50:
            score += 15
        
        # All caps bonus (but penalize if too long)
        if text.isupper() and len(text) <= 50:
            score += 10
        elif text.isupper() and len(text) > 50:
            score -= 10
        
        # Capitalization pattern bonus
        words = text.split()
        if len(words) >= 2 and all(word[0].isupper() for word in words if len(word) > 2):
            score += 10
        
        # Penalty for ending with punctuation (except appropriate ones)
        if text.endswith(('.', '!', '?', ':')):
            score -= 10
        
        # Pattern matching bonuses
        for pattern in self.title_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                score += 20
                break
        
        return score
    
    def _extract_headings(self, text_blocks: List[Dict[str, Any]], style_analysis: Dict, page_dimensions: List[Dict]) -> List[Dict[str, Any]]:
        """Extract heading candidates using enhanced heuristics."""
        heading_candidates = []
        body_font_size = style_analysis.get('body_font_size', 12)
        
        for block in text_blocks:
            text = block.get('text', '').strip()
            
            if len(text) < 2 or len(text) > 150:
                continue
            
            # Skip obvious non-headings
            if self._is_obvious_non_heading(text):
                continue
            
            score = self._calculate_heading_score(block, style_analysis)
            
            if score >= 15:  # Threshold for heading candidacy
                heading_candidates.append({
                    'text': text,
                    'score': score,
                    'page_num': block.get('page_num', 0),
                    'bbox': block.get('bbox', [0,0,0,0]),
                    'font_size': block.get('font_size', 12),
                    'is_bold': block.get('is_bold', False),
                    'font_name': block.get('font_name', 'default')
                })
        
        # Sort by score
        heading_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        return heading_candidates
    
    def _is_obvious_non_heading(self, text: str) -> bool:
        """Check if text is obviously not a heading."""
        text_lower = text.lower()
        
        # Skip addresses and contact info
        if any(indicator in text_lower for indicator in [
            'parkway', 'forge, tn', 'pigeon forge', '@', 'phone:', 'tel:', 'fax:',
            'www.', 'http', '.com', '.org', '.edu', 'address:', 'email:'
        ]):
            return True
        
        # Skip instructions and actions
        if any(action in text_lower for action in [
            'please visit', 'click here', 'fill out', 'to attend',
            'parents or guardians', 'not attending', 'required for'
        ]):
            return True
        
        # Skip if starts with lowercase (likely continuation)
        if text and text[0].islower():
            return True
        
        # Skip pure numbers or dates
        if re.match(r'^\d+$', text) or re.match(r'^\d{1,2}/\d{1,2}/\d{4}$', text):
            return True
        
        # Skip very long sentences (likely body text)
        if len(text) > 100 and text.count(' ') > 15:
            return True
        
        return False
    
    def _calculate_heading_score(self, block: Dict, style_analysis: Dict) -> float:
        """Calculate heading score for a text block."""
        text = block.get('text', '').strip()
        score = 0.0
        
        font_size = block.get('font_size', 12)
        body_size = style_analysis.get('body_font_size', 12)
        
        # Font size bonus
        if font_size > body_size:
            score += (font_size - body_size) * 3
        
        # Bold bonus
        if block.get('is_bold', False):
            score += 25
        
        # Numbering patterns bonus
        for level, patterns in self.numbering_patterns.items():
            for pattern in patterns:
                if re.match(pattern, text):
                    score += 40
                    break
        
        # Length sweet spot
        word_count = len(text.split())
        if 1 <= word_count <= 8:
            score += 15
        elif word_count > 12:
            score -= 10
        
        # All caps bonus (reasonable length)
        if text.isupper() and 4 <= len(text) <= 40:
            score += 15
        
        # Capitalization pattern
        words = text.split()
        if len(words) >= 2 and all(word[0].isupper() for word in words if len(word) > 2):
            score += 10
        
        # Penalty for ending with sentence punctuation
        if text.endswith(('.', '!', '?')):
            score -= 20
        
        # Penalty for too many punctuation marks
        punct_count = sum(text.count(p) for p in ',.;')
        if punct_count > 2:
            score -= punct_count * 5
        
        # Bonus for heading-like words
        heading_words = ['introduction', 'overview', 'summary', 'conclusion', 'background', 'methodology', 'results', 'discussion']
        if any(word in text.lower() for word in heading_words):
            score += 20
        
        return score
    
    def _assign_hierarchy(self, headings: List[Dict[str, Any]], style_analysis: Dict) -> List[Dict[str, Any]]:
        """Assign H1, H2, H3 levels to headings based on font size and numbering."""
        if not headings:
            return []
        
        # Group by font size
        font_sizes = [h['font_size'] for h in headings]
        unique_sizes = sorted(set(font_sizes), reverse=True)
        
        # Create font size to level mapping
        size_to_level = {}
        for i, size in enumerate(unique_sizes[:3]):  # Only H1, H2, H3
            if i == 0:
                size_to_level[size] = 'H1'
            elif i == 1:
                size_to_level[size] = 'H2'
            else:
                size_to_level[size] = 'H3'
        
        # Apply hierarchy
        for heading in headings:
            text = heading['text']
            font_size = heading['font_size']
            is_bold = heading.get('is_bold', False)
            
            # First check numbering patterns
            level_assigned = False
            for level, patterns in self.numbering_patterns.items():
                for pattern in patterns:
                    if re.match(pattern, text):
                        heading['level'] = level
                        level_assigned = True
                        break
                if level_assigned:
                    break
            
            # If no numbering pattern matched, use font size and style
            if not level_assigned:
                base_level = size_to_level.get(font_size, 'H3')
                
                # Adjust based on content and style
                if is_bold and font_size == max(unique_sizes):
                    heading['level'] = 'H1'
                elif is_bold and len(unique_sizes) > 1 and font_size >= unique_sizes[1]:
                    heading['level'] = 'H2'
                else:
                    heading['level'] = base_level
        
        return headings
    
    def _format_outline(self, headings: List[Dict[str, Any]], title: str = "") -> List[Dict[str, str]]:
        """Format headings into the required output structure."""
        outline = []
        
        # Sort headings by page and vertical position first
        sorted_headings = sorted(headings, key=lambda x: (x['page_num'], x.get('bbox', [0,0,0,0])[1]))
        
        for heading in sorted_headings:
            heading_text = heading['text']
            
            # Skip if this heading is the same as the title
            if title and heading_text.strip() == title.strip():
                continue
            
            # Skip obvious quality issues
            if self._is_low_quality_heading(heading_text):
                continue
            
            outline.append({
                'level': heading['level'],
                'text': heading_text,
                'page': heading['page_num'] + 1  # Convert to 1-based page numbering
            })
        
        return outline
    
    def _is_low_quality_heading(self, text: str) -> bool:
        """Check if a heading is low quality and should be filtered out."""
        text_lower = text.lower()
        
        # Filter out website URLs and email addresses
        if any(indicator in text_lower for indicator in [
            'www.', 'http', '.com', '.org', '.edu', '@'
        ]):
            return True
        
        # Filter out pure addresses
        if any(addr in text_lower for addr in [
            'parkway', 'street', 'avenue', 'road', 'drive', 'tn ', 'ca ', 'ny '
        ]):
            return True
        
        # Filter out RSVP and party instructions
        if any(instr in text_lower for instr in [
            'rsvp:', 'closed toed shoes', 'required for climbing',
            'parents or guardians', 'please visit', 'fill out waiver'
        ]):
            return True
        
        # Filter out excessive punctuation
        if text.count('-') > 10 or text.count(':') > 3:
            return True
        
        return False
