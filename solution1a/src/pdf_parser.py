import re
import fitz  # PyMuPDF
from collections import defaultdict, Counter
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class BlockType(Enum):
    """Document structure elements"""
    TITLE = "title"
    HEADING = "heading"
    SUBHEADING = "subheading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    CAPTION = "caption"
    FOOTER = "footer"
    HEADER = "header"
    QUOTE = "quote"
    CODE = "code"
    FIGURE = "figure"
    FOOTNOTE = "footnote"

class DocumentType(Enum):
    """Document categories for optimization"""
    UNIVERSAL = "universal"
    ACADEMIC = "academic"
    BUSINESS = "business"
    TECHNICAL = "technical"
    LEGAL = "legal"
    BOOK = "book"
    MAGAZINE = "magazine"
    FORM = "form"
    SCANNED = "scanned"

@dataclass
class TextProperties:
    """Enhanced text metadata"""
    font_size: float
    font_name: str
    is_bold: bool
    is_italic: bool
    is_underlined: bool
    color: int
    superscript: bool = False
    subscript: bool = False

@dataclass
class UniversalPDFConfig:
    """Optimal settings for any PDF type - optimized for 10-second constraint"""
    detect_tables: bool = False  # Disabled for speed
    extract_images: bool = False  # Disabled for speed
    preserve_layout: bool = True
    merge_hyphenated: bool = True
    detect_columns: bool = False  # Disabled for speed
    clean_artifacts: bool = True
    classify_blocks: bool = True
    detect_lists: bool = False  # Disabled for speed
    identify_headers_footers: bool = False  # Disabled for speed
    min_text_length: int = 2
    table_detection_method: str = "enhanced"  # Options: "none", "native", "heuristic", "enhanced" (both)
    table_min_quality: float = 60.0  # Minimum quality score for heuristic tables (0-100)
    image_min_size: tuple = (10, 10)
    adaptive_processing: bool = False  # Disabled for speed

class AdvancedPDFParser:
    """Universal PDF parser optimized for all document types"""
    
    def __init__(self, config: UniversalPDFConfig = None):
        self.config = config if config else UniversalPDFConfig()
        self.font_analysis = {}
        self.layout_analysis = {}
        
    def _merge_fragmented_lines(self, text_blocks: List[Dict], y_gap: float = 4.0) -> List[Dict]:
        """Merge lines based on vertical proximity."""
        if not text_blocks:
            return []

        blocks = sorted(text_blocks, key=lambda b: (b['page_num'], b['bbox'][1]))
        result = []
        group = []

        for block in blocks:
            if not group:
                group.append(block)
                continue

            prev = group[-1]
            gap = block['bbox'][1] - prev['bbox'][3]

            if (gap <= y_gap and
                block['page_num'] == prev['page_num'] and
                abs(block['font_size'] - prev['font_size']) < 1 and
                block['font_name'] == prev['font_name'] and
                block['is_bold'] == prev['is_bold']):
                group.append(block)
            else:
                if len(group) > 1:
                    text = ' '.join(b['text'] for b in group)
                    merged = group[0].copy()
                    merged['text'] = text
                    merged['bbox'] = [
                        min(b['bbox'][0] for b in group),
                        min(b['bbox'][1] for b in group),
                        max(b['bbox'][2] for b in group),
                        max(b['bbox'][3] for b in group)
                    ]
                    merged['char_count'] = len(text)
                    merged['word_count'] = len(text.split())
                    result.append(merged)
                else:
                    result.append(group[0])
                group = [block]

        if group:
            if len(group) > 1:
                text = ' '.join(b['text'] for b in group)
                merged = group[0].copy()
                merged['text'] = text
                merged['bbox'] = [
                    min(b['bbox'][0] for b in group),
                    min(b['bbox'][1] for b in group),
                    max(b['bbox'][2] for b in group),
                    max(b['bbox'][3] for b in group)
                ]
                merged['char_count'] = len(text)
                merged['word_count'] = len(text.split())
                result.append(merged)
            else:
                result.append(group[0])

        return result

    def parse_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Enhanced main parsing method with improved performance for all PDF sizes.
        - Uses generator for page processing and faster fitz text extraction
        - Optimizes memory usage for large documents
        - Improves heading detection with contextual analysis
        """
        import time
        import os
        start_time = time.time()
        
        # Check file size to adapt processing strategy
        try:
            file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
            is_large_pdf = file_size_mb > 20  # Consider PDFs > 20MB as large
        except Exception:
            is_large_pdf = False  # Default to standard processing if size check fails
            
        # Open document with appropriate settings
        try:
            # For large PDFs, use memory optimization settings
            if is_large_pdf:
                doc = fitz.open(pdf_path, filetype="pdf")
            else:
                doc = fitz.open(pdf_path)
        except Exception as e:
            raise IOError(f"Failed to open PDF: {e}")
            
        result = {
            "text_blocks": [],
            "tables": [],
            "images": [],
            "metadata": self._get_metadata(doc),
            "pages": []
        }
        
        # Process pages with memory optimization for large documents
        page_count = len(doc)
        
        # For large documents, process in batches to optimize memory
        batch_size = 10 if is_large_pdf else page_count
        for batch_start in range(0, page_count, batch_size):
            batch_end = min(batch_start + batch_size, page_count)
            
            # Process each page in the batch
            for page_num in range(batch_start, batch_end):
                page = doc[page_num]
                page_info = self._process_page(page, page_num)
                result["pages"].append(page_info)
                result["text_blocks"].extend(page_info["text_blocks"])
                result["tables"].extend(page_info["tables"])
                result["images"].extend(page_info["images"])
                
                # For very large documents, release page resources after processing
                if is_large_pdf:
                    page = None
        
        # Close document to free resources
        doc.close()
        
        # Check if text extraction was successful
        if not result["text_blocks"]:
            # Don't raise an exception - create a fallback structure for image-only PDFs
            print(f"Warning: No text extracted from PDF, creating fallback structure")
            
            # Add basic metadata about the document structure
            result["is_text_extractable"] = False
            result["likely_document_type"] = "scanned" if page_count > 0 else "empty"
            
            # Try to extract images as a fallback
            self.config.extract_images = True  # Override config to extract images
            
            # Reopen document to extract images
            try:
                img_doc = fitz.open(pdf_path)
                for page_num in range(len(img_doc)):
                    page = img_doc[page_num]
                    page_info = {
                        "width": page.rect.width,
                        "height": page.rect.height,
                        "number": page_num,
                        "text_blocks": [],
                        "tables": [],
                        "images": self._extract_images(page)
                    }
                    result["pages"][page_num]["images"] = page_info["images"]
                    result["images"].extend(page_info["images"])
                img_doc.close()
            except Exception as e:
                print(f"Warning: Failed to extract images as fallback: {e}")
                
            # Update structure analysis to reflect scanned document
            result["structure_analysis"] = {
                "has_columns": False,
                "block_counts": {},
                "estimated_type": DocumentType.SCANNED.value
            }
            
            # Add placeholder for headings
            result["headings"] = []
        else:
            # Normal processing for text-containing PDFs
            result["is_text_extractable"] = True
            
        # Document-level analysis
        self.font_analysis = self._analyze_fonts(result["text_blocks"])
        
        # Apply enhanced block classification with context
        self._enhanced_block_classification(result["text_blocks"])
        
        # --- Merge fragmented lines before returning ---
        result["text_blocks"] = self._merge_fragmented_lines(result["text_blocks"], y_gap=4.0)
        
        # Complete the result with analysis data
        result.update({
            "font_analysis": self.font_analysis,
            "structure_analysis": self._analyze_structure(result["text_blocks"]),
            "stats": self._calculate_stats(result)
        })
        # Provide page_dimensions for analyzer
        result["page_dimensions"] = [
            {"width": page.get("width"), "height": page.get("height")} 
            for page in result.get("pages", [])
        ]
        result["parse_time_sec"] = round(time.time() - start_time, 3)
        
        # Additional metadata about headings for easier access
        headings = [b for b in result["text_blocks"] 
                   if b.get("type") in [BlockType.TITLE.value, BlockType.HEADING.value, BlockType.SUBHEADING.value]]
        
        result["headings"] = [{
            "text": h["text"],
            "type": h["type"],
            "page": h["page_num"],
            "bbox": h["bbox"],
            "font": h["font_name"],
            "size": h["font_size"]
        } for h in headings]
        
        return result
    
    def _process_page(self, page, page_num: int) -> Dict[str, Any]:
        """
        Process a single PDF page (optimized for speed).
        Uses fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE for faster extraction.
        """
        page_info = {
            "width": page.rect.width,
            "height": page.rect.height,
            "number": page_num,
            "text_blocks": [],
            "tables": [],
            "images": []
        }
        
        # Try fast get_text, fallback to default if it fails
        blocks = []
        try:
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE).get("blocks", [])
        except Exception:
            # Fallback to default
            blocks = page.get_text("dict").get("blocks", [])
            
        text_blocks = []
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                merged_line = self._merge_line_spans(line)
                text = self._clean_text(merged_line["text"])
                if len(text) < self.config.min_text_length:
                    continue
                text_block = {
                    **merged_line,
                    "text": text,
                    "bbox": line["bbox"],
                    "page_num": page_num,
                    "char_count": len(text),
                    "word_count": len(text.split())
                }
                text_blocks.append(text_block)
        # Post-processing with optimizations
        text_blocks = self._merge_hyphenated_words(text_blocks)
        if self.config.preserve_layout:
            text_blocks = self._detect_reading_order(text_blocks, page_info["width"])
        page_info["text_blocks"] = text_blocks
        # Extract tables and images if configured
        if self.config.detect_tables:
            page_info["tables"] = self._extract_tables(page, text_blocks)
        if self.config.extract_images:
            page_info["images"] = self._extract_images(page)
        return page_info
    
    def _merge_line_spans(self, line: Dict) -> Dict:
        """Merge text spans with formatting analysis"""
        if not line["spans"]:
            return {
                "text": "",
                "font_size": 12.0,
                "font_name": "unknown",
                "is_bold": False,
                "is_italic": False,
                "is_underlined": False,
                "color": 0,
                "superscript": False,
                "subscript": False
            }
        
        props = {
            "text_parts": [],
            "font_sizes": [],
            "font_names": [],
            "bold_flags": [],
            "italic_flags": [],
            "underline_flags": [],
            "colors": [],
            "superscript_flags": [],
            "subscript_flags": []
        }
        
        for span in line["spans"]:
            props["text_parts"].append(span["text"])
            props["font_sizes"].append(span["size"])
            props["font_names"].append(span["font"])
            
            # Enhanced formatting detection
            font_flags = span.get("flags", 0)
            font_lower = span["font"].lower()
            
            props["bold_flags"].append("bold" in font_lower or font_flags & 2**4)
            props["italic_flags"].append("italic" in font_lower or font_flags & 2**1)
            props["underline_flags"].append(font_flags & 2**2)
            props["colors"].append(span["color"])
            
            # Detect superscript/subscript
            base_size = max(props["font_sizes"]) if props["font_sizes"] else span["size"]
            size_ratio = span["size"] / base_size if base_size > 0 else 1
            props["superscript_flags"].append(size_ratio < 0.8 and font_flags & 2**0)
            props["subscript_flags"].append(size_ratio < 0.8 and font_flags & 2**3)
        
        # Merge text intelligently
        merged_text = self._merge_text_parts(props["text_parts"])
        
        # Determine dominant properties
        max_size = max(props["font_sizes"])
        max_idx = props["font_sizes"].index(max_size)
        
        return {
            "text": merged_text,
            "font_size": max_size,
            "font_name": props["font_names"][max_idx],
            "is_bold": any(props["bold_flags"]),
            "is_italic": any(props["italic_flags"]),
            "is_underlined": any(props["underline_flags"]),
            "color": props["colors"][max_idx],
            "superscript": any(props["superscript_flags"]),
            "subscript": any(props["subscript_flags"])
        }
    
    def _merge_text_parts(self, parts: List[str]) -> str:
        """Intelligently merge text with proper spacing"""
        cleaned = [re.sub(r'\s+', ' ', p.strip()) for p in parts if p.strip()]
        if not cleaned:
            return ""
            
        result = cleaned[0]
        for part in cleaned[1:]:
            # Add space between alphanumeric sequences
            if (result[-1].isalnum() and part[0].isalnum()) or \
               (result[-1].isdigit() and part[0].isalpha()) or \
               (result[-1].isalpha() and part[0].isdigit()):
                result += " " + part
            else:
                result += part
                
        return result
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text from PDF artifacts (preserve hyphens for line breaks).
        """
        # Remove hyphenation line breaks but preserve hyphen (e.g., 'co-\noperation' -> 'co-operation')
        # Remove hyphenation line breaks but preserve hyphen
        text = re.sub(r'-\s*\n\s*', '-', text)
        # Remove all other newlines and normalize whitespace
        text = re.sub(r'[\r\n]+', ' ', text)
        text = ' '.join(text.split())
        # Remove control characters
        text = ''.join(c for c in text if ord(c) >= 32 or ord(c) in {9, 10, 13})
        return text
    
    def _merge_hyphenated_words(self, blocks: List[Dict]) -> List[Dict]:
        """
        Merge words split across lines with hyphens (optimized regex, generator-based).
        """
        if not self.config.merge_hyphenated or len(blocks) < 2:
            return blocks
        merged = []
        i = 0
        n = len(blocks)
        while i < n:
            current = blocks[i]
            text = current["text"].strip()
            if (text.endswith('-') and i+1 < n and current["page_num"] == blocks[i+1]["page_num"]):
                next_block = blocks[i+1]
                next_text = next_block["text"].strip()
                # Use optimized check for hyphenated word
                if next_text and re.match(r'^[A-Za-z]', next_text):
                    merged_text = text[:-1] + next_text
                    merged_block = {
                        **current,
                        "text": merged_text,
                        "bbox": [
                            min(current["bbox"][0], next_block["bbox"][0]),
                            min(current["bbox"][1], next_block["bbox"][1]),
                            max(current["bbox"][2], next_block["bbox"][2]),
                            max(current["bbox"][3], next_block["bbox"][3])
                        ],
                        "char_count": len(merged_text),
                        "word_count": len(merged_text.split())
                    }
                    merged.append(merged_block)
                    i += 2
                    continue
            merged.append(current)
            i += 1
        return merged
    
    def _detect_reading_order(self, blocks: List[Dict], page_width: float) -> List[Dict]:
        """
        Sort blocks into reading order based on their position on the page.
        """
        # Group by approximate vertical position
        tolerance = page_width * 0.02  # 2% of page width
        rows = defaultdict(list)
        
        for block in blocks:
            y_pos = round(block["bbox"][1] / tolerance) * tolerance
            rows[y_pos].append(block)
            
        # Sort rows top to bottom
        sorted_rows = sorted(rows.items(), key=lambda x: x[0])
        ordered_blocks = []
        
        for y_pos, row_blocks in sorted_rows:
            # Sort blocks left to right within row
            row_blocks.sort(key=lambda b: b["bbox"][0])
            ordered_blocks.extend(row_blocks)
            
        return ordered_blocks
    
    def _analyze_fonts(self, blocks: List[Dict]) -> Dict:
        """Analyze font usage patterns"""
        font_stats = defaultdict(lambda: {"count": 0, "chars": 0, "examples": []})
        font_sizes = []
        
        for block in blocks:
            key = f"{block['font_name']}_{block['font_size']}_{block['is_bold']}"
            font_stats[key]["count"] += 1
            font_stats[key]["chars"] += len(block["text"])
            font_stats[key]["examples"].append(block["text"][:50])
            font_sizes.append(block["font_size"])
            
        # Identify common patterns
        sorted_fonts = sorted(font_stats.items(), key=lambda x: x[1]["chars"], reverse=True)
        body_font = sorted_fonts[0][0] if sorted_fonts else None
        common_size = Counter(font_sizes).most_common(1)[0][0] if font_sizes else 12
        
        return {
            "body_font": body_font,
            "common_size": common_size,
            "heading_threshold": common_size * 1.2,
            "font_stats": dict(font_stats)
        }
    
    def _classify_all_blocks(self, blocks: List[Dict]):
        """Classify each block's type individually"""
        if not self.config.classify_blocks:
            for block in blocks:
                block["type"] = BlockType.PARAGRAPH.value
            return
            
        for block in blocks:
            block["type"] = self._classify_block(block).value
            
    def _enhanced_block_classification(self, blocks: List[Dict]):
        """
        Enhanced classification that considers document context and structure.
        This method improves heading detection by:
        1. Analyzing document structure to understand hierarchy
        2. Considering relative positioning of blocks
        3. Applying contextual rules for heading identification
        4. Filtering out false positive headings
        """
        if not self.config.classify_blocks or not blocks:
            return
            
        # Step 1: First pass - basic classification using the regular method
        self._classify_all_blocks(blocks)
            
        # Step 2: Analyze potential heading blocks for structural patterns
        heading_blocks = [b for b in blocks if b["type"] in 
                          [BlockType.TITLE.value, BlockType.HEADING.value, BlockType.SUBHEADING.value]]
        
        # No headings found, nothing to refine
        if not heading_blocks:
            return
            
        # Step 3: Group by font properties to identify heading patterns
        font_groups = defaultdict(list)
        for block in heading_blocks:
            key = f"{block['font_name']}_{block['font_size']}_{block['is_bold']}"
            font_groups[key].append(block)
        
        # Step 4: Identify false positive headings based on contextual patterns
        false_positives = set()
        
        # Check for isolated heading-like blocks (true headings often have pattern/sequence)
        for key, group in font_groups.items():
            if len(group) == 1:  # Isolated format - potential false positive
                block = group[0]
                text = block["text"].strip()
                
                # Additional checks for isolated blocks
                if any([
                    len(text) < 5,                                  # Very short text
                    text.isdigit(),                                 # Just numbers
                    self._is_likely_paragraph_start(text),          # Starts like a paragraph
                    re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text)  # Contains date
                ]):
                    false_positives.add(id(block))
        
        # Step 5: Apply heading sequence rules (headings often have sequence patterns)
        # Sort blocks by vertical position
        sorted_blocks = sorted(blocks, key=lambda b: (b["page_num"], b["bbox"][1]))
        
        # Check sequences of blocks for consistent patterns
        for i, block in enumerate(sorted_blocks):
            if block["type"] not in [BlockType.HEADING.value, BlockType.SUBHEADING.value]:
                continue
                
            # Check preceding and following blocks for context
            prev_block = sorted_blocks[i-1] if i > 0 else None
            next_block = sorted_blocks[i+1] if i < len(sorted_blocks)-1 else None
            
            # False positive cases based on context:
            if any([
                # Isolated between paragraphs with no formatting distinction
                prev_block and next_block and 
                prev_block["type"] == BlockType.PARAGRAPH.value and 
                next_block["type"] == BlockType.PARAGRAPH.value and
                abs(block["font_size"] - prev_block["font_size"]) < 1,
                
                # Headings shouldn't be followed immediately by another heading of same level
                next_block and next_block["type"] == block["type"] and
                next_block["bbox"][1] - block["bbox"][3] < 5
            ]):
                false_positives.add(id(block))
                
        # Step 6: Reclassify false positives as paragraphs
        for block in blocks:
            if id(block) in false_positives:
                block["type"] = BlockType.PARAGRAPH.value
                
        # Step 7: Ensure proper heading hierarchy and consistency
        # This finds blocks that should be headings but were missed due to inconsistent formatting
        potential_headings = {}
        
        # Group blocks by approximate vertical position (line groups)
        y_tolerance = 5  # Approximate line height tolerance
        lines = defaultdict(list)
        
        for block in sorted_blocks:
            y_pos = round(block["bbox"][1] / y_tolerance) * y_tolerance
            lines[y_pos].append(block)
            
        # Look for heading patterns at the start of sections
        for y_pos, line_blocks in lines.items():
            if not line_blocks:
                continue
                
            # Sort blocks in this line by x-position
            line_blocks.sort(key=lambda b: b["bbox"][0])
            first_block = line_blocks[0]
            
            # Check if this might be a section start that was missed
            if (first_block["type"] == BlockType.PARAGRAPH.value and 
                len(first_block["text"]) < 50 and  # Not too long
                (first_block["is_bold"] or first_block["text"].strip().endswith(':')) and
                self._is_semantically_significant(first_block["text"]) and
                not self._is_likely_paragraph_start(first_block["text"])):
                
                # Look at the next line to check if this is the start of a section
                next_y = min([k for k in lines.keys() if k > y_pos], default=None)
                if next_y and next_y - y_pos < 20:  # Close enough to be related
                    # This is likely a missed heading
                    first_block["type"] = BlockType.SUBHEADING.value
    
    def _classify_block(self, block: Dict) -> BlockType:
        """
        Enhanced classification of text blocks with improved heading detection.
        Filters out common false positives like dates, meaningless text, and special characters.
        Always returns a valid BlockType value.
        """
        text = block["text"].strip()
        font_size = block["font_size"]
        is_bold = block["is_bold"]
        is_italic = block.get("is_italic", False)
        
        try:
            # Empty or very short - not a meaningful block
            if len(text) < 3:
                return BlockType.PARAGRAPH
            
            # ======= FALSE POSITIVE FILTERS =======
            # 1. Date/time patterns
            date_patterns = [
                r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # 01/02/2023, 1-2-23
                r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{2,4}',  # 1 Jan 2023
                r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{2,4}',  # Jan 1, 2023
                r'\d{1,2}:\d{2}\s*(?:am|pm|AM|PM)?',  # 10:30 AM
            ]
            if any(re.search(pattern, text) for pattern in date_patterns) and len(text) < 30:
                return BlockType.PARAGRAPH
                
            # 2. Special character patterns (indicators or decorative elements)
            special_char_ratio = len(re.findall(r'[^\w\s]', text)) / max(len(text), 1)
            if special_char_ratio > 0.3:  # Text contains over 30% special characters
                return BlockType.PARAGRAPH
                
            # 3. Very short texts that are capitalized or all-caps (likely labels, not headings)
            if len(text) < 10 and text.isupper() and not any(char.isdigit() for char in text):
                return BlockType.PARAGRAPH
                
            # 4. Single letters, page numbers, etc.
            if re.match(r'^[A-Z]$|^\d+$|^[A-Z]\d+$', text):
                return BlockType.PARAGRAPH
            
            # 5. Check for meaningless text patterns (*****, -----, ______, etc.)
            if re.match(r'^[\*\-\_\=\.\~\#\+]{3,}$', text):
                return BlockType.PARAGRAPH
            
            # ======= TITLE DETECTION =======
            # True titles: Prominent formatting + semantic qualities
            if ((font_size >= self.font_analysis.get("common_size", 12) * 1.8) and 
                (is_bold or text.isupper()) and 
                (3 < len(text.split()) < 15) and  # Reasonable word count for a title
                self._is_semantically_significant(text)):
                return BlockType.TITLE
                
            # ======= HEADING DETECTION =======
            # Enhanced heading detection with formatting AND semantic checks
            heading_threshold = self.font_analysis.get("heading_threshold", 14)
            common_size = self.font_analysis.get("common_size", 12)
            
            if font_size >= heading_threshold:
                # Primary check for headings: larger size + significant meaning
                if (font_size >= common_size * 1.5 and 
                    self._is_semantically_significant(text) and
                    (is_bold or text.strip().endswith(':') or self._has_heading_pattern(text))):
                    return BlockType.HEADING
                    
                # Secondary check for subheadings: slightly larger size + formatting cues
                if ((is_bold or is_italic or text.strip().endswith(':')) and
                    self._is_semantically_significant(text) and
                    not self._is_likely_paragraph_start(text)):
                    return BlockType.SUBHEADING
                    
                # Even if larger font, reject as heading if likely paragraph or meaningless
                if self._is_likely_paragraph_start(text) or not self._is_semantically_significant(text):
                    return BlockType.PARAGRAPH
                    
                return BlockType.SUBHEADING
                
            # Lists
            list_patterns = [
                r'^\s*[-•*]\s+', r'^\s*\d+\.\s+', 
                r'^\s*[a-zA-Z]\.\s+', r'^\s*[ivx]+\.\s+'
            ]
            if any(re.match(p, text) for p in list_patterns):
                return BlockType.LIST_ITEM
                
            # Default case - if nothing else matches
            return BlockType.PARAGRAPH
            
        except Exception as e:
            # Fail-safe: If any error occurs in classification, default to paragraph
            print(f"Error in block classification: {e} - defaulting to paragraph")
            return BlockType.PARAGRAPH
            
    def _is_semantically_significant(self, text: str) -> bool:
        """
        Check if text has semantic qualities expected in a heading:
        - Contains significant words (not just function words)
        - Has reasonable length for a heading
        - Doesn't contain patterns typically found in body text
        """
        text = text.strip()
        
        # Too short or too long for a typical heading
        word_count = len(text.split())
        if word_count < 2 or word_count > 20:
            return False
            
        # Check for noise patterns
        noise_patterns = [
            r'^Page \d+$',  # Page numbers
            r'^\d+$',       # Just numbers
            r'^[A-Z]$',     # Single letters
            r'^[A-Z]\d+$',  # Section numbers like A1, B2
        ]
        if any(re.match(pattern, text) for pattern in noise_patterns):
            return False
        
        # Check for common meaningless phrases
        filler_phrases = [
            'continued from previous page', 'continued on next page', 
            'all rights reserved', 'for internal use only',
            'confidential', 'draft', 'not for distribution'
        ]
        if any(phrase in text.lower() for phrase in filler_phrases):
            return False
        
        # Check if text ends with typical heading punctuation
        if text.endswith((':', '.')):
            return True
        
        # Check for presence of significant words (nouns, verbs)
        # Simple check for now: at least one word with 5+ characters
        if any(len(word) >= 5 for word in text.split()):
            return True
            
        return False
        
    def _has_heading_pattern(self, text: str) -> bool:
        """
        Check if text has structural patterns typical of headings:
        - Numbered sections (1.2, II.A, etc.)
        - ALL CAPS or Title Case
        - Question format
        """
        # Numbered section patterns
        if re.match(r'^\d+\.\d+|^[IVXLCDM]+\.[A-Z]|^[A-Z]\.\d+', text):
            return True
            
        # Check for ALL CAPS (common in headings)
        if text.isupper() and len(text) > 3:
            return True
            
        # Check for Title Case (most words capitalized)
        words = text.split()
        if len(words) >= 2:
            capitalized_ratio = sum(1 for word in words if word and word[0].isupper()) / len(words)
            if capitalized_ratio >= 0.7:  # At least 70% of words start with capital
                return True
                
        # Question headings often end with ?
        if text.endswith('?'):
            return True
            
        return False
        
    def _is_likely_paragraph_start(self, text: str) -> bool:
        """
        Check if text is likely the start of a paragraph rather than a heading:
        - Starts with lowercase
        - Contains sentence-like structures
        - Has connecting words at the start
        """
        # Check if starts with lowercase (unlikely for a heading)
        if text and text[0].islower():
            return True
            
        # Check for connecting words at the beginning
        connecting_words = ['and', 'but', 'or', 'nor', 'yet', 'so', 'for', 'because', 'although', 'however']
        if any(text.lower().startswith(word + ' ') for word in connecting_words):
            return True
            
        # Check for typical sentence structures with multiple clauses
        if text.count(',') > 1 or text.count(';') > 0:
            return True
            
        return False
            
        # Position-based elements
        y_pos = block["bbox"][1]
        if y_pos < block["bbox"][3] * 0.1:  # Top 10%
            return BlockType.HEADER
        if y_pos > block["bbox"][3] * 0.9:  # Bottom 10%
            return BlockType.FOOTER
            
        # Quotes and captions
        if (text.startswith('"') and text.endswith('"')):
            return BlockType.QUOTE
            
        caption_patterns = [
            r'^(figure|fig|table|chart)\s*\d*[:.]\s*',
            r'^(source|note):\s*'
        ]
        if any(re.match(p, text, re.I) for p in caption_patterns):
            return BlockType.CAPTION
            
        return BlockType.PARAGRAPH
    
    def _extract_tables(self, page, text_blocks: List[Dict]) -> List[Dict]:
        """
        Extract table structures from page using the configured detection method.
        Supports various detection strategies:
        - 'native': Use only PyMuPDF's built-in table detection
        - 'heuristic': Use only our enhanced heuristic detection
        - 'enhanced': Use both methods and combine results (default)
        - 'none': Disable table detection
        """
        if not self.config.detect_tables:
            return []
            
        tables = []
        detection_method = self.config.table_detection_method.lower()
        min_quality = self.config.table_min_quality
        
        # Skip if table detection is explicitly disabled
        if detection_method == "none":
            return []
            
        # Step 1: Try native table detection if requested
        if detection_method in ["native", "enhanced"]:
            try:
                if hasattr(page, 'find_tables'):
                    for table in page.find_tables():
                        table_data = {
                            "type": "table",
                            "bbox": table.bbox,
                            "rows": table.extract() if hasattr(table, 'extract') else [],
                            "column_count": table.col_count if hasattr(table, 'col_count') else 0,
                            "row_count": len(table.extract()) if hasattr(table, 'extract') else 0,
                            "detection_method": "native",
                            "confidence": "high",  # Native detection usually has high confidence
                            "quality_score": 90  # Assign high default score to native detection
                        }
                        tables.append(table_data)
            except Exception:
                pass  # Table detection failed, continue with heuristic detection
        
        # Step 2: Try heuristic detection if requested
        if detection_method in ["heuristic", "enhanced"]:
            table_rows = self._detect_table_rows(text_blocks)
            if table_rows:
                heuristic_tables = self._process_table_rows(table_rows)
                
                # Filter and add heuristic tables
                for h_table in heuristic_tables:
                    # Skip low quality tables
                    if h_table.get("quality_score", 0) < min_quality:
                        continue
                    
                    # In enhanced mode, skip if this table significantly overlaps with a native-detected table
                    if detection_method == "enhanced" and any(
                        self._calculate_bbox_overlap(h_table["bbox"], t["bbox"]) > 0.7 for t in tables
                    ):
                        continue
                    
                    # Add detection method information
                    h_table["detection_method"] = "heuristic"
                    tables.append(h_table)
        
        # Sort tables by vertical position for consistent output
        tables.sort(key=lambda t: t["bbox"][1])
        return tables
        
    def _calculate_bbox_overlap(self, bbox1, bbox2):
        """
        Calculate the intersection over union (IoU) between two bounding boxes.
        Returns a value between 0 (no overlap) and 1 (perfect overlap).
        """
        # Extract coordinates
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Calculate intersection area
        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        intersection = x_overlap * y_overlap
        
        # Calculate union area
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        # Return IoU
        return intersection / union if union > 0 else 0
    
    def _detect_table_rows(self, blocks: List[Dict]) -> List[Dict]:
        """
        Detect potential table rows based on alignment with enhanced filtering:
        - Excludes header/footer blocks
        - Applies improved alignment checks
        - Assigns quality score to each detected row
        """
        tolerance = 5  # Tolerance for Y-axis alignment
        rows = defaultdict(list)
        
        # Filter out headers and footers
        filtered_blocks = [
            block for block in blocks 
            if block.get("type") not in [BlockType.HEADER.value, BlockType.FOOTER.value]
        ]
        
        # Group blocks by approximate vertical position
        for block in filtered_blocks:
            y_pos = round(block["bbox"][1] / tolerance) * tolerance
            rows[y_pos].append(block)
        
        table_rows = []
        for y_pos, row_blocks in rows.items():
            # Need at least 2 columns to consider a table row
            if len(row_blocks) >= 2:
                row_blocks.sort(key=lambda b: b["bbox"][0])  # Sort blocks by X-position
                
                # Calculate row quality score (0-100)
                row_quality = self._calculate_row_quality(row_blocks)
                
                # Only include rows with sufficient alignment quality
                if row_quality >= 60:  # Threshold for row quality
                    table_rows.append({
                        "y_pos": y_pos,
                        "blocks": row_blocks,
                        "columns": len(row_blocks),
                        "quality": row_quality
                    })
        
        return table_rows
        
    def _calculate_row_quality(self, blocks: List[Dict]) -> float:
        """
        Calculate a quality score (0-100) for a potential table row based on multiple factors:
        - Alignment of blocks (start positions)
        - Consistency of block widths
        - Text density and formatting consistency
        - Word count similarity between blocks
        """
        if len(blocks) < 2:
            return 0
            
        # 1. Check alignment using the enhanced alignment function
        alignment_score = 0
        if self._is_roughly_aligned(blocks):
            alignment_score = 70  # Base score for good alignment
            
            # Bonus for more columns (up to +15)
            col_count_bonus = min(len(blocks) * 3, 15)
            alignment_score += col_count_bonus
        
        # 2. Check formatting consistency
        font_names = [block.get("font_name", "") for block in blocks]
        font_sizes = [block.get("font_size", 0) for block in blocks]
        bold_flags = [block.get("is_bold", False) for block in blocks]
        
        # Calculate ratios of most common attributes
        most_common_font = Counter(font_names).most_common(1)[0][1] / len(blocks) if font_names else 0
        size_variance = max(font_sizes) - min(font_sizes) if font_sizes else 0
        bold_consistency = Counter(bold_flags).most_common(1)[0][1] / len(blocks) if bold_flags else 0
        
        # Score formatting consistency (0-15)
        format_score = 15 * (0.5 * most_common_font + 0.3 * (1 - min(size_variance/5, 1)) + 0.2 * bold_consistency)
        
        # 3. Check word count consistency
        word_counts = [len(block.get("text", "").split()) for block in blocks]
        if not word_counts or max(word_counts) == 0:
            word_count_score = 0
        else:
            # More consistent word counts (relative to max) get higher scores
            avg_words = sum(word_counts) / len(word_counts)
            variance = sum((count - avg_words)**2 for count in word_counts) / len(word_counts)
            normalized_variance = min(variance / (max(word_counts)**2 + 1), 1)
            word_count_score = 15 * (1 - normalized_variance)
            
        # Combine scores with appropriate weighting
        total_score = alignment_score + format_score + word_count_score
        
        # Ensure score is in 0-100 range
        return min(max(total_score, 0), 100)
    
    def _is_roughly_aligned(self, blocks: List[Dict], x_tolerance: float = 10, 
                            width_variance_threshold: float = 0.3, alignment_ratio: float = 0.75) -> bool:
        """
        Enhanced check for column alignment that considers:
        1. X-position spacing uniformity
        2. Width consistency of blocks
        3. Tolerance for outliers (75% alignment rule)
        """
        if len(blocks) < 2:
            return False
        
        # Extract positions and widths
        x_positions = [b["bbox"][0] for b in blocks]
        widths = [b["bbox"][2] - b["bbox"][0] for b in blocks]
        
        # Calculate differences between consecutive X positions
        spacings = [x_positions[i+1] - x_positions[i] for i in range(len(x_positions) - 1)]
        
        if not spacings:
            return False
        
        # Calculate the average spacing and width
        avg_spacing = sum(spacings) / len(spacings)
        avg_width = sum(widths) / len(widths)
        
        # Count aligned columns (allowing for outliers)
        aligned_spacings = 0
        for spacing in spacings:
            if abs(spacing - avg_spacing) <= x_tolerance:
                aligned_spacings += 1
                
        # Check width consistency
        consistent_widths = 0
        for width in widths:
            # Allow width variance within threshold percentage of average width
            if abs(width - avg_width) <= (avg_width * width_variance_threshold):
                consistent_widths += 1
        
        # Calculate alignment ratios
        spacing_alignment_ratio = aligned_spacings / len(spacings) if spacings else 0
        width_consistency_ratio = consistent_widths / len(widths) if widths else 0
        
        # Combined check: Either good spacing alignment or good width consistency
        return (spacing_alignment_ratio >= alignment_ratio or 
                width_consistency_ratio >= alignment_ratio)
    
    def _process_table_rows(self, table_rows: List[Dict]) -> List[Dict]:
        """
        Process detected table rows with enhanced consolidation:
        - Consolidates vertically close rows that might be split cells
        - Uses font and formatting similarity for improved merging
        - Builds tables with improved row structure
        """
        if len(table_rows) < 2:
            return []
            
        # Sort rows by vertical position
        table_rows.sort(key=lambda r: r["y_pos"])
        
        # Step 1: Consolidate vertically close rows that might be split cells
        consolidated_rows = self._consolidate_split_rows(table_rows)
        
        # Step 2: Group consolidated rows into tables
        tables = []
        if len(consolidated_rows) < 2:
            return tables
            
        # Group rows into tables based on vertical proximity and structure similarity
        current_table = [consolidated_rows[0]]
        table_row_gap = self._estimate_table_row_gap(consolidated_rows)
        
        for i in range(1, len(consolidated_rows)):
            row = consolidated_rows[i]
            prev_row = current_table[-1]
            
            # Check if rows likely belong to the same table
            vertical_gap = row["y_pos"] - prev_row["y_pos"]
            structural_similarity = self._calculate_row_structure_similarity(prev_row, row)
            
            # Two conditions for table continuation:
            # 1. Rows are close together (within expected row gap)
            # 2. Rows have similar structure (columns align)
            if (vertical_gap <= table_row_gap * 1.5 and structural_similarity >= 0.6):
                current_table.append(row)
            else:
                # Finish current table and start a new one
                if len(current_table) >= 2:  # Minimum 2 rows for a table
                    tables.append(self._create_table_from_rows(current_table))
                current_table = [row]
        
        # Add the last table if it has enough rows
        if len(current_table) >= 2:
            tables.append(self._create_table_from_rows(current_table))
        
        return tables
        
    def _consolidate_split_rows(self, table_rows: List[Dict]) -> List[Dict]:
        """
        Consolidate rows that might be vertically split parts of the same logical row.
        Uses proximity, font similarity, and column alignment to determine what to merge.
        """
        if len(table_rows) <= 1:
            return table_rows
            
        consolidated = []
        i = 0
        
        while i < len(table_rows):
            current_row = table_rows[i]
            merge_candidates = []
            j = i + 1
            
            # Look ahead for potential rows to merge with current row
            while j < len(table_rows):
                next_row = table_rows[j]
                vertical_gap = next_row["y_pos"] - current_row["y_pos"]
                
                # Check if the next row is very close vertically (potential split cell)
                # and has fewer columns (suggesting it might be continuation text)
                if vertical_gap <= 10 and next_row["columns"] <= current_row["columns"]:
                    # Calculate if blocks in next_row are likely continuations of current_row blocks
                    if self._are_rows_continuations(current_row, next_row):
                        merge_candidates.append(next_row)
                        j += 1
                        continue
                
                # No more candidates to merge
                break
                
            if merge_candidates:
                # Merge current row with all identified candidates
                merged_row = self._merge_row_blocks(current_row, merge_candidates)
                consolidated.append(merged_row)
                i = j  # Skip the merged rows
            else:
                consolidated.append(current_row)
                i += 1
                
        return consolidated
        
    def _are_rows_continuations(self, main_row: Dict, candidate_row: Dict) -> bool:
        """
        Determine if the candidate row is likely a continuation of text in the main row.
        Checks vertical alignment, font similarity, and text characteristics.
        """
        main_blocks = main_row["blocks"]
        candidate_blocks = candidate_row["blocks"]
        
        # Check if candidate blocks generally align with main blocks horizontally
        for c_block in candidate_blocks:
            c_center_x = (c_block["bbox"][0] + c_block["bbox"][2]) / 2
            
            # Look for a main block that this candidate block might continue
            for m_block in main_blocks:
                m_left = m_block["bbox"][0]
                m_right = m_block["bbox"][2]
                
                # Check if candidate block's center falls within the horizontal span of a main block
                # and if the font properties are similar (suggesting continuation)
                if (m_left <= c_center_x <= m_right and
                    c_block.get("font_name") == m_block.get("font_name") and
                    abs(c_block.get("font_size", 0) - m_block.get("font_size", 0)) <= 1):
                    
                    # Additional checks for continuation text patterns
                    main_text = m_block.get("text", "").strip()
                    candidate_text = c_block.get("text", "").strip()
                    
                    # Check for sentence continuation patterns
                    if (not main_text.endswith('.') or 
                        candidate_text[0:1].islower() or
                        candidate_text.startswith(('and', 'or', 'but', 'nor', 'yet', 'so'))):
                        return True
                        
        return False
        
    def _merge_row_blocks(self, main_row: Dict, continuation_rows: List[Dict]) -> Dict:
        """
        Merge the blocks from main_row and continuation_rows where they align horizontally.
        """
        # Start with a copy of the main row
        merged_row = {
            "y_pos": main_row["y_pos"],
            "blocks": [],
            "columns": main_row["columns"],
            "quality": main_row.get("quality", 0)
        }
        
        # Collect all blocks
        all_blocks = main_row["blocks"].copy()
        for row in continuation_rows:
            all_blocks.extend(row["blocks"])
        
        # Group blocks by approximate horizontal alignment
        x_tolerance = 10  # Horizontal tolerance for alignment
        x_groups = defaultdict(list)
        
        for block in all_blocks:
            center_x = (block["bbox"][0] + block["bbox"][2]) / 2
            group_x = round(center_x / x_tolerance) * x_tolerance
            x_groups[group_x].append(block)
        
        # For each horizontal group, merge the blocks vertically
        for _, group_blocks in sorted(x_groups.items()):
            # Sort blocks in the group by vertical position
            group_blocks.sort(key=lambda b: b["bbox"][1])
            
            if group_blocks:
                # Start with the first block
                merged_block = group_blocks[0].copy()
                merged_text = merged_block["text"]
                
                # Merge with subsequent blocks
                for block in group_blocks[1:]:
                    # Add a space if needed between text blocks
                    if merged_text and not merged_text.endswith((' ', '-')):
                        merged_text += " "
                    merged_text += block["text"]
                    
                    # Update the bounding box to encompass all merged blocks
                    merged_block["bbox"] = [
                        min(merged_block["bbox"][0], block["bbox"][0]),
                        min(merged_block["bbox"][1], block["bbox"][1]),
                        max(merged_block["bbox"][2], block["bbox"][2]),
                        max(merged_block["bbox"][3], block["bbox"][3])
                    ]
                
                # Update text and metrics
                merged_block["text"] = merged_text
                merged_block["char_count"] = len(merged_text)
                merged_block["word_count"] = len(merged_text.split())
                
                merged_row["blocks"].append(merged_block)
        
        # Update column count
        merged_row["blocks"].sort(key=lambda b: b["bbox"][0])  # Sort by X position
        merged_row["columns"] = len(merged_row["blocks"])
        
        return merged_row
        
    def _estimate_table_row_gap(self, rows: List[Dict]) -> float:
        """
        Estimate the typical row gap in the table to help with grouping rows into tables.
        """
        if len(rows) < 2:
            return 15  # Default gap
            
        # Calculate gaps between consecutive rows
        gaps = [rows[i+1]["y_pos"] - rows[i]["y_pos"] 
                for i in range(len(rows)-1)]
        
        # Use median to be robust against outliers
        gaps.sort()
        median_gap = gaps[len(gaps)//2] if gaps else 15
        
        return median_gap
        
    def _calculate_row_structure_similarity(self, row1: Dict, row2: Dict) -> float:
        """
        Calculate structural similarity between two table rows.
        Returns a value between 0 (completely different) and 1 (identical structure).
        """
        blocks1 = row1["blocks"]
        blocks2 = row2["blocks"]
        
        # If row column counts are very different, they're likely not from the same table
        col_diff_ratio = min(len(blocks1), len(blocks2)) / max(len(blocks1), len(blocks2))
        if col_diff_ratio < 0.5:
            return 0.0
            
        # Compare horizontal alignment of blocks
        # Convert to arrays of start and end positions
        x_starts1 = [b["bbox"][0] for b in blocks1]
        x_ends1 = [b["bbox"][2] for b in blocks1]
        
        x_starts2 = [b["bbox"][0] for b in blocks2]
        x_ends2 = [b["bbox"][2] for b in blocks2]
        
        # Normalize positions relative to row width
        width1 = max(x_ends1) - min(x_starts1) if x_ends1 and x_starts1 else 1
        width2 = max(x_ends2) - min(x_starts2) if x_ends2 and x_starts2 else 1
        
        norm_starts1 = [(x - min(x_starts1)) / width1 for x in x_starts1] if width1 > 0 else []
        norm_starts2 = [(x - min(x_starts2)) / width2 for x in x_starts2] if width2 > 0 else []
        
        # Compare structure using column positions
        # Find closest matching positions between rows
        total_similarity = 0
        matches = 0
        
        # Match starts in row1 to closest starts in row2
        for pos1 in norm_starts1:
            if norm_starts2:
                closest_distance = min(abs(pos1 - pos2) for pos2 in norm_starts2)
                position_similarity = 1 - min(closest_distance, 1.0)
                total_similarity += position_similarity
                matches += 1
        
        # Calculate average similarity
        return total_similarity / matches if matches > 0 else 0.0
    
    def _has_uniform_spacing(self, positions: List[float], tolerance: float = 20) -> bool:
        """Check if positions are uniformly spaced"""
        if len(positions) < 2:
            return False
            
        spacings = [positions[i+1]-positions[i] for i in range(len(positions)-1)]
        avg = sum(spacings) / len(spacings)
        return all(abs(s-avg) <= tolerance for s in spacings)
    
    def _create_table_from_rows(self, rows: List[Dict]) -> Dict:
        """
        Create an enhanced table structure from aligned rows.
        Includes table quality metrics and more detailed structure information.
        """
        if not rows:
            return {}
            
        # Extract all blocks
        blocks = [b for row in rows for b in row["blocks"]]
        
        # Calculate bounding box
        x0 = min(b["bbox"][0] for b in blocks) if blocks else 0
        y0 = min(b["bbox"][1] for b in blocks) if blocks else 0
        x1 = max(b["bbox"][2] for b in blocks) if blocks else 0
        y1 = max(b["bbox"][3] for b in blocks) if blocks else 0
        
        # Determine column structure by analyzing horizontal alignment across rows
        column_structure = self._analyze_table_columns(rows)
        
        # Calculate overall table quality based on individual row qualities
        avg_row_quality = sum(row.get("quality", 0) for row in rows) / len(rows) if rows else 0
        structure_quality = self._evaluate_table_structure(rows)
        overall_quality = (avg_row_quality * 0.6) + (structure_quality * 0.4)
        
        return {
            "type": "table",
            "bbox": [x0, y0, x1, y1],
            "rows": [[b["text"] for b in row["blocks"]] for row in rows],
            "row_count": len(rows),
            "column_count": len(column_structure),
            "column_structure": column_structure,
            "quality_score": overall_quality,
            "confidence": "high" if overall_quality > 85 else "medium" if overall_quality > 70 else "low"
        }
    
    def _analyze_table_columns(self, rows: List[Dict]) -> List[Dict]:
        """
        Analyze column structure across all rows to identify consistent columns.
        Returns a list of column definitions with position and width information.
        """
        if not rows:
            return []
        
        # Collect all horizontal positions
        all_x_starts = []
        all_x_ends = []
        
        for row in rows:
            for block in row["blocks"]:
                all_x_starts.append(block["bbox"][0])
                all_x_ends.append(block["bbox"][2])
        
        # Find column boundaries using clustering
        x_tolerance = 10  # Tolerance for horizontal position clustering
        
        # Cluster start positions
        start_clusters = defaultdict(list)
        for x in all_x_starts:
            cluster_x = round(x / x_tolerance) * x_tolerance
            start_clusters[cluster_x].append(x)
        
        # Calculate average positions for each cluster
        column_starts = [sum(positions) / len(positions) 
                         for positions in start_clusters.values() if positions]
        column_starts.sort()
        
        # Cluster end positions
        end_clusters = defaultdict(list)
        for x in all_x_ends:
            cluster_x = round(x / x_tolerance) * x_tolerance
            end_clusters[cluster_x].append(x)
        
        # Calculate average end positions
        column_ends = [sum(positions) / len(positions) 
                      for positions in end_clusters.values() if positions]
        column_ends.sort()
        
        # Match starts with appropriate ends to define columns
        columns = []
        for i, start in enumerate(column_starts):
            # Find the closest end that's greater than this start
            valid_ends = [end for end in column_ends if end > start]
            if valid_ends:
                end = min(valid_ends)  # Take the closest valid end
            else:
                # If no valid end, use the rightmost position
                end = max(column_ends) if column_ends else start + 50
            
            columns.append({
                "index": i,
                "x_start": start,
                "x_end": end,
                "width": end - start
            })
        
        return columns
    
    def _evaluate_table_structure(self, rows: List[Dict]) -> float:
        """
        Evaluate the quality of the table structure based on:
        - Consistency of column count across rows
        - Alignment of column positions
        - Distribution of text across columns
        
        Returns a quality score between 0-100.
        """
        if not rows:
            return 0
            
        # 1. Check column count consistency
        column_counts = [row["columns"] for row in rows]
        most_common_count = Counter(column_counts).most_common(1)[0][0]
        column_consistency = sum(1 for count in column_counts if count == most_common_count) / len(column_counts)
        
        # 2. Check column alignment across rows
        alignment_scores = []
        
        # For each pair of consecutive rows, calculate alignment score
        for i in range(len(rows) - 1):
            row1 = rows[i]
            row2 = rows[i + 1]
            alignment_scores.append(self._calculate_row_structure_similarity(row1, row2))
        
        avg_alignment = sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0
        
        # 3. Check text distribution (avoid empty cells)
        cell_text_lengths = []
        for row in rows:
            for block in row["blocks"]:
                cell_text_lengths.append(len(block.get("text", "").strip()))
        
        # Calculate percentage of non-empty cells
        non_empty_ratio = sum(1 for length in cell_text_lengths if length > 0) / len(cell_text_lengths) if cell_text_lengths else 0
        
        # Combine scores with appropriate weighting
        structure_score = (
            column_consistency * 40 +  # Column consistency is very important
            avg_alignment * 40 +       # Alignment is equally important
            non_empty_ratio * 20       # Text distribution less important but still relevant
        )
        
        return min(max(structure_score, 0), 100)
    
    def _extract_images(self, page) -> List[Dict]:
        """Extract image metadata"""
        if not self.config.extract_images:
            return []
            
        images = []
        
        try:
            for img in page.get_images():
                xref = img[0]
                bbox = page.get_image_bbox(xref)
                
                # Check minimum size
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                if width < self.config.image_min_size[0] or height < self.config.image_min_size[1]:
                    continue
                    
                images.append({
                    "type": "image",
                    "xref": xref,
                    "bbox": bbox,
                    "width": width,
                    "height": height
                })
        except Exception as e:
            pass  # Image extraction failed
            
        return images
    
    def _get_metadata(self, doc) -> Dict:
        """Extract document metadata"""
        return {
            "title": doc.metadata.get("title"),
            "author": doc.metadata.get("author"),
            "subject": doc.metadata.get("subject"),
            "creator": doc.metadata.get("creator"),
            "producer": doc.metadata.get("producer"),
            "creation_date": doc.metadata.get("creationDate"),
            "mod_date": doc.metadata.get("modDate"),
            "pages": len(doc)
        }
    
    def _analyze_structure(self, blocks: List[Dict]) -> Dict:
        """Analyze document structure"""
        type_counts = Counter(b["type"] for b in blocks)
        has_columns = len(set(round(b["bbox"][0]/50)*50 for b in blocks)) > 2
        
        return {
            "has_columns": has_columns,
            "block_counts": dict(type_counts),
            "estimated_type": self._estimate_document_type(blocks)
        }
    
    def _estimate_document_type(self, blocks: List[Dict]) -> str:
        """Estimate document type based on content"""
        type_counts = Counter(b["type"] for b in blocks)
        
        if type_counts.get("table", 0) > len(blocks) * 0.2:
            return DocumentType.BUSINESS.value
        elif type_counts.get("heading", 0) > len(blocks) * 0.1:
            return DocumentType.ACADEMIC.value
        elif type_counts.get("list_item", 0) > len(blocks) * 0.15:
            return DocumentType.TECHNICAL.value
        elif any(b["text"].strip().startswith("§") for b in blocks):
            return DocumentType.LEGAL.value
        else:
            return DocumentType.UNIVERSAL.value
    
    def _calculate_stats(self, result: Dict) -> Dict:
        """Calculate extraction statistics"""
        return {
            "total_blocks": len(result["text_blocks"]),
            "total_pages": len(result["pages"]),
            "total_tables": len(result["tables"]),
            "total_images": len(result["images"]),
            "total_chars": sum(b["char_count"] for b in result["text_blocks"]),
            "total_words": sum(b["word_count"] for b in result["text_blocks"])
        }

# Preset configurations - optimized for 10-second constraint
PRESETS = {
    "universal": UniversalPDFConfig(),
    "fast": UniversalPDFConfig(
        detect_tables=False,
        extract_images=False,
        preserve_layout=False,
        merge_hyphenated=False,
        detect_columns=False,
        classify_blocks=False
    ),
    "heading_extraction": UniversalPDFConfig(
        detect_tables=False,
        extract_images=False,
        preserve_layout=True,
        merge_hyphenated=True,
        detect_columns=False,
        classify_blocks=True
    )
}

def parse_pdf(
    file_path: str, 
    config: Optional[UniversalPDFConfig] = None,
    preset: Optional[str] = None,
    fallback_on_error: bool = True
) -> Dict[str, Any]:
    """
    Universal PDF parsing function optimized for heading extraction
    with robust error handling and fallback for unreadable PDFs
    
    Args:
        file_path: Path to PDF file
        config: Custom configuration (optional)
        preset: Configuration preset name (optional)
        fallback_on_error: If True, return a fallback structure instead of raising errors
        
    Returns:
        Parsed document structure with metadata
    """
    import time
    import os
    
    start_time = time.time()
    
    try:
        # Configure parser
        if preset and preset in PRESETS:
            config = PRESETS[preset]
        elif not config:
            # Use heading_extraction preset by default for optimal performance
            config = PRESETS.get("heading_extraction", UniversalPDFConfig())
            
        # Parse document
        parser = AdvancedPDFParser(config)
        return parser.parse_pdf(file_path)
        
    except Exception as e:
        if not fallback_on_error:
            # Re-raise the exception if fallback is disabled
            raise
            
        # Create fallback document structure
        print(f"ERROR: Could not process '{os.path.basename(file_path)}' after {time.time() - start_time:.2f}s. "
              f"Reason: {str(e)}")
        print("     Creating fallback empty document JSON.")
        
        # Basic metadata
        fallback = {
            "text_blocks": [],
            "tables": [],
            "images": [],
            "pages": [],
            "headings": [],
            "metadata": {
                "title": None,
                "author": None,
                "subject": None,
                "creator": None,
                "producer": None,
                "creation_date": None,
                "mod_date": None,
                "pages": 0
            },
            "parse_time_sec": round(time.time() - start_time, 3),
            "is_text_extractable": False,
            "likely_document_type": "unknown",
            "error": str(e),
            "stats": {
                "total_blocks": 0,
                "total_pages": 0,
                "total_tables": 0,
                "total_images": 0,
                "total_chars": 0,
                "total_words": 0
            },
            "structure_analysis": {
                "has_columns": False,
                "block_counts": {},
                "estimated_type": DocumentType.SCANNED.value
            }
        }
        
        return fallback

# Example usage
if __name__ == "__main__":
    # Parse with default universal settings
    result = parse_pdf("/Users/kumarswamikallimath/Desktop/AIH_NEW/backup/input/Data-Science-for-Business-1-60.pdf")
    print(result)
    # Parse with business preset
    # business_result = parse_pdf("report.pdf", preset="business")
    
    # Parse with custom config
    custom_config = UniversalPDFConfig(
        detect_tables=True,
        extract_images=False,
        min_text_length=3
    )
    custom_result = parse_pdf("contract.pdf", config=custom_config)