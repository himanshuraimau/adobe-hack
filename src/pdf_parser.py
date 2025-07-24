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
    table_detection_method: str = "none"  # Disabled for speed
    image_min_size: tuple = (10, 10)
    adaptive_processing: bool = False  # Disabled for speed

class AdvancedPDFParser:
    """Universal PDF parser optimized for all document types"""
    
    def __init__(self, config: UniversalPDFConfig = None):
        self.config = config if config else UniversalPDFConfig()
        self.font_analysis = {}
        self.layout_analysis = {}
        
    def _merge_fragmented_lines(self, text_blocks: List[Dict]) -> List[Dict]:
        # Merge adjacent text blocks on the same vertical line (y0/y1 overlap, same page, similar font)
        if not text_blocks:
            return []
        merged = []
        i = 0
        while i < len(text_blocks):
            current = text_blocks[i]
            group = [current]
            j = i + 1
            while j < len(text_blocks):
                next_block = text_blocks[j]
                # Same page, vertical overlap, similar font size and name
                if (next_block['page_num'] == current['page_num'] and
                    abs(next_block['bbox'][1] - group[-1]['bbox'][3]) < 2 and
                    abs(next_block['font_size'] - current['font_size']) < 1 and
                    next_block['font_name'] == current['font_name'] and
                    next_block['is_bold'] == current['is_bold']):
                    group.append(next_block)
                    j += 1
                else:
                    break
            if len(group) > 1:
                merged_text = ' '.join([b['text'] for b in group])
                merged_block = group[0].copy()
                merged_block['text'] = merged_text
                merged_block['bbox'] = [
                    min(b['bbox'][0] for b in group),
                    min(b['bbox'][1] for b in group),
                    max(b['bbox'][2] for b in group),
                    max(b['bbox'][3] for b in group)
                ]
                merged.append(merged_block)
                i += len(group)
            else:
                merged.append(current)
                i += 1
        return merged

    def parse_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Main parsing method"""
        try:
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
        
        for page_num, page in enumerate(doc):
            page_info = self._process_page(page, page_num)
            result["pages"].append(page_info)
            result["text_blocks"].extend(page_info["text_blocks"])
            result["tables"].extend(page_info["tables"])
            result["images"].extend(page_info["images"])
        
        doc.close()
        
        if not result["text_blocks"]:
            raise ValueError("No text extracted from PDF")
        
        # Document-level analysis
        self.font_analysis = self._analyze_fonts(result["text_blocks"])
        self._classify_all_blocks(result["text_blocks"])
        # --- Merge fragmented lines before returning ---
        result["text_blocks"] = self._merge_fragmented_lines(result["text_blocks"])
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
        return result
    
    def _process_page(self, page, page_num: int) -> Dict[str, Any]:
        """Process a single PDF page"""
        page_info = {
            "width": page.rect.width,
            "height": page.rect.height,
            "number": page_num,
            "text_blocks": [],
            "tables": [],
            "images": []
        }
        
        # Extract text blocks with enhanced processing
        blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT).get("blocks", [])
        text_blocks = []
        
        for block in blocks:
            if "lines" not in block:
                continue
                
            for line in block["lines"]:
                if not line["spans"]:
                    continue
                    
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
        """Clean extracted text from PDF artifacts"""
        # Remove hyphenation artifacts
        text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove control characters
        text = ''.join(c for c in text if ord(c) >= 32 or ord(c) in {9, 10, 13})
        return text
    
    def _merge_hyphenated_words(self, blocks: List[Dict]) -> List[Dict]:
        """Merge words split across lines with hyphens"""
        if not self.config.merge_hyphenated or len(blocks) < 2:
            return blocks
            
        merged = []
        i = 0
        
        while i < len(blocks):
            current = blocks[i]
            text = current["text"].strip()
            
            if (text.endswith('-') and i+1 < len(blocks) and 
                current["page_num"] == blocks[i+1]["page_num"]):
                
                next_block = blocks[i+1]
                next_text = next_block["text"].strip()
                
                if next_text and next_text[0].isalpha():
                    # Merge the blocks
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
        """Determine proper reading order based on layout"""
        if not self.config.preserve_layout:
            return blocks
            
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
        """Classify each block's type"""
        if not self.config.classify_blocks:
            for block in blocks:
                block["type"] = BlockType.PARAGRAPH.value
            return
            
        for block in blocks:
            block["type"] = self._classify_block(block).value
    
    def _classify_block(self, block: Dict) -> BlockType:
        """Determine block type based on content and formatting"""
        text = block["text"].strip()
        font_size = block["font_size"]
        is_bold = block["is_bold"]
        
        # Empty or very short
        if len(text) < 3:
            return BlockType.PARAGRAPH
            
        # Title detection
        if (font_size >= self.font_analysis["common_size"] * 1.8 and 
            len(text) < 100 and is_bold):
            return BlockType.TITLE
            
        # Headings
        if font_size >= self.font_analysis["heading_threshold"]:
            if font_size >= self.font_analysis["common_size"] * 1.5:
                return BlockType.HEADING
            return BlockType.SUBHEADING
            
        # Lists
        list_patterns = [
            r'^\s*[-•*]\s+', r'^\s*\d+\.\s+', 
            r'^\s*[a-zA-Z]\.\s+', r'^\s*[ivx]+\.\s+'
        ]
        if any(re.match(p, text) for p in list_patterns):
            return BlockType.LIST_ITEM
            
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
        """Extract table structures from page"""
        if not self.config.detect_tables:
            return []
            
        tables = []
        
        try:
            # Try native table detection first
            if hasattr(page, 'find_tables'):
                for table in page.find_tables():
                    table_data = {
                        "type": "table",
                        "bbox": table.bbox,
                        "rows": table.extract() if hasattr(table, 'extract') else [],
                        "columns": table.col_count if hasattr(table, 'col_count') else 0
                    }
                    tables.append(table_data)
        except Exception as e:
            pass  # Native table detection failed
            
        # Fallback to alignment-based detection
        if not tables and len(text_blocks) >= 4:  # Minimum for table
            tables.extend(self._detect_tables_by_alignment(text_blocks))
            
        return tables
    
    def _detect_tables_by_alignment(self, blocks: List[Dict]) -> List[Dict]:
        """Detect tables based on text alignment patterns"""
        # Group by rows
        tolerance = 5
        rows = defaultdict(list)
        
        for block in blocks:
            y_pos = round(block["bbox"][1] / tolerance) * tolerance
            rows[y_pos].append(block)
            
        # Find potential table rows
        table_rows = []
        for y_pos, row_blocks in rows.items():
            if len(row_blocks) >= 2:  # At least 2 columns
                row_blocks.sort(key=lambda b: b["bbox"][0])
                x_positions = [b["bbox"][0] for b in row_blocks]
                
                if self._has_uniform_spacing(x_positions):
                    table_rows.append({
                        "y_pos": y_pos,
                        "blocks": row_blocks,
                        "columns": len(row_blocks)
                    })
                    
        # Group consecutive rows into tables
        tables = []
        if table_rows:
            table_rows.sort(key=lambda r: r["y_pos"])
            current_table = [table_rows[0]]
            
            for row in table_rows[1:]:
                if abs(row["y_pos"] - current_table[-1]["y_pos"]) <= tolerance * 2:
                    current_table.append(row)
                else:
                    if len(current_table) >= 2:  # Minimum 2 rows
                        tables.append(self._create_table_from_rows(current_table))
                    current_table = [row]
                    
            if len(current_table) >= 2:
                tables.append(self._create_table_from_rows(current_table))
                
        return tables
    
    def _has_uniform_spacing(self, positions: List[float], tolerance: float = 20) -> bool:
        """Check if positions are uniformly spaced"""
        if len(positions) < 2:
            return False
            
        spacings = [positions[i+1]-positions[i] for i in range(len(positions)-1)]
        avg = sum(spacings) / len(spacings)
        return all(abs(s-avg) <= tolerance for s in spacings)
    
    def _create_table_from_rows(self, rows: List[Dict]) -> Dict:
        """Create table structure from aligned rows"""
        bboxes = [b for row in rows for b in row["blocks"]]
        x0 = min(b["bbox"][0] for b in bboxes)
        y0 = min(b["bbox"][1] for b in bboxes)
        x1 = max(b["bbox"][2] for b in bboxes)
        y1 = max(b["bbox"][3] for b in bboxes)
        
        return {
            "type": "table",
            "bbox": [x0, y0, x1, y1],
            "rows": [[b["text"] for b in row["blocks"]] for row in rows],
            "columns": rows[0]["columns"] if rows else 0
        }
    
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
    preset: Optional[str] = None
) -> Dict[str, Any]:
    """
    Universal PDF parsing function optimized for heading extraction
    
    Args:
        file_path: Path to PDF file
        config: Custom configuration (optional)
        preset: Configuration preset name (optional)
        
    Returns:
        Parsed document structure with metadata
    """
    if preset and preset in PRESETS:
        config = PRESETS[preset]
    elif not config:
        # Use heading_extraction preset by default for optimal performance
        config = PRESETS.get("heading_extraction", UniversalPDFConfig())
        
    parser = AdvancedPDFParser(config)
    return parser.parse_pdf(file_path)

# Example usage
if __name__ == "__main__":
    # Parse with default universal settings
    result = parse_pdf("document.pdf")
    
    # Parse with business preset
    business_result = parse_pdf("report.pdf", preset="business")
    
    # Parse with custom config
    custom_config = UniversalPDFConfig(
        detect_tables=True,
        extract_images=False,
        min_text_length=3
    )
    custom_result = parse_pdf("contract.pdf", config=custom_config)