"""PDF text extraction and section identification module."""

import fitz  # PyMuPDF
import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class TextBlock:
    """Represents a text block with layout information."""
    text: str
    font_size: float
    font_name: str
    bbox: Tuple[float, float, float, float]
    page: int


@dataclass
class Section:
    """Represents a document section with header and content."""
    document: str
    section_title: str
    content: str
    page_number: int
    font_info: Dict[str, Any]


class PDFAnalyzer:
    """Handles PDF parsing, sectioning, and content extraction."""
    
    def __init__(self):
        self.font_size_threshold_multiplier = 1.2
        
    def extract_text_blocks(self, pdf_path: str) -> List[TextBlock]:
        """Extract text blocks with layout information using PyMuPDF."""
        doc = fitz.open(pdf_path)
        all_blocks = []
        
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if block["type"] == 0:  # text block
                    for line in block["lines"]:
                        for span in line["spans"]:
                            if span["text"].strip():  # Skip empty text
                                all_blocks.append(TextBlock(
                                    text=span["text"].strip(),
                                    font_size=span["size"],
                                    font_name=span["font"],
                                    bbox=span["bbox"],
                                    page=page_num + 1
                                ))
        
        doc.close()
        return all_blocks
    
    def _calculate_median_font_size(self, blocks: List[TextBlock]) -> float:
        """Calculate median font size to identify headers."""
        font_sizes = [block.font_size for block in blocks if len(block.text) > 3]
        if not font_sizes:
            return 12.0
        
        font_sizes.sort()
        n = len(font_sizes)
        if n % 2 == 0:
            return (font_sizes[n//2 - 1] + font_sizes[n//2]) / 2
        else:
            return font_sizes[n//2]
    
    def _is_header(self, block: TextBlock, median_font_size: float) -> bool:
        """Determine if a text block is likely a header."""
        # Skip very short text (likely fragments)
        if len(block.text.strip()) < 3:
            return False
            
        # Font size heuristic - must be significantly larger
        if block.font_size > median_font_size * self.font_size_threshold_multiplier:
            return True
        
        # Font weight heuristic (check for bold in font name)
        if any(keyword in block.font_name.lower() for keyword in ['bold', 'black', 'heavy']):
            # Additional check: should be reasonable length for a header
            if 5 <= len(block.text) <= 150:
                return True
        
        # Pattern-based heuristics for common header patterns
        text = block.text.strip()
        
        # Numbered sections (1., 2.1, etc.)
        if re.match(r'^\d+\.?\d*\.?\s+[A-Z]', text):
            return True
            
        # All caps headers (but not too long)
        if text.isupper() and 5 <= len(text) <= 80:
            return True
            
        # Title case with reasonable length
        if text.istitle() and 10 <= len(text) <= 100:
            return True
            
        return False
    
    def identify_sections(self, blocks: List[TextBlock], document_name: str) -> List[Section]:
        """Group text blocks into logical sections based on headers."""
        if not blocks:
            return []
        
        median_font_size = self._calculate_median_font_size(blocks)
        sections = []
        current_section_title = "Introduction"
        current_section_content = []
        current_page = blocks[0].page
        
        for i, block in enumerate(blocks):
            if self._is_header(block, median_font_size):
                # Save previous section if it has content
                if current_section_content:
                    sections.append(Section(
                        document=document_name,
                        section_title=current_section_title,
                        content=" ".join(current_section_content),
                        page_number=current_page,
                        font_info={"median_font_size": median_font_size}
                    ))
                
                # Start new section
                current_section_title = block.text
                current_section_content = []
                current_page = block.page
            else:
                # Add to current section content
                current_section_content.append(block.text)
                if not current_section_content or block.page < current_page:
                    current_page = block.page
        
        # Add final section
        if current_section_content:
            sections.append(Section(
                document=document_name,
                section_title=current_section_title,
                content=" ".join(current_section_content),
                page_number=current_page,
                font_info={"median_font_size": median_font_size}
            ))
        
        return sections
    
    def get_section_content(self, pdf_path: str) -> List[Section]:
        """Extract and return structured section data from a PDF."""
        document_name = pdf_path.split('/')[-1]  # Get filename
        blocks = self.extract_text_blocks(pdf_path)
        sections = self.identify_sections(blocks, document_name)
        return sections