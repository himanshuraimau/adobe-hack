"""PDF text extraction and section identification module."""

import fitz  # PyMuPDF
import re
import logging
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
        self.logger = logging.getLogger(__name__)
        
    def extract_text_blocks(self, pdf_path: str) -> List[TextBlock]:
        """Extract text blocks with layout information using PyMuPDF."""
        if not pdf_path:
            raise ValueError("PDF path cannot be empty")
        
        doc = None
        try:
            # Attempt to open the PDF
            doc = fitz.open(pdf_path)
            
            if doc.is_encrypted:
                self.logger.warning(f"PDF is encrypted: {pdf_path}")
                # Try to authenticate with empty password
                if not doc.authenticate(""):
                    raise RuntimeError(f"Cannot decrypt PDF: {pdf_path}")
            
            if doc.page_count == 0:
                self.logger.warning(f"PDF has no pages: {pdf_path}")
                return []
            
            all_blocks = []
            failed_pages = 0
            
            for page_num in range(doc.page_count):
                try:
                    page = doc.load_page(page_num)
                    
                    # Extract text with error handling
                    try:
                        text_dict = page.get_text("dict")
                        blocks = text_dict.get("blocks", [])
                    except Exception as e:
                        self.logger.warning(f"Failed to extract text from page {page_num + 1} in {pdf_path}: {e}")
                        failed_pages += 1
                        continue
                    
                    # Process text blocks
                    for block in blocks:
                        if block.get("type") == 0:  # text block
                            for line in block.get("lines", []):
                                for span in line.get("spans", []):
                                    text = span.get("text", "").strip()
                                    if text:  # Skip empty text
                                        try:
                                            all_blocks.append(TextBlock(
                                                text=text,
                                                font_size=span.get("size", 12.0),
                                                font_name=span.get("font", "unknown"),
                                                bbox=span.get("bbox", (0, 0, 0, 0)),
                                                page=page_num + 1
                                            ))
                                        except Exception as e:
                                            self.logger.debug(f"Error creating TextBlock from span: {e}")
                                            continue
                
                except Exception as e:
                    self.logger.warning(f"Error processing page {page_num + 1} in {pdf_path}: {e}")
                    failed_pages += 1
                    continue
            
            if failed_pages > 0:
                self.logger.warning(f"Failed to process {failed_pages}/{doc.page_count} pages in {pdf_path}")
            
            if not all_blocks:
                self.logger.warning(f"No text blocks extracted from {pdf_path}")
            else:
                self.logger.debug(f"Extracted {len(all_blocks)} text blocks from {pdf_path}")
            
            return all_blocks
            
        except FileNotFoundError:
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        except PermissionError:
            raise PermissionError(f"Permission denied accessing PDF: {pdf_path}")
        except Exception as e:
            if "cannot open" in str(e).lower():
                raise RuntimeError(f"Cannot open PDF file (possibly corrupted): {pdf_path}")
            else:
                raise RuntimeError(f"Error extracting text from PDF {pdf_path}: {e}")
        finally:
            if doc:
                try:
                    doc.close()
                except:
                    pass
    
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