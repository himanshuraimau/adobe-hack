import os
import fitz  # PyMuPDF
import time
import concurrent.futures
from multiprocessing import cpu_count

def is_likely_scanned(page):
    """
    Detect if a page is likely scanned by analyzing its content.
    Returns True if the page has characteristics of a scanned document.
    """
    # Check for text content
    text = page.get_text().strip()
    if not text:
        # No text found, likely a scanned page or image
        return True
        
    # Check for images
    image_list = page.get_images()
    if len(image_list) > 0:
        # Has images but no text - likely scanned
        if len(text) < 10:
            return True
            
    # Additional check: If text is present but very sparse compared to page size
    # (indicating potential OCR artifacts), treat as scanned
    if text and len(text) < 100 and (page.rect.width * page.rect.height) > 200000:
        # Less than 100 chars on a large page
        return True
        
    return False




def analyze_page(args):
    """
    Analyze a single page for parallel processing.
    Args:
        args: Tuple containing (doc, page_num)
    Returns:
        Dictionary with page analysis results
    """
    doc, page_num = args
    
    # We need to get the page from the document
    page = doc[page_num]
    
    # Check if page is likely scanned
    page_is_scanned = is_likely_scanned(page)
    
    # For pages that appear to be scanned, analyze image content
    image_analysis = {}
    if page_is_scanned:
        image_analysis = analyze_image_content(page)
        
    return {
        "page_number": page_num,
        "is_scanned": page_is_scanned,
        **image_analysis
    }

def detect_scanned_documents(pdf_path):
    """
    Analyze a PDF to determine if it's a scanned document and provide
    recommendations for OCR processing.
    """
    try:
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        
        if page_count == 0:
            return {
                "is_scanned": False,
                "empty_document": True,
                "ocr_recommended": False
            }
            
        scanned_pages = 0
        pages_analysis = []
        
        # Analyze each page
        for page_num in range(min(page_count, 10)):  # Check first 10 pages max
            page = doc[page_num]
            
            # Check if page is likely scanned
            page_is_scanned = is_likely_scanned(page)
            
            # For pages that appear to be scanned, analyze image content
            image_analysis = {}
            if page_is_scanned:
                scanned_pages += 1
                image_analysis = analyze_image_content(page)
                
            pages_analysis.append({
                "page_number": page_num,
                "is_scanned": page_is_scanned,
                **image_analysis
            })
            
        # Overall document assessment
        is_scanned_doc = scanned_pages > 0 and (scanned_pages / min(page_count, 10)) > 0.5
        
        # Determine if OCR is recommended
        ocr_recommended = is_scanned_doc and any(
            p.get("contains_text_in_images", False) for p in pages_analysis
        )
        
        result = {
            "is_scanned": is_scanned_doc,
            "scanned_page_ratio": scanned_pages / min(page_count, 10),
            "total_pages": page_count,
            "pages_analyzed": min(page_count, 10),
            "pages_scanned": scanned_pages,
            "page_analysis": pages_analysis,
            "ocr_recommended": ocr_recommended
        }
        
        doc.close()
        return result
        
    except Exception as e:
        return {
            "error": str(e),
            "is_scanned": True,  # Assume scanned if analysis fails
            "ocr_recommended": True
        }
