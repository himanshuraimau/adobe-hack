#!/usr/bin/env python3
"""
Enhanced Semantic PDF Document Structure Extractor
Identifies titles, headings, and captions using semantic analysis
"""

import fitz  # PyMuPDF
import json
import re
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

class DocumentStructurePredictor:
    """Enhanced semantic document structure predictor"""
    
    def __init__(self):
        self.document_stats = {}
        
    def analyze_document_statistics(self, text_elements):
        """Analyze document for statistical patterns"""
        if not text_elements:
            return {}
        
        font_sizes = [elem.get('font_size', 12) for elem in text_elements]
        font_size_counts = Counter(font_sizes)
        
        # Calculate statistics
        stats = {
            'avg_font_size': sum(font_sizes) / len(font_sizes),
            'body_text_size': font_size_counts.most_common(1)[0][0],  # Most frequent size
            'max_font_size': max(font_sizes),
            'potential_heading_sizes': [],
            'total_elements': len(text_elements)
        }
        
        # Identify potential heading sizes
        body_size = stats['body_text_size']
        for size, count in font_size_counts.items():
            if size > body_size and count < len(text_elements) * 0.15:  # Larger and less frequent
                stats['potential_heading_sizes'].append(size)
        
        stats['potential_heading_sizes'] = sorted(stats['potential_heading_sizes'], reverse=True)[:4]
        self.document_stats = stats
        return stats
    
    def is_caption(self, text, element):
        """Semantic analysis to identify captions"""
        text_lower = text.lower().strip()
        
        # Caption patterns - semantic indicators
        caption_indicators = [
            # Direct caption markers
            'figure', 'fig', 'table', 'image', 'chart', 'graph', 'diagram',
            'photo', 'picture', 'illustration', 'map', 'screenshot',
            
            # Caption prefixes/suffixes  
            'caption:', 'source:', 'credit:', 'courtesy of', 'photo by',
            'image source', 'data source', 'reproduced from',
            
            # Common caption structures
            'shown above', 'shown below', 'pictured', 'depicted',
            'as seen in', 'reference:', 'adapted from'
        ]
        
        # Check for caption indicators
        has_caption_words = any(indicator in text_lower for indicator in caption_indicators)
        
        # Structural caption patterns
        starts_with_fig = bool(re.match(r'^(figure|fig|table|chart|graph|image)\\s*\\d*[:.\\s]', text_lower))
        has_parenthetical = '(' in text and ')' in text
        
        # Position-based caption detection (captions often in specific positions)
        font_size = element.get('font_size', 12)
        is_small_font = font_size < self.document_stats.get('body_text_size', 12) * 0.9
        
        # Length-based (captions are often descriptive but not too long)
        word_count = len(text.split())
        is_medium_length = 3 <= word_count <= 20
        
        # Combined caption score
        caption_score = 0
        if has_caption_words: caption_score += 0.4
        if starts_with_fig: caption_score += 0.5
        if has_parenthetical: caption_score += 0.1
        if is_small_font: caption_score += 0.2
        if is_medium_length: caption_score += 0.1
        
        return caption_score > 0.4
    
    def analyze_text_meaning(self, text, element):
        """Semantic analysis to determine text type and level"""
        text_clean = text.strip()
        text_lower = text_clean.lower()
        word_count = len(text_clean.split())
        
        # Skip if caption
        if self.is_caption(text_clean, element):
            return {'type': 'caption', 'confidence': 0.8}
        
        # Title semantic analysis
        title_indicators = {
            'document_type': ['report', 'proposal', 'study', 'analysis', 'plan', 'strategy', 'guide', 'manual'],
            'academic': ['research', 'thesis', 'dissertation', 'paper', 'journal', 'proceedings'],
            'business': ['business plan', 'proposal', 'presentation', 'overview', 'executive summary'],
            'institutional': ['university', 'college', 'department', 'organization', 'company', 'corporation']
        }
        
        # Heading level semantic analysis  
        h1_indicators = [
            'introduction', 'background', 'overview', 'executive summary', 
            'methodology', 'conclusion', 'recommendations', 'abstract',
            'chapter', 'part', 'section i', 'appendix'
        ]
        
        h2_indicators = [
            'objectives', 'goals', 'scope', 'timeline', 'budget', 'resources',
            'implementation', 'results', 'findings', 'discussion', 'analysis',
            'requirements', 'specifications', 'features', 'benefits'
        ]
        
        h3_indicators = [
            'details', 'examples', 'case study', 'sub-section', 'item',
            'component', 'element', 'aspect', 'factor', 'criterion'
        ]
        
        # Calculate semantic scores
        title_score = 0
        h1_score = 0
        h2_score = 0
        h3_score = 0
        
        # Title scoring
        for category, keywords in title_indicators.items():
            if any(keyword in text_lower for keyword in keywords):
                title_score += 0.3
        
        # Heading level scoring
        if any(keyword in text_lower for keyword in h1_indicators):
            h1_score += 0.4
        if any(keyword in text_lower for keyword in h2_indicators):
            h2_score += 0.4  
        if any(keyword in text_lower for keyword in h3_indicators):
            h3_score += 0.4
        
        # Structural analysis
        font_size = element.get('font_size', 12)
        page = element.get('page', 0)
        y_pos = element.get('y', 0)
        is_bold = element.get('is_bold', False)
        
        # Font size contribution
        if self.document_stats:
            body_size = self.document_stats.get('body_text_size', 12)
            size_ratio = font_size / body_size
            
            if size_ratio > 1.8:  # Very large
                title_score += 0.3
                h1_score += 0.2
            elif size_ratio > 1.4:  # Large
                h1_score += 0.3
                h2_score += 0.2
            elif size_ratio > 1.2:  # Medium large
                h2_score += 0.3
                h3_score += 0.2
        
        # Position contribution
        if page == 0 and y_pos > 600:  # Top of first page
            title_score += 0.4
        if page == 0:
            title_score += 0.1
            
        # Formatting contribution
        if is_bold:
            title_score += 0.1
            h1_score += 0.2
            h2_score += 0.1
            
        # Text characteristics
        if text_clean.isupper() and word_count <= 8:
            title_score += 0.2
            h1_score += 0.1
        elif text_clean.istitle():
            title_score += 0.1
            
        # Length considerations
        if 2 <= word_count <= 10:  # Good title/heading length
            title_score += 0.2
            h1_score += 0.1
            h2_score += 0.1
        elif word_count > 15:  # Too long for title
            title_score -= 0.3
            
        # Determine best type
        scores = {
            'title': title_score,
            'h1': h1_score, 
            'h2': h2_score,
            'h3': h3_score
        }
        
        best_type = max(scores, key=scores.get)
        confidence = scores[best_type]
        
        return {
            'type': best_type,
            'confidence': confidence,
            'scores': scores
        }
    
    def predict_title(self, text_elements):
        """Semantic-based title prediction with caption filtering"""
        self.analyze_document_statistics(text_elements)
        
        first_page_elements = [elem for elem in text_elements if elem.get('page', 0) == 0]
        if not first_page_elements:
            return ""
        
        title_candidates = []
        
        for elem in first_page_elements:
            text = elem.get('text', '').strip()
            if not text or len(text) < 3:
                continue
                
            # Semantic analysis
            analysis = self.analyze_text_meaning(text, elem)
            
            # Skip captions
            if analysis['type'] == 'caption':
                continue
                
            # Consider for title if high confidence
            if analysis['type'] == 'title' and analysis['confidence'] > 0.3:
                title_candidates.append({
                    'element': elem,
                    'text': text,
                    'confidence': analysis['confidence'],
                    'y_position': elem.get('y', 0)
                })
        
        if not title_candidates:
            return ""
        
        # Sort by confidence and position
        title_candidates.sort(key=lambda x: (x['confidence'], x['y_position']), reverse=True)
        
        best_candidate = title_candidates[0]
        
        # Final validation - check for invitation/event documents
        best_text = best_candidate['text']
        if any(word in best_text.lower() for word in 
              ['invitation', 'party', 'event', 'hope', 'see you there', 'rsvp']):
            return ""
        
        # Length check
        if len(best_text) > 150:
            words = best_text[:100].split()
            best_text = " ".join(words[:-1]) if len(words) > 1 else words[0] if words else ""
        
        return best_text.strip()
    
    def predict_headings(self, text_elements, exclude_title_text=None):
        """Semantic-based heading prediction with caption filtering"""
        self.analyze_document_statistics(text_elements)
        
        headings = []
        
        for element in text_elements:
            text = element.get('text', '').strip()
            if not text or len(text) > 150:
                continue
                
            # Skip title text (mutual exclusion)
            if exclude_title_text and text == exclude_title_text:
                continue
                
            # Semantic analysis
            analysis = self.analyze_text_meaning(text, element)
            
            # Skip captions and low confidence items
            if analysis['type'] == 'caption' or analysis['confidence'] < 0.4:
                continue
            
            # Accept heading types with sufficient confidence
            if analysis['type'] in ['h1', 'h2', 'h3']:
                level_str = analysis['type'].upper()
                
                # Additional validation
                word_count = len(text.split())
                if word_count > 20 or word_count < 1:  # Too long or too short
                    continue
                    
                # Skip common non-heading patterns
                if any(pattern in text.lower() for pattern in 
                      ['www.', 'http', '.com', '@', 'phone:', 'email:', 'copyright']):
                    continue
                
                headings.append({
                    'level': level_str,
                    'text': text,
                    'page': element.get('page', 0),
                    'confidence': analysis['confidence'],
                    'y': element.get('y', 0)
                })
        
        # Sort by page and position
        headings.sort(key=lambda x: (x['page'], -x['y']))
        
        # Remove confidence field for final output and limit H1 headings
        final_headings = []
        h1_count = 0
        
        for heading in headings:
            if heading['level'] == 'H1':
                h1_count += 1
                if h1_count > 5:  # Limit H1 headings
                    heading['level'] = 'H2'
                    
            final_headings.append({
                'level': heading['level'],
                'text': heading['text'],
                'page': heading['page']
            })
        
        return final_headings

def extract_text_elements_for_ml(pdf_path):
    """Extract text elements with sentence-aware processing"""
    doc = fitz.open(pdf_path)
    elements = []
    
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        
        # Collect all text spans with their properties
        all_spans = []
        for block in blocks:
            for line in block.get("lines", []):
                line_spans = line.get("spans", [])
                for span in line_spans:
                    span_text = span["text"].strip()
                    if span_text:
                        all_spans.append({
                            'text': span_text,
                            'font_size': span["size"],
                            'font_name': span.get("font", ""),
                            'is_bold': 'bold' in span.get("font", "").lower(),
                            'bbox': span["bbox"],
                            'page': page_num
                        })
        
        if not all_spans:
            continue
            
        # Sort spans by position (top to bottom, left to right)
        all_spans.sort(key=lambda x: (-x['bbox'][1], x['bbox'][0]))
        
        # Group spans into text blocks
        current_block = []
        current_y = None
        
        for span in all_spans:
            bbox = span['bbox']
            y_pos = bbox[1]
            
            # Start new block if significant vertical gap
            if current_y is None or abs(y_pos - current_y) > 5:
                if current_block:
                    # Process current block
                    block_element = merge_text_spans(current_block, page_num)
                    if block_element:
                        elements.append(block_element)
                current_block = []
            
            current_block.append(span)
            current_y = y_pos
        
        # Don't forget the last block
        if current_block:
            block_element = merge_text_spans(current_block, page_num)
            if block_element:
                elements.append(block_element)
    
    doc.close()
    
    # Merge sentence fragments
    merged_elements = merge_sentence_fragments(elements)
    
    return merged_elements

def merge_text_spans(spans, page_num):
    """Merge a group of spans into a single text element"""
    if not spans:
        return None
    
    # Combine text from all spans
    combined_text = ""
    total_font_size = 0
    font_names = []
    is_bold = False
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = 0, 0
    
    for span in spans:
        # Add space between spans if needed
        if combined_text and not combined_text.endswith(' ') and not span['text'].startswith(' '):
            combined_text += " "
        combined_text += span['text']
        
        total_font_size += span['font_size']
        font_names.append(span['font_name'])
        is_bold = is_bold or span['is_bold']
        
        bbox = span['bbox']
        min_x = min(min_x, bbox[0])
        min_y = min(min_y, bbox[1])
        max_x = max(max_x, bbox[2])
        max_y = max(max_y, bbox[3])
    
    combined_text = combined_text.strip()
    if len(combined_text) < 2:
        return None
    
    avg_font_size = total_font_size / len(spans)
    primary_font = max(set(font_names), key=font_names.count) if font_names else ""
    
    return {
        'text': combined_text,
        'font_size': avg_font_size,
        'font_name': primary_font,
        'is_bold': is_bold,
        'page': page_num,
        'x': min_x,
        'y': max_y,  # Use max_y for proper top-to-bottom sorting
        'is_left_aligned': min_x < 100,
        'is_top_area': max_y > 600,  # Adjust for PDF coordinate system
    }

def merge_sentence_fragments(elements):
    """Merge fragmented text elements that belong to the same sentence"""
    if not elements:
        return elements
    
    merged = []
    i = 0
    
    while i < len(elements):
        current = elements[i]
        current_text = current['text']
        
        # Look ahead to see if we should merge with next elements
        j = i + 1
        candidates_to_merge = [current]
        
        while j < len(elements) and j < i + 3:  # Limit lookahead
            next_elem = elements[j]
            next_text = next_elem['text']
            
            # Check if these elements should be merged
            should_merge = False
            
            # Same page and similar properties
            if (current['page'] == next_elem['page'] and
                abs(current['font_size'] - next_elem['font_size']) <= 1 and
                current['is_bold'] == next_elem['is_bold']):
                
                # Check if current text looks incomplete
                if (not current_text.endswith(('.', '!', '?', ':')) and
                    len(current_text.split()) < 8):  # Short fragments are more likely incomplete
                    
                    # Check if the next text could be a continuation
                    if (not next_text[0].isupper() or  # Doesn't start with capital
                        len(next_text.split()) < 5 or    # Also short
                        next_text.startswith(('for', 'of', 'to', 'in', 'on', 'at', 'by'))):  # Common continuation words
                        should_merge = True
            
            if should_merge:
                candidates_to_merge.append(next_elem)
                current_text = current_text + " " + next_text
                j += 1
            else:
                break
        
        # Create merged element
        if len(candidates_to_merge) > 1:
            # Merge multiple elements
            merged_text = " ".join(elem['text'] for elem in candidates_to_merge).strip()
            # Clean up multiple spaces
            merged_text = " ".join(merged_text.split())
            
            merged_element = {
                'text': merged_text,
                'font_size': candidates_to_merge[0]['font_size'],  # Use first element's properties
                'font_name': candidates_to_merge[0]['font_name'],
                'is_bold': candidates_to_merge[0]['is_bold'],
                'page': candidates_to_merge[0]['page'],
                'x': candidates_to_merge[0]['x'],
                'y': candidates_to_merge[0]['y'],
                'is_left_aligned': candidates_to_merge[0]['is_left_aligned'],
                'is_top_area': candidates_to_merge[0]['is_top_area'],
            }
            merged.append(merged_element)
            i = j
        else:
            # Keep as single element
            merged.append(current)
            i += 1
    
    return merged

def process_pdf_with_ml(pdf_path):
    """Process PDF using semantic model with mutual exclusion"""
    # Extract text elements
    text_elements = extract_text_elements_for_ml(pdf_path)
    
    # Initialize and use predictor
    predictor = DocumentStructurePredictor()
    
    # First predict title
    title = predictor.predict_title(text_elements)
    
    # Now predict headings, excluding the title text
    outline = predictor.predict_headings(text_elements, exclude_title_text=title)
    
    return {
        "title": title,
        "outline": outline
    }

# -------- MAIN EXECUTION -------- #
def process_pdfs():
    root = Path(__file__).resolve().parent
    input_dir = root / "Datasets" / "Pdfs"
    output_dir = root / "Datasets" / "output"

    print(f"🤖 Starting semantic PDF processing...")
    print(f"🔍 Looking for PDFs in: {input_dir}")
    print(f"📤 Will write JSONs to: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = list(input_dir.glob("*.pdf"))
    print(f"📄 Found {len(pdf_files)} PDFs: {[p.name for p in pdf_files]}")

    for pdf_file in pdf_files:
        try:
            print(f"⚙️  Processing: {pdf_file.name}")
            data = process_pdf_with_ml(str(pdf_file))
            output_file = output_dir / f"{pdf_file.stem}.json"
            
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✅ Done: {output_file.name}")
            
        except Exception as e:
            print(f"❌ Error processing {pdf_file.name}: {e}")

if __name__ == "__main__":
    process_pdfs()
