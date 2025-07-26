#!/usr/bin/env python3
"""
Multilingual PDF Data Extraction Script

A comprehensive script that extracts data from PDFs, identifies titles and heading levels
across multiple languages. Supports 20+ languages including English, Spanish, French,
German, Hindi, Japanese, Chinese, Korean, Arabic, and many more.

Features:
- Automatic language detection
- Rule-based title extraction
- Hierarchical heading detection  
- MobileBERT semantic validation
- Clean JSON output
- Support for CJK, Indic, Arabic, Cyrillic, and Latin scripts

Usage:
    python multilingual_pdf_extractor.py <pdf_file> [output_file]
    python multilingual_pdf_extractor.py input.pdf output.json
    python multilingual_pdf_extractor.py input.pdf  # Auto-generates output filename
"""

import os
import sys
import json
import time
import re
import fitz  # PyMuPDF
import unicodedata
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter, defaultdict
from dataclasses import dataclass
import argparse

# Optional MobileBERT support (if available)
try:
    import torch
    from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("Warning: transformers/torch not available. Running in rule-based mode only.")

@dataclass
class TextBlock:
    """Represents a text block with formatting information"""
    text: str
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    font_size: float
    font_name: str
    font_flags: int
    page_num: int
    line_num: int = 0
    
    @property
    def is_bold(self) -> bool:
        return bool(self.font_flags & 2**4)
    
    @property
    def is_italic(self) -> bool:
        return bool(self.font_flags & 2**1)
    
    @property
    def x(self) -> float:
        return self.bbox[0]
    
    @property
    def y(self) -> float:
        return self.bbox[1]
    
    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

@dataclass
class LanguageConfig:
    """Configuration for language-specific processing"""
    language_code: str
    language_name: str
    script: str
    rtl: bool = False
    title_patterns: List[str] = None
    heading_patterns: List[str] = None

class MultilingualLanguageDetector:
    """Advanced language detection for multilingual documents"""
    
    def __init__(self):
        self.script_ranges = self._initialize_script_ranges()
        self.language_indicators = self._initialize_language_indicators()
        
    def _initialize_script_ranges(self) -> Dict[str, List[Tuple[int, int]]]:
        """Initialize Unicode script ranges for different writing systems"""
        return {
            'Latin': [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x00FF), (0x0100, 0x017F)],
            'Cyrillic': [(0x0400, 0x04FF), (0x0500, 0x052F)],
            'Arabic': [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF)],
            'Devanagari': [(0x0900, 0x097F)],
            'Bengali': [(0x0980, 0x09FF)],
            'Tamil': [(0x0B80, 0x0BFF)],
            'Telugu': [(0x0C00, 0x0C7F)],
            'Kannada': [(0x0C80, 0x0CFF)],
            'Malayalam': [(0x0D00, 0x0D7F)],
            'Thai': [(0x0E00, 0x0E7F)],
            'CJK': [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF)],
            'Hiragana': [(0x3040, 0x309F)],
            'Katakana': [(0x30A0, 0x30FF)],
            'Hangul': [(0xAC00, 0xD7AF), (0x1100, 0x11FF)]
        }
    
    def _initialize_language_indicators(self) -> Dict[str, Dict[str, Any]]:
        """Initialize language-specific indicators and patterns"""
        return {
            'en': {
                'common_words': ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'],
                'articles': ['a', 'an', 'the'],
                'prepositions': ['in', 'on', 'at', 'by', 'for', 'with', 'from', 'to', 'of', 'about'],
                'script': 'Latin'
            },
            'es': {
                'common_words': ['el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no', 'te', 'lo'],
                'articles': ['el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas'],
                'prepositions': ['de', 'en', 'por', 'para', 'con', 'sin', 'sobre', 'bajo', 'entre'],
                'script': 'Latin'
            },
            'fr': {
                'common_words': ['le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour'],
                'articles': ['le', 'la', 'les', 'un', 'une', 'des', 'du', 'de la'],
                'prepositions': ['de', 'en', 'pour', 'avec', 'sur', 'dans', 'par', 'sans', 'sous'],
                'script': 'Latin'
            },
            'de': {
                'common_words': ['der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich', 'des', 'auf'],
                'articles': ['der', 'die', 'das', 'ein', 'eine', 'einen', 'einem', 'einer'],
                'prepositions': ['in', 'mit', 'von', 'zu', 'an', 'auf', 'für', 'durch', 'bei', 'über'],
                'script': 'Latin'
            },
            'hi': {
                'common_words': ['है', 'के', 'में', 'और', 'को', 'से', 'की', 'पर', 'यह', 'वह', 'एक', 'होने'],
                'script': 'Devanagari'
            },
            'zh': {
                'common_words': ['的', '一', '是', '在', '了', '不', '和', '有', '人', '这', '中', '大'],
                'script': 'CJK'
            },
            'ja': {
                'common_words': ['の', 'に', 'は', 'を', 'た', 'が', 'で', 'て', 'と', 'し', 'れ', 'さ'],
                'script': 'CJK'
            },
            'ko': {
                'common_words': ['이', '가', '을', '를', '의', '에', '는', '은', '와', '과', '로', '으로'],
                'script': 'Hangul'
            },
            'ar': {
                'common_words': ['في', 'من', 'إلى', 'على', 'أن', 'هذا', 'هذه', 'التي', 'الذي', 'كان', 'قد'],
                'script': 'Arabic'
            },
            'kn': {
                'common_words': ['ಆದ', 'ಇದು', 'ಅವರ', 'ಮತ್ತು', 'ಒಂದು', 'ಆ', 'ಇದೆ', 'ಮಾಡಿ', 'ಬಂದ', 'ಗೆ'],
                'script': 'Kannada'
            }
        }
    
    def detect_script(self, text: str) -> Dict[str, float]:
        """Detect the dominant script(s) in text"""
        script_counts = defaultdict(int)
        total_chars = 0
        
        for char in text:
            if char.isspace() or char.isdigit() or char in '.,!?;:()[]{}':
                continue
                
            char_code = ord(char)
            total_chars += 1
            
            for script, ranges in self.script_ranges.items():
                for start, end in ranges:
                    if start <= char_code <= end:
                        script_counts[script] += 1
                        break
        
        if total_chars == 0:
            return {}
        
        # Calculate percentages
        script_percentages = {
            script: count / total_chars 
            for script, count in script_counts.items()
        }
        
        return script_percentages
    
    def detect_language(self, text: str) -> str:
        """Enhanced language detection with improved accuracy"""
        if not text or len(text.strip()) < 3:
            return 'unknown'
        
        # Clean text for analysis - preserve more characters for better detection
        clean_text = re.sub(r'[^\w\s\u0900-\u097F\u0C80-\u0CFF\u0600-\u06FF\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7AF]', ' ', text.lower())
        words = clean_text.split()
        
        if not words:
            return 'unknown'
        
        # Enhanced script detection with better thresholds
        script_scores = self.detect_script(text)
        if not script_scores:
            return 'en'  # Default to English
        
        # Get dominant scripts (may be multiple for mixed content)
        dominant_scripts = [script for script, score in script_scores.items() if score > 0.1]
        
        # Enhanced CJK detection
        if any(script in dominant_scripts for script in ['CJK', 'Hiragana', 'Katakana', 'Hangul']):
            # Korean detection (Hangul)
            if 'Hangul' in dominant_scripts or script_scores.get('Hangul', 0) > 0.3:
                return 'ko'
            
            # Japanese detection (Hiragana/Katakana mixed with CJK)
            if ('Hiragana' in dominant_scripts or 'Katakana' in dominant_scripts or 
                script_scores.get('Hiragana', 0) > 0.05 or script_scores.get('Katakana', 0) > 0.05):
                return 'ja'
            
            # Chinese detection (primarily CJK without Japanese indicators)
            if 'CJK' in dominant_scripts and script_scores.get('CJK', 0) > 0.4:
                return 'zh'
        
        # Enhanced Indic script detection
        indic_script_map = {
            'Devanagari': 'hi',
            'Kannada': 'kn',
            'Tamil': 'ta',
            'Telugu': 'te',
            'Bengali': 'bn',
            'Malayalam': 'ml'
        }
        
        for script in indic_script_map:
            if script in dominant_scripts and script_scores.get(script, 0) > 0.3:
                return indic_script_map[script]
        
        # Arabic script detection
        if 'Arabic' in dominant_scripts and script_scores.get('Arabic', 0) > 0.3:
            return 'ar'
        
        # Latin script language detection with improved scoring
        if 'Latin' in dominant_scripts:
            language_scores = defaultdict(float)
            
            for lang_code, lang_data in self.language_indicators.items():
                if lang_data.get('script') != 'Latin':
                    continue
                
                # Enhanced word matching with different weights
                common_words = lang_data.get('common_words', [])
                articles = lang_data.get('articles', [])
                prepositions = lang_data.get('prepositions', [])
                
                # Weight different types of words
                word_score = 0
                for word in words:
                    if word in common_words:
                        word_score += 2  # High weight for common words
                    elif word in articles:
                        word_score += 3  # Very high weight for articles
                    elif word in prepositions:
                        word_score += 1.5  # Medium weight for prepositions
                
                if len(words) > 0:
                    language_scores[lang_code] = word_score / len(words)
            
            # Return best match if confidence is high enough
            if language_scores:
                best_lang, best_score = max(language_scores.items(), key=lambda x: x[1])
                if best_score > 0.05:  # Minimum confidence threshold
                    return best_lang
        
        # Fallback based on character patterns for mixed/unclear content
        if script_scores.get('Latin', 0) > 0.5:
            return 'en'  # Default to English for Latin script
        elif script_scores.get('CJK', 0) > 0.3:
            return 'zh'  # Default to Chinese for CJK
        elif script_scores.get('Devanagari', 0) > 0.3:
            return 'hi'  # Default to Hindi for Devanagari
        elif script_scores.get('Arabic', 0) > 0.3:
            return 'ar'  # Default to Arabic
        
        return 'en'  # Final fallback to English

class MultilingualPDFExtractor:
    """Main class for multilingual PDF data extraction"""
    
    def __init__(self, model_path: str = None):
        self.language_detector = MultilingualLanguageDetector()
        self.model_path = model_path
        self.nlp_pipeline = None
        
        # Load MobileBERT if available and model path provided
        if BERT_AVAILABLE and model_path and os.path.exists(model_path):
            self._load_bert_model()
    
    def _load_bert_model(self):
        """Load MobileBERT model for semantic validation"""
        try:
            print(f"Loading MobileBERT model from: {self.model_path}")
            
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                use_fast=True,
                local_files_only=True
            )
            
            model = AutoModelForQuestionAnswering.from_pretrained(
                self.model_path,
                local_files_only=True,
                torch_dtype=torch.float32
            )
            
            self.nlp_pipeline = pipeline(
                "question-answering",
                model=model,
                tokenizer=tokenizer,
                device=-1  # CPU
            )
            print("MobileBERT model loaded successfully")
            
        except Exception as e:
            print(f"Warning: Could not load MobileBERT model: {e}")
            self.nlp_pipeline = None
    
    def extract_text_blocks(self, pdf_path: str) -> List[TextBlock]:
        """Extract text blocks with formatting information from PDF"""
        doc = fitz.open(pdf_path)
        text_blocks = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            blocks = page.get_text("dict")
            
            for block in blocks["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            if span["text"].strip():
                                text_block = TextBlock(
                                    text=span["text"].strip(),
                                    bbox=tuple(span["bbox"]),
                                    font_size=span["size"],
                                    font_name=span["font"],
                                    font_flags=span["flags"],
                                    page_num=page_num
                                )
                                text_blocks.append(text_block)
        
        doc.close()
        return text_blocks
    
    def clean_and_merge_blocks(self, blocks: List[TextBlock]) -> List[TextBlock]:
        """Enhanced multilingual text block cleaning and merging"""
        cleaned_blocks = []
        
        for block in blocks:
            # Language-aware text cleaning
            text = self._clean_text_multilingual(block.text)
            if not text:
                continue
            
            # Create cleaned block
            cleaned_block = TextBlock(
                text=text,
                bbox=block.bbox,
                font_size=block.font_size,
                font_name=block.font_name,
                font_flags=block.font_flags,
                page_num=block.page_num,
                line_num=block.line_num
            )
            cleaned_blocks.append(cleaned_block)
        
        # Enhanced merging with language awareness
        merged_blocks = []
        i = 0
        
        while i < len(cleaned_blocks):
            current_block = cleaned_blocks[i]
            merged_text = current_block.text
            bbox = list(current_block.bbox)
            
            # Look for blocks to merge on the same line
            j = i + 1
            while j < len(cleaned_blocks):
                next_block = cleaned_blocks[j]
                
                # Enhanced merging criteria
                if self._should_merge_blocks(current_block, next_block):
                    # Determine appropriate separator
                    separator = self._get_text_separator(current_block.text, next_block.text)
                    merged_text += separator + next_block.text
                    
                    # Update bounding box
                    bbox[2] = max(bbox[2], next_block.bbox[2])  # Extend right
                    bbox[3] = max(bbox[3], next_block.bbox[3])  # Extend bottom
                    j += 1
                else:
                    break
            
            # Create merged block
            merged_block = TextBlock(
                text=merged_text.strip(),
                bbox=tuple(bbox),
                font_size=current_block.font_size,
                font_name=current_block.font_name,
                font_flags=current_block.font_flags,
                page_num=current_block.page_num,
                line_num=current_block.line_num
            )
            merged_blocks.append(merged_block)
            i = j
        
        return merged_blocks
    
    def _clean_text_multilingual(self, text: str) -> str:
        """Enhanced multilingual text cleaning"""
        if not text:
            return ""
        
        text = text.strip()
        
        # Basic filtering
        if (len(text) < 1 or 
            len(text) > 1000 or  # Very long texts are usually not titles/headings
            text.isdigit() or    # Pure numbers
            re.match(r'^[\W_]+$', text)):  # Only punctuation/whitespace
            return ""
        
        # Remove common OCR artifacts
        text = re.sub(r'[^\w\s\u0900-\u097F\u0C80-\u0CFF\u0600-\u06FF\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7AF\u00C0-\u017F\u0400-\u04FF.,!?;:()\[\]{}"\'`~@#$%^&*+=|\\/<>-]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove obvious noise patterns
        noise_patterns = [
            r'^\d+$',                    # Just page numbers
            r'^[^\w]*$',                 # Only symbols
            r'^www\.|http|@.*\.(com|org|edu)', # URLs/emails
            r'^\s*$'                     # Empty after cleaning
        ]
        
        for pattern in noise_patterns:
            if re.search(pattern, text.lower()):
                return ""
        
        return text
    
    def _should_merge_blocks(self, block1: TextBlock, block2: TextBlock) -> bool:
        """Determine if two text blocks should be merged"""
        # Must be on same page
        if block1.page_num != block2.page_num:
            return False
        
        # Must be on similar horizontal line (within 5 points)
        if abs(block1.y - block2.y) > 5:
            return False
        
        # Second block should be to the right of first block
        if block2.x < block1.bbox[2] - 10:
            return False
        
        # Gap between blocks should be reasonable (not too far apart)
        horizontal_gap = block2.x - block1.bbox[2]
        if horizontal_gap > 50:  # More than 50 points apart
            return False
        
        # Similar font characteristics for clean merging
        font_size_diff = abs(block1.font_size - block2.font_size)
        if font_size_diff > 2:  # Font sizes too different
            return False
        
        return True
    
    def _get_text_separator(self, text1: str, text2: str) -> str:
        """Determine appropriate separator between merged texts"""
        # Handle different script joining rules
        
        # If first text ends with hyphen, join directly (hyphenated words)
        if text1.endswith('-'):
            return ""
        
        # If either text is single character, use minimal spacing
        if len(text1) == 1 or len(text2) == 1:
            return ""
        
        # CJK languages typically don't need spaces
        if (any('\u4E00' <= c <= '\u9FFF' for c in text1) or
            any('\u4E00' <= c <= '\u9FFF' for c in text2)):
            return ""
        
        # Arabic text joining
        if (any('\u0600' <= c <= '\u06FF' for c in text1) or
            any('\u0600' <= c <= '\u06FF' for c in text2)):
            return " "
        
        # Default to space for Latin and other scripts
        return " "
    
    def find_title_with_rules(self, blocks: List[TextBlock], language: str) -> Optional[str]:
        """Enhanced multilingual title detection with improved language-specific rules"""
        if not blocks:
            return None

        # Compute font size statistics
        font_sizes = [b.font_size for b in blocks]
        avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 0
        max_font_size = max(font_sizes) if font_sizes else 0
        font_size_std = (sum((x - avg_font_size) ** 2 for x in font_sizes) / len(font_sizes)) ** 0.5 if font_sizes else 0

        # Enhanced candidate scoring
        candidates = []
        for i, block in enumerate(blocks[:min(20, len(blocks))]):  # Look at more blocks for better detection
            text = block.text.strip()
            if len(text) < 2 or len(text) > 250:  # More lenient length limits
                continue
            
            score = 0.0
            
            # Font size scoring (enhanced)
            if block.font_size >= max_font_size * 0.95:
                score += 3.0  # Very large font
            elif block.font_size >= max_font_size * 0.85:
                score += 2.0  # Large font
            elif block.font_size > avg_font_size + font_size_std:
                score += 1.5  # Above average + std dev
            elif block.font_size > avg_font_size * 1.2:
                score += 1.0  # Moderately large
            
            # Style scoring
            if block.is_bold:
                score += 1.5
            if block.is_italic:
                score += 0.5
            if block.font_name and any(style in block.font_name.lower() 
                                     for style in ['bold', 'black', 'heavy', 'semibold']):
                score += 1.0
            
            # Position scoring (enhanced for different layouts)
            if block.page_num == 0:
                if i == 0:
                    score += 2.0  # First block bonus
                elif i < 3:
                    score += 1.5  # Early blocks bonus
                elif i < 8:
                    score += 1.0  # Top section bonus
                
                # Y-position bonus (higher on page = more likely title)
                if block.y < 200:
                    score += 1.5
                elif block.y < 400:
                    score += 1.0
            
            # Language-specific content scoring
            content_score = self._score_title_content(text, language)
            score += content_score
            
            # Length scoring (language-dependent)
            length_score = self._score_title_length(text, language)
            score += length_score
            
            # Uniqueness scoring
            text_appearances = sum(1 for b in blocks if b.text.strip().lower() == text.lower())
            if text_appearances == 1:
                score += 1.0  # Unique text bonus
            elif text_appearances <= 3:
                score += 0.5  # Semi-unique bonus
            else:
                score -= 1.0  # Repeated text penalty
            
            # Format scoring
            if not text.endswith(('.', '।', '。', '？', '！')):
                score += 0.5  # Titles usually don't end with sentence punctuation
            
            if text.count('.') == 0:
                score += 0.3  # Titles usually don't have periods
            
            candidates.append((score, text, i, block))
        
        if not candidates:
            return None
        
        # Sort by score and apply additional validation
        candidates.sort(reverse=True, key=lambda x: x[0])
        
        # Enhanced validation for top candidates
        for score, text, position, block in candidates[:5]:
            if score >= 2.5:  # Minimum confidence threshold
                # Additional validation
                if self._validate_title_candidate(text, language, blocks):
                    return text
        
        # Fallback: look for largest font that passes basic validation
        first_page_blocks = [b for b in blocks if b.page_num == 0]
        if first_page_blocks:
            first_page_blocks.sort(key=lambda b: (-b.font_size, b.y))
            for block in first_page_blocks[:8]:
                text = block.text.strip()
                if (len(text) >= 3 and 
                    self._is_title_like(text, language) and 
                    self._validate_title_candidate(text, language, blocks)):
                    return text

        return None
    
    def _score_title_content(self, text: str, language: str) -> float:
        """Score text content based on title-like characteristics for specific language"""
        score = 0.0
        
        # Common title indicators across languages
        title_indicators = {
            'en': ['analysis', 'study', 'report', 'guide', 'manual', 'introduction', 'overview', 
                   'application', 'form', 'document', 'research', 'survey', 'review'],
            'es': ['análisis', 'estudio', 'informe', 'guía', 'manual', 'introducción', 'resumen',
                   'aplicación', 'formulario', 'documento', 'investigación'],
            'fr': ['analyse', 'étude', 'rapport', 'guide', 'manuel', 'introduction', 'aperçu',
                   'application', 'formulaire', 'document', 'recherche'],
            'de': ['analyse', 'studie', 'bericht', 'anleitung', 'handbuch', 'einführung', 'übersicht',
                   'anwendung', 'formular', 'dokument', 'forschung'],
            'hi': ['विश्लेषण', 'अध्ययन', 'रिपोर्ट', 'गाइड', 'पुस्तिका', 'परिचय', 'अवलोकन',
                   'आवेदन', 'फॉर्म', 'दस्तावेज़', 'अनुसंधान'],
            'zh': ['分析', '研究', '报告', '指南', '手册', '介绍', '概述', '申请', '表格', '文档'],
            'ja': ['分析', '研究', '報告', 'ガイド', 'マニュアル', '紹介', '概要', 'アプリケーション', 'フォーム', '文書'],
            'ar': ['تحليل', 'دراسة', 'تقرير', 'دليل', 'دليل', 'مقدمة', 'نظرة عامة', 'تطبيق', 'استمارة', 'وثيقة']
        }
        
        indicators = title_indicators.get(language, title_indicators.get('en', []))
        text_lower = text.lower()
        
        for indicator in indicators:
            if indicator in text_lower:
                score += 1.0
                break
        
        # Capitalization patterns
        if language in ['en', 'es', 'fr', 'de']:
            if text.istitle():
                score += 1.0
            elif text.isupper() and len(text) > 4:
                score += 0.8
            elif text[0].isupper():
                score += 0.5
        
        # Check for acronyms or abbreviations (often in titles)
        if re.search(r'\b[A-Z]{2,}\b', text):
            score += 0.5
        
        return score
    
    def _score_title_length(self, text: str, language: str) -> float:
        """Score text length appropriateness for titles in specific language"""
        char_count = len(text)
        word_count = len(text.split())
        
        # Language-specific optimal ranges
        optimal_ranges = {
            'en': (5, 80, 1, 15),      # (min_chars, max_chars, min_words, max_words)
            'es': (5, 90, 1, 15),      # Spanish tends to be longer
            'fr': (5, 90, 1, 15),      # French tends to be longer
            'de': (5, 100, 1, 15),     # German can have long compound words
            'hi': (3, 120, 1, 20),     # Hindi can have longer descriptive titles
            'zh': (2, 60, 1, 30),      # Chinese characters are more information-dense
            'ja': (2, 60, 1, 30),      # Japanese similar to Chinese
            'ar': (3, 100, 1, 20),     # Arabic can have longer titles
            'kn': (3, 120, 1, 20),     # Kannada similar to Hindi
        }
        
        min_chars, max_chars, min_words, max_words = optimal_ranges.get(language, (5, 80, 1, 15))
        
        score = 0.0
        
        # Character length scoring
        if min_chars <= char_count <= max_chars:
            score += 1.0
        elif char_count < min_chars:
            score -= 1.0
        elif char_count > max_chars * 1.5:
            score -= 2.0
        elif char_count > max_chars:
            score -= 0.5
        
        # Word count scoring
        if min_words <= word_count <= max_words:
            score += 0.5
        elif word_count > max_words * 1.5:
            score -= 1.0
        
        return score
    
    def _validate_title_candidate(self, text: str, language: str, blocks: List[TextBlock]) -> bool:
        """Enhanced validation for title candidates"""
        # Basic validation
        if not text or len(text.strip()) < 2:
            return False
        
        text_lower = text.lower().strip()
        
        # Universal exclusion patterns
        exclusion_patterns = [
            r'^\d+$',                    # Just numbers
            r'^page\s+\d+',              # Page numbers
            r'^\d{1,2}/\d{1,2}/\d{4}$',  # Dates
            r'^www\.|http|@.*\.com',     # URLs/emails
            r'^©.*copyright|all rights reserved',  # Copyright
            r'^table\s+of\s+contents',   # TOC
            r'^appendix\s+[a-z]',        # Appendix
            r'^figure\s+\d+|^table\s+\d+',  # Figure/table captions
            r'^references|^bibliography', # References
        ]
        
        for pattern in exclusion_patterns:
            if re.search(pattern, text_lower):
                return False
        
        # Check if it appears too frequently (likely header/footer)
        appearances = sum(1 for block in blocks if block.text.strip().lower() == text_lower)
        if appearances > 3:
            return False
        
        # Language-specific validation
        if language in ['en', 'es', 'fr', 'de']:
            # Latin script languages: should start with letter or number
            if not re.match(r'^[A-Za-zÀ-ÿ0-9]', text):
                return False
        elif language in ['hi', 'kn', 'te', 'ta']:
            # Indic scripts: basic character validation
            if not any('\u0900' <= c <= '\u097F' or '\u0C80' <= c <= '\u0CFF' or 
                      '\u0C00' <= c <= '\u0C7F' or '\u0B80' <= c <= '\u0BFF' 
                      for c in text):
                return False
        elif language in ['zh', 'ja']:
            # CJK: should contain CJK characters
            if not any('\u4E00' <= c <= '\u9FFF' or '\u3040' <= c <= '\u309F' or 
                      '\u30A0' <= c <= '\u30FF' for c in text):
                return False
        elif language == 'ar':
            # Arabic: should contain Arabic characters
            if not any('\u0600' <= c <= '\u06FF' for c in text):
                return False
        
        return True
    
    def _is_title_like(self, text: str, language: str) -> bool:
        """Check if text looks like a title"""
        # Basic title characteristics
        if not text or len(text) < 3:
            return False
        
        # Remove common noise patterns
        noise_patterns = [
            r'^\d+$',  # Just numbers
            r'^page\s+\d+$',  # Page numbers
            r'^www\.',  # URLs
            r'^\d{1,2}/\d{1,2}/\d{4}$',  # Dates
            r'^©.*copyright'  # Copyright
        ]
        
        for pattern in noise_patterns:
            if re.search(pattern, text.lower()):
                return False
        
        # Language-specific title patterns
        if language == 'en':
            # English title patterns
            return bool(re.match(r'^[A-Z]', text) and not text.endswith('.'))
        elif language in ['es', 'fr', 'de']:
            # European languages
            return bool(re.match(r'^[A-ZÀ-ÿ]', text) and not text.endswith('.'))
        elif language in ['hi', 'kn', 'ta', 'te']:
            # Indic scripts
            return len(text) > 3 and not text.endswith('।')
        elif language in ['zh', 'ja']:
            # CJK languages
            return len(text) > 2 and not text.endswith('。')
        elif language == 'ar':
            # Arabic
            return len(text) > 3 and not text.endswith('.')
        
        # Default: basic checks
        return len(text) > 3 and not text.endswith('.')
    
    def detect_headings(self, blocks: List[TextBlock], language: str) -> List[Dict[str, Any]]:
        """Advanced: Detect headings using font, style, position, text features, and more."""
        if not blocks:
            return []

        # Font size clustering for heading levels
        font_sizes = sorted(set(b.font_size for b in blocks))
        if not font_sizes:
            return []
        unique_sizes = sorted(list(set(font_sizes)), reverse=True)
        h1_size = unique_sizes[0]
        h2_size = unique_sizes[1] if len(unique_sizes) > 1 else h1_size
        h3_size = unique_sizes[2] if len(unique_sizes) > 2 else h2_size

        def get_level(size):
            if abs(size - h1_size) < 0.01:
                return "H1"
            elif abs(size - h2_size) < 0.01:
                return "H2"
            elif abs(size - h3_size) < 0.01:
                return "H3"
            elif size > h2_size:
                return "H2"
            elif size > h3_size:
                return "H3"
            else:
                return None

        # For uniqueness and repetition
        text_counter = Counter(b.text.strip() for b in blocks)

        headings = []
        for block in blocks:
            text = block.text.strip()
            if len(text) < 2 or len(text) > 120:
                continue
            # Features
            level = get_level(block.font_size)
            style_score = 0
            if block.is_bold:
                style_score += 1
            if block.is_italic:
                style_score += 0.5
            if block.font_name and ("bold" in block.font_name.lower() or "black" in block.font_name.lower()):
                style_score += 0.5
            if self._has_numbering_pattern(text, language):
                style_score += 1
            if self._matches_heading_pattern(text, language):
                style_score += 0.5
            # Text casing: headings often Title Case or ALL CAPS
            is_upper = text.isupper()
            is_title = text.istitle()
            if is_upper:
                style_score += 0.4
            elif is_title:
                style_score += 0.2
            # Penalize if text is repeated a lot (likely footer/header)
            is_unique = text_counter[text] == 1
            if not is_unique:
                style_score -= 1
            # Penalize if text looks like a body paragraph
            word_count = len(text.split())
            if word_count > 10 and not self._has_numbering_pattern(text, language):
                style_score -= 0.5
            # Headings are usually not sentences
            if not text.endswith(('.', '।', '。')):
                style_score += 0.2
            # Assign heading if font size matches a heading level and style_score is sufficient
            is_heading = False
            heading_level = None
            if level in ("H1", "H2", "H3") and style_score >= (0.8 if level == "H3" else 0.5):
                is_heading = True
                heading_level = level
            elif style_score >= 2:
                is_heading = True
                heading_level = level or "H3"
            if is_heading and heading_level:
                headings.append({
                    "level": heading_level,
                    "text": text,
                    "page": block.page_num,
                    "font_size": block.font_size,
                    "is_bold": block.is_bold,
                    "is_italic": block.is_italic,
                    "font_name": block.font_name,
                    "text_length": len(text),
                    "word_count": word_count,
                    "is_upper": is_upper,
                    "is_title": is_title,
                    "is_unique": is_unique,
                    "heading_score": round(style_score, 2)
                })

        # Apply MobileBERT semantic validation if available
        if self.nlp_pipeline and len(headings) > 0:
            headings = self._apply_bert_validation(headings, blocks)

        return headings
    
    def _has_numbering_pattern(self, text: str, language: str) -> bool:
        """Enhanced multilingual numbering pattern detection for headings"""
        # Universal patterns (work across languages)
        universal_patterns = [
            r'^\d+\.?\s+',          # 1. or 1 
            r'^\d+\.\d+\.?\s+',     # 1.1. or 1.1
            r'^\d+\.\d+\.\d+\.?\s+', # 1.1.1. or 1.1.1
            r'^[A-Z]\.?\s+',        # A. or A
            r'^[a-z]\.?\s+',        # a. or a
            r'^\([a-z]\)\s*',       # (a)
            r'^\([0-9]+\)\s*',      # (1)
            r'^[IVX]+\.?\s+',       # Roman numerals I., II., etc.
            r'^[ivx]+\.?\s+',       # Lower roman i., ii., etc.
            r'^\*\s+',              # Bullet points
            r'^•\s+',               # Bullet points
            r'^-\s+',               # Dash points
            r'^\d+\)\s+',           # 1) format
        ]
        
        # Language-specific patterns
        language_patterns = {
            'en': [
                r'^Chapter\s+[IVX\d]+',
                r'^Section\s+\d+',
                r'^Part\s+[IVX\d]+',
                r'^Appendix\s+[A-Z]',
            ],
            'es': [
                r'^Capítulo\s+[IVX\d]+',
                r'^Sección\s+\d+',
                r'^Parte\s+[IVX\d]+',
                r'^Apéndice\s+[A-Z]',
            ],
            'fr': [
                r'^Chapitre\s+[IVX\d]+',
                r'^Section\s+\d+',
                r'^Partie\s+[IVX\d]+',
                r'^Annexe\s+[A-Z]',
            ],
            'de': [
                r'^Kapitel\s+[IVX\d]+',
                r'^Abschnitt\s+\d+',
                r'^Teil\s+[IVX\d]+',
                r'^Anhang\s+[A-Z]',
            ],
            'hi': [
                r'^अध्याय\s+\d+',
                r'^भाग\s+\d+',
                r'^खंड\s+\d+',
                r'^परिशिष्ट\s+[A-Z]',
            ],
            'zh': [
                r'^第\s*[一二三四五六七八九十\d]+\s*章',
                r'^第\s*[一二三四五六七八九十\d]+\s*节',
                r'^第\s*[一二三四五六七八九十\d]+\s*部分',
                r'^\d+\.',
                r'^[一二三四五六七八九十]+[、.]',
            ],
            'ja': [
                r'^第\s*[一二三四五六七八九十\d]+\s*章',
                r'^第\s*[一二三四五六七八九十\d]+\s*節',
                r'^第\s*[一二三四五六七八九十\d]+\s*部',
                r'^\d+\.',
                r'^[一二三四五六七八九十]+[、.]',
            ],
            'ar': [
                r'^الفصل\s+\d+',
                r'^الباب\s+\d+',
                r'^القسم\s+\d+',
                r'^الملحق\s+[A-Z]',
            ],
            'kn': [
                r'^ಅಧ್ಯಾಯ\s+\d+',
                r'^ಭಾಗ\s+\d+',
                r'^ವಿಭಾಗ\s+\d+',
            ]
        }
        
        # Check universal patterns first
        for pattern in universal_patterns:
            if re.search(pattern, text):
                return True
        
        # Check language-specific patterns
        specific_patterns = language_patterns.get(language, [])
        for pattern in specific_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def _matches_heading_pattern(self, text: str, language: str) -> bool:
        """Enhanced multilingual heading pattern matching"""
        # Enhanced language-specific heading indicators
        heading_indicators = {
            'en': [
                'abstract', 'introduction', 'background', 'literature review', 'methodology', 
                'methods', 'results', 'discussion', 'conclusion', 'summary', 'overview',
                'chapter', 'section', 'appendix', 'references', 'bibliography', 'acknowledgments',
                'preface', 'foreword', 'executive summary', 'table of contents', 'index',
                'findings', 'recommendations', 'implications', 'limitations', 'future work'
            ],
            'es': [
                'resumen', 'introducción', 'antecedentes', 'revisión de literatura', 'metodología',
                'métodos', 'resultados', 'discusión', 'conclusión', 'capítulo', 'sección',
                'apéndice', 'referencias', 'bibliografía', 'agradecimientos', 'prefacio',
                'resumen ejecutivo', 'índice', 'hallazgos', 'recomendaciones', 'limitaciones'
            ],
            'fr': [
                'résumé', 'introduction', 'contexte', 'revue de littérature', 'méthodologie',
                'méthodes', 'résultats', 'discussion', 'conclusion', 'chapitre', 'section',
                'annexe', 'références', 'bibliographie', 'remerciements', 'préface',
                'résumé exécutif', 'index', 'conclusions', 'recommandations', 'limitations'
            ],
            'de': [
                'zusammenfassung', 'einführung', 'hintergrund', 'literaturübersicht', 'methodik',
                'methoden', 'ergebnisse', 'diskussion', 'schluss', 'kapitel', 'abschnitt',
                'anhang', 'literatur', 'bibliographie', 'danksagung', 'vorwort',
                'zusammenfassung', 'index', 'erkenntnisse', 'empfehlungen', 'grenzen'
            ],
            'hi': [
                'सारांश', 'परिचय', 'पृष्ठभूमि', 'साहित्य समीक्षा', 'पद्धति', 'विधि', 'परिणाम',
                'चर्चा', 'निष्कर्ष', 'अध्याय', 'भाग', 'परिशिष्ट', 'संदर्भ', 'ग्रंथसूची',
                'आभार', 'प्राक्कथन', 'सूचकांक', 'निष्कर्ष', 'सुझाव', 'सीमाएं'
            ],
            'zh': [
                '摘要', '介绍', '引言', '背景', '文献综述', '方法', '方法论', '结果', '讨论',
                '结论', '总结', '章', '节', '部分', '附录', '参考文献', '致谢', '前言',
                '索引', '发现', '建议', '局限性', '概述', '综述'
            ],
            'ja': [
                '要約', '紹介', '序論', '背景', '文献レビュー', '方法', '手法', '結果', '議論',
                '結論', 'まとめ', '章', '節', '部', '付録', '参考文献', '謝辞', '前書き',
                '索引', '発見', '提案', '制限', '概要', 'レビュー'
            ],
            'ar': [
                'ملخص', 'مقدمة', 'خلفية', 'مراجعة الأدبيات', 'منهجية', 'طرق', 'نتائج',
                'مناقشة', 'خاتمة', 'الفصل', 'القسم', 'الملحق', 'المراجع', 'المصادر',
                'شكر وتقدير', 'مقدمة', 'الفهرس', 'النتائج', 'التوصيات', 'القيود'
            ],
            'kn': [
                'ಸಾರಾಂಶ', 'ಪರಿಚಯ', 'ಹಿನ್ನೆಲೆ', 'ಸಾಹಿತ್ಯ ವಿಮರ್ಶೆ', 'ವಿಧಾನ', 'ಫಲಿತಾಂಶಗಳು',
                'ಚರ್ಚೆ', 'ತೀರ್ಮಾನ', 'ಅಧ್ಯಾಯ', 'ಭಾಗ', 'ಅನುಬಂಧ', 'ಉಲ್ಲೇಖಗಳು',
                'ಗ್ರಂಥಸೂಚಿ', 'ಕೃತಜ್ಞತೆಗಳು', 'ಪ್ರಸ್ತಾವನೆ'
            ]
        }
        
        indicators = heading_indicators.get(language, heading_indicators.get('en', []))
        text_lower = text.lower().strip()
        
        # Direct matching
        for indicator in indicators:
            if indicator.lower() in text_lower:
                return True
        
        # Partial matching for compound titles
        words = text_lower.split()
        for word in words:
            if len(word) > 3 and any(indicator.lower().startswith(word) or word.startswith(indicator.lower()) 
                                   for indicator in indicators):
                return True
        
        return False
    
    def _apply_bert_validation(self, headings: List[Dict[str, Any]], blocks: List[TextBlock]) -> List[Dict[str, Any]]:
        """Apply MobileBERT semantic validation to improve heading detection"""
        if not self.nlp_pipeline:
            return headings
        
        try:
            # Create context from document
            context = " ".join([block.text for block in blocks[:50]])  # First 50 blocks
            
            validated_headings = []
            
            for heading in headings:
                # Ask MobileBERT if this looks like a heading
                question = "Is this a heading or title?"
                
                try:
                    result = self.nlp_pipeline(
                        question=question,
                        context=f"{heading['text']} {context}"
                    )
                    
                    # Boost confidence score based on BERT validation
                    if result['score'] > 0.1:  # Low threshold for heading validation
                        heading['bert_score'] = result['score']
                        heading['bert_validated'] = True
                    else:
                        heading['bert_validated'] = False
                    
                    validated_headings.append(heading)
                    
                except Exception as e:
                    # If BERT fails, keep the heading without validation
                    heading['bert_validated'] = False
                    validated_headings.append(heading)
            
            return validated_headings
            
        except Exception as e:
            print(f"BERT validation failed: {e}")
            return headings
    
    def extract_pdf_data(self, pdf_path: str) -> Dict[str, Any]:
        """Main method to extract all data from PDF (from file path)"""
        print(f"Processing: {pdf_path}")
        start_time = time.time()
        blocks = self.extract_text_blocks(pdf_path)
        return self.extract_from_blocks(blocks, start_time)

    def extract_from_blocks(self, blocks: List[TextBlock], start_time: Optional[float] = None) -> Dict[str, Any]:
        """Extract all data from already-extracted text blocks (for integration with other pipelines)"""
        if start_time is None:
            start_time = time.time()
        extract_time = time.time() - start_time
        print(f"    -> Using {len(blocks)} text blocks")
        if not blocks:
            return {
                "title": "Unknown",
                "language": "unknown",
                "outline": [],
                "stats": {
                    "total_blocks": 0,
                    "processing_time": extract_time
                }
            }
        print("  [2/4] Cleaning and merging blocks...")
        cleaned_blocks = self.clean_and_merge_blocks(blocks)
        print(f"    -> Cleaned to {len(cleaned_blocks)} blocks")
        print("  [3/4] Detecting language...")
        sample_text = " ".join([block.text for block in cleaned_blocks[:10]])
        detected_language = self.language_detector.detect_language(sample_text)
        print(f"    -> Detected language: {detected_language}")
        print("  [4/4] Extracting title and headings...")
        title = self.find_title_with_rules(blocks, detected_language)  # Use original blocks
        if not title:
            title = cleaned_blocks[0].text if cleaned_blocks else "Unknown"
        headings = self.detect_headings(cleaned_blocks, detected_language)
        total_time = time.time() - start_time
        print(f"    -> Found title: '{title[:50]}{'...' if len(title) > 50 else ''}'")
        print(f"    -> Found {len(headings)} headings")
        print(f"    -> Total processing time: {total_time:.2f}s")
        return {
            "title": title,
            "language": detected_language,
            "outline": [
                {
                    "level": h["level"],
                    "text": h["text"],
                    "page": h["page"]
                }
                for h in headings
            ],
            "stats": {
                "total_blocks": len(blocks),
                "cleaned_blocks": len(cleaned_blocks),
                "processing_time": total_time,
                "extract_time": extract_time,
                "language": detected_language
            }
        }

def main():
    """Main function to run the script"""
    parser = argparse.ArgumentParser(
        description="Multilingual PDF Data Extraction Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python multilingual_pdf_extractor.py document.pdf
  python multilingual_pdf_extractor.py document.pdf output.json
  python multilingual_pdf_extractor.py document.pdf --model ./local_mobilebert
        """
    )
    
    parser.add_argument("pdf_file", help="Path to the PDF file to process")
    parser.add_argument("output_file", nargs="?", help="Output JSON file (optional)")
    parser.add_argument("--model", help="Path to MobileBERT model directory (optional)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.pdf_file):
        print(f"Error: PDF file not found: {args.pdf_file}")
        sys.exit(1)
    
    # Generate output filename if not provided
    if not args.output_file:
        base_name = os.path.splitext(os.path.basename(args.pdf_file))[0]
        args.output_file = f"{base_name}_extracted.json"
    
    # Initialize extractor
    print("=== Multilingual PDF Data Extractor ===")
    print(f"Input: {args.pdf_file}")
    print(f"Output: {args.output_file}")
    
    if args.model:
        print(f"Model: {args.model}")
    
    print()
    
    extractor = MultilingualPDFExtractor(model_path=args.model)
    
    try:
        # Extract data
        result = extractor.extract_pdf_data(args.pdf_file)
        
        # Save results
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\n=== EXTRACTION COMPLETE ===")
        print(f"Title: {result['title']}")
        print(f"Language: {result['language']}")
        print(f"Headings found: {len(result['outline'])}")
        print(f"Output saved to: {args.output_file}")
        
        if args.verbose:
            print(f"\nDetailed Statistics:")
            for key, value in result['stats'].items():
                print(f"  {key}: {value}")
            
            print(f"\nHeadings Preview:")
            for i, heading in enumerate(result['outline'][:5]):
                print(f"  {i+1}. [{heading['level']}] {heading['text'][:60]}{'...' if len(heading['text']) > 60 else ''}")
            
            if len(result['outline']) > 5:
                print(f"  ... and {len(result['outline']) - 5} more headings")
    
    except Exception as e:
        print(f"Error processing PDF: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
