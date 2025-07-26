# File Documentation - PDF Outline Extractor

This document provides detailed information about each file in the project, their purpose, key functions, and relationships.

## Project Structure Overview

```
pdf_outline_extractor/
├── run.py                     # Main entry point
├── download_model.py          # Model download utility
├── requirements.txt           # Dependencies
├── README.md                  # Project documentation
├── input/                     # PDF input directory
├── output/                    # JSON output directory
├── models/local_mobilebert/   # NLP model files
└── src/                       # Core processing modules
    ├── __init__.py
    ├── config.py
    ├── enhanced_document_analyzer_backup.py
    ├── pdf_parser.py
    ├── multilingual_pdf_extractor.py
    ├── json_builder.py
    └── scanned_document_detector.py
```

---

## Main Scripts

### `run.py` - Main Entry Point
**Purpose**: Primary execution script that orchestrates the entire PDF processing pipeline.

**Key Functions**:
- `main()`: Main processing function with Docker environment detection
- `OptimizedProcessor.__init__()`: Initialize processing components
- `initialize_extractors()`: Set up analyzers and language detection
- `process_single_pdf()`: Process individual PDF files
- `load_nlp_model()`: Load NLP pipeline with error handling
- `print_processing_summary()`: Display processing statistics

**Key Features**:
- **Docker Auto-Detection**: Automatically detects if running in Docker (`/app/input`) vs local environment
- **Environment Flexibility**: Supports both containerized and local development
- **Error Handling**: Graceful failure handling with detailed error reporting
- **Performance Tracking**: Measures processing time and success rates
- **Model Loading**: Handles optional BERT model loading with fallbacks

**Dependencies**: All src modules, transformers, torch

---

### `download_model.py` - Model Download Utility
**Purpose**: One-time script to download MobileBERT model (~200MB) for semantic validation.

**Key Functions**:
- `download_mobilebert()`: Main download function
- Progress tracking and error handling
- Directory structure creation

**Features**:
- **Automatic Setup**: Creates `models/local_mobilebert/` directory
- **Progress Feedback**: Shows download progress and file sizes
- **Error Handling**: Network and storage error management
- **Verification**: Lists downloaded files with sizes

**Usage**:
```bash
python download_model.py  # One-time setup
```

---

## Core Processing Modules

### `src/enhanced_document_analyzer_backup.py` - Main Analysis Engine
**Purpose**: Advanced document analysis with multi-strategy title detection and comprehensive filtering.

**Key Classes**:
- `EnhancedDocumentAnalyzer`: Main analyzer class with BERT integration

**Key Functions**:

#### Title Extraction
- `analyze_document()`: Main analysis orchestration
- `_extract_title()`: Multi-strategy title detection with 3 approaches:
  1. Candidate scoring and validation
  2. Document type-specific detection  
  3. Author-title combination detection

#### Document Type Detection
- `_detect_document_type_title()`: Specialized patterns for:
  - Academic books ("Introduction to...", "Guide to...")
  - Government forms ("Application form", "Request for...")
  - Event invitations ("Party", "Invitation", "Event")
  - Reports and official documents

#### Scoring and Validation
- `_calculate_title_score_enhanced()`: Advanced scoring algorithm:
  - Font size importance (larger = higher score)
  - Page position (earlier pages preferred)
  - Text length optimization
  - Bold formatting bonus
  - Document type pattern bonuses (300 points)

#### Intelligent Filtering
- `_is_obvious_non_title_enhanced()`: Comprehensive filtering:
  - 70+ form field patterns
  - Testimonial and quote detection
  - Decorative text identification
  - Contact information filtering
- `_is_bad_title()`: Final validation with pattern matching
- `_is_form_field_or_table_content()`: Specialized form field detection

#### Helper Functions
- `_find_author_title_combination()`: Detect author-title patterns
- `_construct_fallback_title()`: Generate titles when none found
- `_is_valid_title_final_check()`: Multi-stage validation pipeline
- `_get_context_around_text()`: Context analysis for validation

**Advanced Features**:
- **MobileBERT Integration**: Semantic validation using pre-trained language model
- **Multi-language Support**: Handles English, Hindi, Japanese, and other scripts
- **Context-Aware Analysis**: Uses surrounding text for better decisions
- **Fallback Mechanisms**: Robust title construction when clear titles unavailable

---

### `src/pdf_parser.py` - PDF Text Extraction
**Purpose**: Extract text blocks from PDFs with layout and formatting metadata preservation.

**Key Functions**:
- `parse_pdf()`: Main PDF parsing function
- `extract_text_blocks()`: Extract text with metadata (font, size, position)
- `detect_scanned_content()`: Identify image-based PDFs
- `clean_text()`: Text normalization and cleaning

**Features**:
- **PyMuPDF Integration**: High-quality text extraction with layout preservation
- **Font Metadata**: Captures font family, size, bold/italic formatting
- **Position Tracking**: Records bounding boxes and page positions
- **Scanned Detection**: Identifies documents requiring OCR
- **Text Cleaning**: Removes artifacts and normalizes Unicode

**Data Structure**:
```python
{
    'text_blocks': [
        {
            'text': 'Extracted text',
            'page_num': 0,
            'bbox': [x0, y0, x1, y1],
            'font_size': 12.0,
            'font_name': 'Arial-Bold',
            'is_bold': True
        }
    ],
    'metadata': {
        'total_pages': 10,
        'has_images': True,
        'is_scanned': False
    }
}
```

---

### `src/multilingual_pdf_extractor.py` - Language Support
**Purpose**: Detect document languages and apply language-specific processing patterns.

**Key Classes**:
- `MultilingualLanguageDetector`: Language detection and analysis
- `MultilingualPDFExtractor`: Language-aware extraction

**Key Functions**:
- `detect_language()`: Automatic language detection
- `extract_multilingual_content()`: Language-specific processing
- `get_language_patterns()`: Language-specific title patterns
- `normalize_unicode()`: Unicode text normalization

**Supported Languages**:
- **English**: Academic, business, government documents
- **Hindi**: Devanagari script support with cultural patterns
- **Japanese**: Hiragana, Katakana, Kanji character handling
- **Auto-detection**: Automatic language identification

**Features**:
- **Script Detection**: Handles multiple writing systems
- **Cultural Patterns**: Language-specific title conventions
- **Unicode Normalization**: Consistent character handling
- **Fallback Support**: Graceful degradation for unsupported languages

---

### `src/json_builder.py` - Output Formatting
**Purpose**: Standardize and format extraction results into structured JSON output.

**Key Functions**:
- `build_json()`: Main JSON construction
- `format_extraction_results()`: Standardize data structure
- `add_metadata()`: Include processing metadata
- `validate_output()`: Ensure output consistency

**Output Schema**:
```json
{
  "title": "Document Title",
  "content_type": "academic_paper|form|report|invitation",
  "language": "en|hi|ja|auto",
  "extracted_content": "Full document text...",
  "headings": [
    {
      "text": "Section Heading",
      "level": 1,
      "page": 1,
      "font_size": 14.0
    }
  ],
  "metadata": {
    "pages": 10,
    "word_count": 5000,
    "processing_time": 2.3,
    "extraction_method": "enhanced_analyzer",
    "model_used": "mobilebert",
    "confidence_score": 0.95
  }
}
```

---

### `src/scanned_document_detector.py` - Document Type Detection
**Purpose**: Identify scanned/image-based PDFs that require different processing approaches.

**Key Functions**:
- `is_scanned_document()`: Main detection algorithm
- `calculate_image_ratio()`: Analyze image-to-text ratio
- `detect_ocr_artifacts()`: Identify OCR processing artifacts
- `analyze_text_quality()`: Assess text extraction quality

**Detection Methods**:
- **Image Ratio Analysis**: High image-to-text ratio indicates scanning
- **Text Quality Assessment**: Poor text quality suggests OCR
- **Layout Analysis**: Irregular text positioning patterns
- **Character Artifacts**: OCR-specific character errors

---

### `src/config.py` - Configuration Settings
**Purpose**: Centralized configuration for thresholds, patterns, and processing parameters.

**Key Configuration Sections**:

#### Processing Directories
```python
INPUT_DIR = "input"
OUTPUT_DIR = "output"
```

#### Length Constraints
```python
MIN_HEADING_LENGTH = 2
MAX_HEADING_LENGTH = 100
MAX_TEXT_LENGTH = 200
```

#### Title Exclusion Patterns
- Date formats (multiple international formats)
- URLs and email addresses
- Copyright notices
- Table/figure captions
- Form field patterns

#### Language Support
```python
SUPPORTED_LANGUAGES = ['en', 'hi', 'ja', 'auto']
DEFAULT_LANGUAGE = 'auto'
```

#### Processing Thresholds
```python
TITLE_CONFIDENCE_THRESHOLD = 0.7
SCANNED_TEXT_RATIO_THRESHOLD = 0.3
MIN_TITLE_SCORE = 50
```

---

## Supporting Files

### `requirements.txt` - Dependencies
**Core Dependencies**:
- **PyMuPDF>=1.23.0**: PDF text extraction and layout analysis
- **transformers>=4.30.0**: HuggingFace transformers library for BERT
- **torch>=2.0.0**: PyTorch deep learning framework
- **numpy>=1.21.0**: Numerical computing
- **python-docx>=0.8.11**: Document processing utilities
- **langdetect>=1.0.9**: Language detection
- **requests>=2.28.0**: HTTP utilities for model downloads

### `src/__init__.py` - Package Initialization
**Purpose**: Makes `src` directory a Python package for clean imports.

---

## File Relationships

### Data Flow Between Files:
1. `run.py` → `pdf_parser.py`: PDF → text blocks
2. `pdf_parser.py` → `enhanced_document_analyzer_backup.py`: text blocks → analysis
3. `enhanced_document_analyzer_backup.py` → `multilingual_pdf_extractor.py`: language detection
4. `enhanced_document_analyzer_backup.py` → `json_builder.py`: results → JSON
5. `scanned_document_detector.py`: Used by parser for document type detection

### Shared Dependencies:
- All modules use `config.py` for settings
- Enhanced analyzer integrates with all processing modules
- JSON builder receives data from all extraction components

### Model Integration:
- `download_model.py` → `models/local_mobilebert/`
- `enhanced_document_analyzer_backup.py` loads from `models/`
- Fallback gracefully when models unavailable

This architecture ensures modularity, maintainability, and extensibility while providing robust PDF processing capabilities across diverse document types and languages.
