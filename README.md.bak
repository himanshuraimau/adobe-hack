# PDF Outline Extraction Solution

## Overview

This solution extracts structured outlines from PDF documents, identifying titles and headings (H1, H2, H3) with their hierarchical relationships and page numbers. The system is optimized for the 10-second execution constraint while maintaining high accuracy across diverse document types.

## Approach

### Architecture: Enhanced Document Analyzer

The solution uses a **multi-stage filtering and analysis approach**:

1. **PDF Parsing**: Extract text blocks with metadata (font size, style, position)
2. **Text Cleaning**: Remove fragments and noise using pattern matching
3. **Text Merging**: Combine multi-line headings that were split during extraction
4. **Style Analysis**: Identify document font patterns and body text characteristics
5. **Title Extraction**: Find the main document title using position and style heuristics
6. **Heading Detection**: Identify heading candidates using scoring algorithms
7. **Hierarchy Assignment**: Assign H1/H2/H3 levels based on font size and numbering patterns
8. **Quality Filtering**: Remove low-quality headings and duplicates

### Key Features

#### 1. Fragment Detection and Removal
- Advanced pattern matching to identify and remove text fragments
- Handles broken words like "quest for Pr", "r Proposal", "RFP: Request f"
- Filters out incomplete sentences and noise

#### 2. Smart Text Merging
- Merges multi-line headings that were split across text blocks
- Uses font consistency and proximity to determine merge candidates
- Preserves semantic integrity of headings

#### 3. Title Extraction
- Position-based scoring (early blocks get higher scores)
- Font size and style analysis
- Content pattern recognition
- Quality validation to avoid selecting fragments as titles

#### 4. Heading Hierarchy
- Numbering pattern recognition (1., 1.1., 1.1.1., A., (a), etc.)
- Font size-based hierarchy when numbering is absent
- Style consistency analysis (bold, font family)

#### 5. Quality Assurance
- Filters out addresses, contact information, instructions
- Removes duplicate content between title and headings
- Validates heading length and content quality
- Advanced pattern matching for form fields and noise

## Models and Libraries Used

### Core Libraries
- **PyMuPDF (fitz)**: PDF parsing and text extraction
- **transformers**: HuggingFace transformers for MobileBERT
- **torch**: PyTorch for model execution
- **re**: Regular expressions for pattern matching
- **collections**: Counter and defaultdict for data analysis
- **typing**: Type hints for better code clarity

### Model
- **MobileBERT**: Lightweight BERT model (≤200MB) for text quality validation
- Used selectively for edge cases to maintain performance constraints
- Runs entirely offline with local model files
- Model size: ~95MB (well under 200MB constraint)

## Performance Optimizations

### 1. Execution Time Optimizations
- Pre-filtering reduces candidate pool before expensive operations
- Pattern-based classification over heavy NLP processing
- Efficient text block sorting and processing
- Minimal model inference calls
- Optimized for sub-10-second execution

### 2. Memory Optimizations
- Streaming text processing
- Early filtering to reduce memory footprint
- Efficient data structures for text block management

### 3. CPU-Only Design
- No GPU dependencies
- Optimized for CPU execution on amd64 architecture
- Fast tokenizer usage where available
- Works on systems with 8 CPUs and 16GB RAM

## Project Structure

```
pdf_processing_project/
├── README.md                    # This documentation
├── run.py                      # Main execution script
├── requirements.txt            # Python dependencies
├── input/                      # Directory for input PDF files
├── output/                     # Directory for generated JSON files
├── local_mobilebert/          # Local MobileBERT model files
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── ...
└── src/                       # Source code modules
    ├── __init__.py
    ├── config.py              # Configuration and constants
    ├── pdf_parser.py          # PDF parsing and text extraction
    ├── enhanced_document_analyzer.py  # Main analysis logic
    └── json_builder.py        # JSON output formatting
```

## How to Build and Run

### Prerequisites
- Python 3.8 or higher
- All dependencies are included in the project
- No internet connection required (offline execution)

### Expected Execution
The solution is designed to run with the following command:

```bash
python run.py
```

### Setup Instructions
1. **Ensure Python is available**:
   ```bash
   python --version  # Should be 3.8+
   ```

2. **Place PDF files in input directory**:
   ```bash
   # Copy your PDF files to the input/ directory
   cp your_document.pdf input/
   ```

3. **Run the solution**:
   ```bash
   python run.py
   ```

4. **Check results**:
   ```bash
   # JSON files will be generated in output/ directory
   ls output/
   cat output/your_document.json
   ```

### Dependencies
All required dependencies are included:
- **PyMuPDF**: For PDF parsing
- **transformers**: For MobileBERT model
- **torch**: For neural network inference
- **safetensors**: For model loading

No additional installation is required as the project includes:
- Local virtual environment (`myvenv/`)
- Pre-downloaded MobileBERT model (`local_mobilebert/`)
- All necessary Python packages

## Input/Output Format

### Input
- PDF files (up to 50 pages) placed in the `input/` directory
- Supports various document types: academic papers, business documents, forms, technical manuals

### Output
JSON format with the following structure:
```json
{
  "title": "Document Title",
  "outline": [
    { "level": "H1", "text": "Main Heading", "page": 1 },
    { "level": "H2", "text": "Sub Heading", "page": 2 },
    { "level": "H3", "text": "Sub-sub Heading", "page": 3 }
  ]
}
```

## Algorithm Details

### Heading Detection Score Calculation
Each text block receives a score based on:
- Font size relative to body text (weight: 3x difference)
- Bold styling (bonus: +25 points)
- Numbering patterns (bonus: +40 points)
- Position and alignment
- Length appropriateness
- Content quality indicators

### Hierarchy Assignment Logic
1. **Numbering Patterns**: Direct mapping (1. → H1, 1.1. → H2, 1.1.1. → H3)
2. **Font Size Hierarchy**: Largest fonts → H1, medium → H2, smaller → H3
3. **Style Consistency**: Bold + large font → higher hierarchy level

### Quality Filters
- Minimum length: 2 characters
- Maximum length: 150 characters
- Fragment pattern exclusion
- Address and contact info removal
- Form field and instruction text filtering

## Performance Characteristics

- **Execution Time**: ~3-7 seconds for typical documents
- **Model Size**: MobileBERT (~95MB, well under 200MB limit)
- **Memory Usage**: Optimized for 16GB RAM systems
- **CPU Usage**: Efficient single-threaded processing
- **Offline Operation**: No internet connectivity required

## Constraints Compliance

✅ **Execution Time**: ≤ 10 seconds for 50-page PDFs  
✅ **Model Size**: ≤ 200MB (actual: ~95MB)  
✅ **Network**: No internet access required  
✅ **Runtime**: CPU-only, amd64 architecture  
✅ **System Requirements**: 8 CPUs, 16GB RAM  

## Multilingual Support

The solution includes basic multilingual handling:
- Unicode text support
- Pattern-based detection works across languages
- Font-based hierarchy assignment is language-agnostic

## Error Handling

- Graceful degradation when model inference fails
- Fallback to heuristic-only processing
- Empty document handling
- Malformed PDF recovery
- Robust exception handling throughout the pipeline

## Testing and Validation

The solution has been tested on:
- Academic papers with complex structures
- Business documents and proposals
- Technical manuals and specifications
- Forms and applications
- Marketing materials and flyers

All test cases meet the accuracy and performance requirements specified in the challenge.

## Solution Quality

The solution provides:
- **High Precision**: Accurate heading detection with minimal false positives
- **Robust Processing**: Handles diverse document types and layouts
- **Fast Execution**: Optimized for the 10-second constraint
- **Clean Output**: Well-structured JSON with proper hierarchy
- **Offline Operation**: No external dependencies or API calls
