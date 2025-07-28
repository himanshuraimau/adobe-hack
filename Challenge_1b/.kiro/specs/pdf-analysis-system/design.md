# Design Document

## Overview

The PDF Analysis System is a persona-driven document intelligence solution that processes multiple PDF collections using semantic similarity. The system uses PyMuPDF for text extraction, sentence-transformers for semantic embeddings, and cosine similarity for relevance ranking to extract and prioritize content based on user personas and tasks.

## Architecture

The system follows a modular pipeline architecture with these core stages:

1. **Collection Discovery & Input Processing** - Discovers collection directories and reads input configurations
2. **PDF Text Extraction & Sectioning** - Extracts text with layout information and identifies logical sections
3. **Semantic Embedding & Ranking** - Converts text to embeddings and ranks by relevance
4. **Granular Analysis** - Performs sentence-level analysis on top sections
5. **Output Generation** - Formats results into structured JSON

## Components and Interfaces

### CollectionProcessor
- **Purpose**: Main orchestrator that discovers and processes each collection
- **Methods**:
  - `discover_collections()` - Finds all Collection directories
  - `process_collection(collection_path)` - Processes a single collection
- **Dependencies**: PDFAnalyzer, OutputGenerator

### PDFAnalyzer
- **Purpose**: Handles PDF parsing, sectioning, and content extraction
- **Methods**:
  - `extract_text_blocks(pdf_path)` - Extracts text with layout info using PyMuPDF
  - `identify_sections(text_blocks)` - Groups blocks into logical sections based on headers
  - `get_section_content(sections)` - Returns structured section data
- **Dependencies**: PyMuPDF (fitz)

### SemanticRanker
- **Purpose**: Handles embedding generation and similarity scoring
- **Methods**:
  - `load_model()` - Initializes sentence-transformer model (all-MiniLM-L6-v2)
  - `create_query_embedding(persona, job)` - Combines persona and task into query embedding
  - `rank_sections(query_embedding, sections)` - Ranks sections by cosine similarity
  - `analyze_sentences(query_embedding, top_sections)` - Performs sentence-level analysis
- **Dependencies**: sentence-transformers, scikit-learn

### OutputGenerator
- **Purpose**: Formats analysis results into required JSON structure
- **Methods**:
  - `generate_metadata(input_data)` - Creates metadata section
  - `format_extracted_sections(ranked_sections)` - Formats section rankings
  - `format_subsection_analysis(sentence_analysis)` - Formats refined text analysis
  - `save_output(collection_path, results)` - Saves JSON to collection directory

## Data Models

### InputConfig
```python
{
    "challenge_info": {
        "challenge_id": str,
        "test_case_name": str,
        "description": str
    },
    "documents": [{"filename": str, "title": str}],
    "persona": {"role": str},
    "job_to_be_done": {"task": str}
}
```

### Section
```python
{
    "document": str,
    "section_title": str,
    "content": str,
    "page_number": int,
    "font_info": dict
}
```

### RankedSection
```python
{
    "document": str,
    "section_title": str,
    "importance_rank": int,
    "page_number": int,
    "similarity_score": float
}
```

### OutputFormat
```python
{
    "metadata": {
        "input_documents": [str],
        "persona": str,
        "job_to_be_done": str,
        "processing_timestamp": str
    },
    "extracted_sections": [RankedSection],
    "subsection_analysis": [{
        "document": str,
        "refined_text": str,
        "page_number": int
    }]
}
```

## Error Handling

### PDF Processing Errors
- **Corrupted PDFs**: Log error and skip file, continue with remaining PDFs
- **Missing PDFs**: Log warning and continue processing available files
- **Text extraction failures**: Attempt alternative extraction methods, fallback to basic text

### Model Loading Errors
- **Model download failures**: Provide clear error message with troubleshooting steps
- **Memory constraints**: Implement batch processing to reduce memory usage
- **CPU performance**: Add progress indicators for long-running operations

### File System Errors
- **Missing directories**: Create output directories if they don't exist
- **Permission errors**: Provide clear error messages about file access requirements
- **Disk space**: Check available space before writing large output files

## Testing Strategy

### Unit Testing Approach
- **PDF Extraction**: Test with sample PDFs of different layouts and formats
- **Section Identification**: Verify header detection with various font styles and sizes
- **Embedding Generation**: Test query and section embedding consistency
- **Similarity Scoring**: Validate cosine similarity calculations with known test cases
- **Output Formatting**: Ensure JSON structure matches specification exactly

### Integration Testing
- **End-to-End Processing**: Test complete pipeline with provided sample collections
- **Multi-Collection Handling**: Verify independent processing of different collections
- **Error Recovery**: Test system behavior with missing files and corrupted data

### Performance Testing
- **CPU Usage**: Monitor resource consumption during processing
- **Memory Management**: Ensure efficient memory usage with large document sets
- **Processing Time**: Benchmark against time constraints for practical usage