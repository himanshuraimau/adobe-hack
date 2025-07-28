# PDF Structure Extractor

A machine learning-powered system for extracting structured outlines from PDF documents using MobileBERT and advanced feature engineering.

## Project Structure

```
pdf-structure-extractor/
├── src/                           # Source code
│   └── pdf_extractor/            # Main package
│       ├── __init__.py           # Package exports
│       ├── core/                 # Core processing modules
│       │   ├── __init__.py
│       │   ├── pdf_parser.py     # PDF text extraction
│       │   ├── preprocessor.py   # Text preprocessing
│       │   ├── feature_extractor.py # Feature engineering
│       │   ├── classifier.py     # MobileBERT classification
│       │   ├── structure_analyzer.py # Document hierarchy
│       │   ├── json_builder.py   # JSON output generation
│       │   └── error_handler.py  # Error handling
│       ├── models/               # Data models
│       │   ├── __init__.py
│       │   └── models.py         # Core data structures
│       ├── config/               # Configuration
│       │   ├── __init__.py
│       │   └── config.py         # System configuration
│       └── utils/                # Utility functions
│           └── __init__.py
├── tests/                        # Test suite
│   ├── __init__.py
│   └── test_*.py                # Unit and integration tests
├── scripts/                      # Demo and utility scripts
│   ├── __init__.py
│   ├── demo_json_builder.py     # JSON builder demo
│   ├── performance_profiler.py  # Performance monitoring
│   ├── final_performance_test.py # 10-second compliance test
│   ├── simple_performance_test.py # Component performance tests
│   └── check_dependencies.py    # Dependency verification
├── docs/                         # Documentation
│   ├── README.md                # Detailed documentation
│   ├── DOCKER_USAGE.md          # Docker instructions
│   └── MULTILINGUAL_IMPROVEMENTS.md # Multilingual features
├── docker/                       # Docker configuration
│   ├── Dockerfile               # Container definition
│   ├── .dockerignore           # Docker ignore rules
│   └── docker-build-test.sh    # Build and test script
├── data/                         # Data directories
│   ├── input/                   # Input PDF files
│   └── output/                  # Generated JSON outputs
├── models/                       # MobileBERT model files
│   └── local_mobilebert/        # Pre-trained model (95MB)
├── logs/                         # Application logs
│   └── errors.log              # Error log file
├── main.py                      # Main application entry point
├── pyproject.toml              # Project configuration
├── uv.lock                     # Dependency lock file
└── __init__.py                 # Root package marker
```

## Installation

### Prerequisites
- Python 3.12+
- uv package manager (recommended) or pip

### Setup

1. Clone the repository and navigate to the project directory:
```bash
cd Challenge_1a
```

2. Install dependencies using uv (recommended) or pip:
```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

3. The MobileBERT model is included in the `models/local_mobilebert/` directory.

### Docker Installation

Build the Docker image:
```bash
# Build from project root (recommended)
docker build --platform linux/amd64 -f docker/Dockerfile -t pdf-structure-extractor .

# Or use the automated build and test script
cd docker
./docker-build-test.sh
```

## Usage

### Command Line Interface

Process a single PDF file:
```bash
python main.py data/input/document.pdf -o data/output/
```

Process all PDFs in a directory:
```bash
python main.py data/input/ -o data/output/
```

Example with sample files:
```bash
# Process all sample PDFs
python main.py data/input/ -o data/output/

# Process with verbose logging
python main.py data/input/ -o data/output/ --verbose

# Process with custom timeout
python main.py data/input/ -o data/output/ --timeout 15
```

Options:
- `-o, --output`: Output directory for JSON files (default: data/output/)
- `--timeout`: Processing timeout in seconds (default: 10)
- `-v, --verbose`: Enable verbose logging

### Docker Usage

Process PDFs using Docker:
```bash
# Process all PDFs in input directory
docker run --rm \
  --platform linux/amd64 \
  -v "$(pwd)/data/input:/app/input:ro" \
  -v "$(pwd)/data/output:/app/output" \
  --memory=16g --cpus=8 --network=none \
  pdf-structure-extractor

# Process with custom timeout
docker run --rm \
  --platform linux/amd64 \
  -v "$(pwd)/data/input:/app/input:ro" \
  -v "$(pwd)/data/output:/app/output" \
  --memory=16g --cpus=8 --network=none \
  pdf-structure-extractor python main.py /app/input --output /app/output --timeout 15
```

### Python API

```python
from src.pdf_extractor import PDFParser, TextPreprocessor, FeatureExtractor
from src.pdf_extractor import HeadingClassifier, StructureAnalyzer, JSONBuilder

# Initialize components
parser = PDFParser()
preprocessor = TextPreprocessor()
feature_extractor = FeatureExtractor()
classifier = HeadingClassifier()
structure_analyzer = StructureAnalyzer()
json_builder = JSONBuilder()

# Process a PDF
text_blocks = parser.parse_document("document.pdf")
processed_blocks = preprocessor.preprocess_blocks(text_blocks)
feature_extractor.initialize_document_stats(processed_blocks)

# Extract features and classify
classification_results = []
for block in processed_blocks:
    features = feature_extractor.extract_features(block)
    result = classifier.classify_block(features, block.text)
    classification_results.append(result)

# Build document structure and generate JSON
document_structure = structure_analyzer.analyze_structure(classification_results)
json_output = json_builder.build_json(document_structure)
```

## Output Format

The system generates JSON files with the following structure:

```json
{
  "title": "Document Title",
  "outline": [
    {"level": "H1", "text": "Introduction", "page": 1},
    {"level": "H2", "text": "Overview", "page": 1},
    {"level": "H2", "text": "Objectives", "page": 2},
    {"level": "H1", "text": "Methodology", "page": 3}
  ]
}
```

## Testing

Run the test suite:
```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_classifier.py
python -m pytest tests/test_integration_*.py

# Run with coverage
python -m pytest tests/ --cov=src/pdf_extractor
```

Run performance tests:
```bash
# Component performance tests
python scripts/simple_performance_test.py

# 10-second compliance test
python scripts/final_performance_test.py
```

## Development

### Code Organization

- **Core Modules** (`src/pdf_extractor/core/`): Main processing logic
- **Models** (`src/pdf_extractor/models/`): Data structures and type definitions
- **Configuration** (`src/pdf_extractor/config/`): System configuration management
- **Tests** (`tests/`): Comprehensive test suite with unit and integration tests
- **Scripts** (`scripts/`): Utility scripts and demos

### Adding New Features

1. Implement core logic in appropriate `src/pdf_extractor/core/` module
2. Add data models to `src/pdf_extractor/models/models.py` if needed
3. Update configuration in `src/pdf_extractor/config/config.py` if needed
4. Add comprehensive tests in `tests/`
5. Update documentation

### Performance Monitoring

The system includes built-in performance monitoring:

```python
from scripts.performance_profiler import get_global_profiler

profiler = get_global_profiler()
with profiler.profile_operation("my_operation"):
    # Your code here
    pass

# Get performance report
report = profiler.get_performance_report()
```

## Architecture

### Processing Pipeline

1. **PDF Parsing**: Extract text blocks with formatting metadata using PyMuPDF
2. **Preprocessing**: Clean and normalize text while preserving structure
3. **Feature Extraction**: Generate classification features (font, position, content)
4. **Classification**: Use MobileBERT + rule-based fallback for heading detection
5. **Structure Analysis**: Build hierarchical document structure
6. **JSON Generation**: Format output according to specification

### Key Components

- **MobileBERT Adapter**: Fine-tuned transformer model for heading classification
- **Feature Engineering**: Multi-dimensional feature extraction (font, spatial, textual)
- **Fallback Classification**: Rule-based system for robust operation
- **Error Handling**: Comprehensive error recovery and graceful degradation
- **Performance Monitoring**: Built-in profiling and optimization tools

## Performance

**Processing Speed:**
- Small documents (1-10 pages): <2 seconds
- Medium documents (10-30 pages): 2-5 seconds
- Large documents (30-50 pages): 5-10 seconds

**Memory Usage:**
- Base memory: ~400MB
- Peak processing: ~700MB
- Model loading: ~200MB additional

**Accuracy:**
- Title detection: >90% accuracy
- Heading classification: >85% accuracy
- Hierarchy building: >80% accuracy

## Troubleshooting

### Common Issues

**"Model loading failed":**
- Ensure the `models/local_mobilebert/` directory exists
- Check available memory (requires ~200MB)
- Verify PyTorch CPU installation

**"Processing timeout":**
- Increase timeout with `--timeout` parameter
- Check document complexity and size
- Monitor system resources

**"PDF parsing failed":**
- Verify PDF file is not corrupted
- Check file permissions
- Ensure PyMuPDF is properly installed

### Debug Mode

Enable verbose logging for troubleshooting:
```bash
python main.py input.pdf -o output/ --verbose
```

## Contributing

1. Follow PEP 8 style guidelines
2. Add comprehensive tests for new features
3. Update documentation for API changes
4. Ensure Docker compatibility
5. Test performance impact

## License

This project is developed for the Adobe PDF Processing Challenge.