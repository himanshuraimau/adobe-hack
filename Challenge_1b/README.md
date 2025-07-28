# PDF Analysis System - Persona-Driven Document Intelligence

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![CPU Only](https://img.shields.io/badge/cpu-only-green.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🎯 Overview

An intelligent document analysis system that extracts and prioritizes the most relevant sections from PDF collections based on specific personas and their job-to-be-done. Built for **Challenge 1B: Persona-Driven Document Intelligence**.

### 🚀 Key Features

- **🧠 Semantic Understanding**: Uses sentence transformers for true semantic similarity matching
- **👤 Persona-Driven**: Tailors content extraction to specific user roles and tasks
- **📊 Intelligent Ranking**: Ranks sections by relevance with mathematical precision
- **🔍 Granular Analysis**: Sentence-level extraction from top-ranked sections
- **⚡ High Performance**: Processes documents in under 60 seconds
- **🐳 Docker Ready**: Production-ready containerization
- **🛡️ Robust**: Comprehensive error handling and logging

## 📁 Project Structure

```
pdf-analysis-system/
├── src/
│   └── pdf_analysis/
│       ├── __init__.py
│       ├── collection_processor.py    # Main orchestration
│       ├── pdf_analyzer.py           # PDF text extraction
│       ├── semantic_ranker.py        # Semantic similarity ranking
│       └── output_generator.py       # JSON output formatting
├── Collection 1/                     # Travel Planning
├── Collection 2/                     # HR Forms Management
├── Collection 3/                     # Food Menu Planning
├── logs/                            # System logs
├── main.py                          # Entry point
├── requirements.txt                 # Dependencies
├── Dockerfile                       # Container setup
├── approach_explanation.md          # Methodology
├── EXECUTION_INSTRUCTIONS.md        # Usage guide
└── README.md                        # This file
```

## 🏃‍♂️ Quick Start

### Option 1: Docker (Recommended)

```bash
# Build the image
docker build --platform linux/amd64 -t pdf-analysis-system .

# Run with your collections
docker run --rm -v $(pwd):/app/input pdf-analysis-system
```

### Option 2: Direct Python

```bash
# Install dependencies
pip install -r requirements.txt

# Run the system
python main.py
```

## 📊 Sample Collections

### Collection 1: Travel Planning 🌍
- **Persona**: Travel Planner
- **Task**: Plan a 4-day trip for 10 college friends
- **Documents**: 7 South of France travel guides
- **Output**: Ranked travel tips, activities, and planning advice

### Collection 2: HR Forms Management 📋
- **Persona**: HR Professional  
- **Task**: Create and manage fillable forms for onboarding
- **Documents**: 15 Adobe Acrobat tutorials
- **Output**: Form creation and compliance guidance

### Collection 3: Food Menu Planning 🍽️
- **Persona**: Food Contractor
- **Task**: Prepare vegetarian buffet menu with gluten-free options
- **Documents**: 9 recipe and cooking guides
- **Output**: Vegetarian recipes and dietary accommodation tips

## 🔧 Technical Specifications

### System Requirements
- **CPU**: Multi-core recommended (uses up to 4 threads)
- **RAM**: Minimum 2GB, 4GB+ recommended
- **Storage**: ~500MB for model cache + document space
- **OS**: Linux, macOS, Windows (cross-platform)

### Performance Metrics
- **Processing Speed**: ~1 second per document
- **Model Size**: 90MB (all-MiniLM-L6-v2)
- **Memory Usage**: Efficient batch processing
- **Success Rate**: 100% across test collections

### Constraints Compliance
- ✅ **CPU Only**: No GPU dependencies
- ✅ **Model Size**: 90MB << 1GB limit
- ✅ **Processing Time**: ~32 seconds << 60 seconds limit
- ✅ **Offline Operation**: Works without internet after setup

## 📋 Input/Output Format

### Input Structure
```json
{
  "challenge_info": {
    "challenge_name": "Challenge 1B"
  },
  "documents": [
    {"filename": "document.pdf"}
  ],
  "persona": {
    "role": "Your persona description"
  },
  "job_to_be_done": {
    "task": "Your specific task"
  }
}
```

### Output Structure
```json
{
  "metadata": {
    "input_documents": ["document.pdf"],
    "persona": "Your persona",
    "job_to_be_done": "Your task",
    "processing_timestamp": "2025-01-28T12:00:00"
  },
  "extracted_sections": [
    {
      "document": "document.pdf",
      "section_title": "Section Title",
      "importance_rank": 1,
      "page_number": 5
    }
  ],
  "subsection_analysis": [
    {
      "document": "document.pdf", 
      "refined_text": "Most relevant sentences...",
      "page_number": 5
    }
  ]
}
```

## 🧠 Methodology

Our approach combines modern NLP with efficient document processing:

1. **Document Processing**: PyMuPDF for robust PDF text extraction
2. **Section Identification**: Font analysis and layout heuristics
3. **Semantic Embedding**: all-MiniLM-L6-v2 sentence transformer
4. **Relevance Ranking**: Cosine similarity between query and sections
5. **Granular Analysis**: Sentence-level extraction from top sections

See [approach_explanation.md](approach_explanation.md) for detailed methodology.

## 🐳 Docker Features

- **Multi-stage build** for optimized image size
- **Pre-cached model** for offline operation
- **Security** with non-root user execution
- **Volume mounting** for flexible data handling
- **Health checks** for monitoring
- **CPU optimization** with proper environment variables

## 📈 Performance Results

```
Total collections found: 3
Successfully processed: 3
Failed to process: 0
Total documents processed: 31
Total sections extracted: 3025
Total processing time: 32.87 seconds
Average time per document: 1.06 seconds
```

## 🔍 Usage Examples

### Basic Usage
```bash
python main.py
```

### Custom Directory
```bash
python main.py --base-path /path/to/collections --log-level DEBUG
```

### Docker with Custom Volumes
```bash
docker run --rm \
  -v /path/to/input:/app/input \
  -v /path/to/output:/app/output \
  pdf-analysis-system
```

## 🧪 Testing

Run the integration tests:
```bash
python test_integration_final.py
```

## 📚 Documentation

- [EXECUTION_INSTRUCTIONS.md](EXECUTION_INSTRUCTIONS.md) - Detailed setup and usage
- [approach_explanation.md](approach_explanation.md) - Technical methodology
- [REQUIREMENTS_COMPLIANCE.md](REQUIREMENTS_COMPLIANCE.md) - Requirements verification

## 🛠️ Development

### Dependencies
- **PyMuPDF**: PDF text extraction
- **sentence-transformers**: Semantic embeddings
- **scikit-learn**: Similarity calculations
- **numpy**: Numerical operations
- **psutil**: System monitoring

### Architecture
- **Modular Design**: Separate components for each processing stage
- **Error Handling**: Comprehensive error recovery
- **Logging**: Detailed progress tracking
- **Batch Processing**: Memory-efficient operation

## 🏆 Challenge Compliance

This system fully complies with all Challenge 1B requirements:

- ✅ **Section Relevance (60 points)**: Semantic similarity ranking
- ✅ **Sub-Section Relevance (40 points)**: Granular sentence analysis
- ✅ **Technical Constraints**: CPU-only, <1GB model, <60s processing
- ✅ **Output Format**: Exact JSON structure compliance
- ✅ **Deliverables**: All required files and documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Built for Adobe India Hackathon 2025 - Challenge 1B: Persona-Driven Document Intelligence** 