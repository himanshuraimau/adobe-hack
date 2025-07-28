# Adobe Hackathon: PDF Intelligence System

A comprehensive PDF processing system with two distinct solutions for document structure extraction and persona-driven document analysis.

## Solutions Overview

This repository contains complete solutions for both challenges:

- **Challenge 1A**: PDF Structure Extractor - Extracts titles and headings (H1-H3) from PDFs with page numbers
- **Challenge 1B**: Persona-Driven Document Intelligence - Finds relevant sections based on user persona and tasks

## Getting Started

Each challenge has its own complete implementation with detailed setup instructions:

### Challenge 1A - PDF Structure Extractor
📁 **Location**: `Challenge_1a/`

A machine learning-powered system using MobileBERT for extracting structured outlines from PDF documents.

**Quick Start:**
```bash
cd Challenge_1a
uv sync  # or pip install -e .
python main.py data/input/ -o data/output/
```

**Key Features:**
- MobileBERT-based heading classification
- 10-second processing constraint compliance
- Docker support with offline operation
- Comprehensive test suite and performance monitoring

📖 **Full Documentation**: See `Challenge_1a/README.md` for complete setup, usage, and API documentation.

### Challenge 1B - Persona-Driven Document Intelligence
📁 **Location**: `Challenge_1b/`

An intelligent document analysis system that extracts and prioritizes relevant sections based on specific personas and tasks.

**Quick Start:**
```bash
cd Challenge_1b
pip install -r requirements.txt
python main.py
```

**Key Features:**
- Semantic similarity using sentence transformers
- Persona-driven content extraction
- 60-second processing constraint compliance
- Three sample collections with different personas

📖 **Full Documentation**: See `Challenge_1b/README.md` for complete setup, usage, and methodology.

## Repository Structure

```
├── Challenge_1a/              # Challenge 1A: PDF Structure Extractor
│   ├── src/                   # Source code with modular architecture
│   ├── tests/                 # Comprehensive test suite
│   ├── docker/                # Docker configuration
│   ├── data/                  # Input/output directories
│   ├── models/                # MobileBERT model files
│   ├── main.py                # Entry point
│   └── README.md              # Detailed documentation
├── Challenge_1b/              # Challenge 1B: Persona Intelligence
│   ├── src/                   # Source code modules
│   ├── Collection 1/          # Travel planning sample
│   ├── Collection 2/          # HR forms sample
│   ├── Collection 3/          # Food menu sample
│   ├── main.py                # Entry point
│   └── README.md              # Detailed documentation
└── README.md                  # This overview file
```

## Technical Highlights

### Challenge 1A
- **ML Model**: MobileBERT (95MB) for heading classification
- **Performance**: <10 seconds for 50-page PDFs
- **Architecture**: Modular pipeline with feature engineering
- **Testing**: Unit tests, integration tests, performance tests

### Challenge 1B
- **ML Model**: all-MiniLM-L6-v2 (90MB) for semantic similarity
- **Performance**: <60 seconds for document collections
- **Architecture**: Semantic ranking with granular analysis
- **Collections**: 3 complete sample collections with different personas

## Docker Support

Both challenges include production-ready Docker configurations:

**Challenge 1A:**
```bash
cd Challenge_1a
docker build --platform linux/amd64 -f docker/Dockerfile -t pdf-structure-extractor .
```

**Challenge 1B:**
```bash
cd Challenge_1b
docker build --platform linux/amd64 -t pdf-analysis-system .
```

## Compliance & Constraints

Both solutions fully comply with hackathon requirements:
- ✅ CPU-only operation (no GPU required)
- ✅ Model size limits (95MB and 90MB respectively)
- ✅ Processing time constraints (10s and 60s respectively)
- ✅ Offline operation capability
- ✅ Exact output format compliance

## Next Steps

1. **For Challenge 1A**: Navigate to `Challenge_1a/` and follow the README for PDF structure extraction
2. **For Challenge 1B**: Navigate to `Challenge_1b/` and follow the README for persona-driven analysis
3. **For Development**: Each challenge includes comprehensive documentation for extending functionality

Each solution is self-contained with its own dependencies, documentation, and examples. Choose the challenge you want to explore and follow the respective README for detailed instructions.