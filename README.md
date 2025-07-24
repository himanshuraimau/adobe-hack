# Adobe Hackathon: PDF Intelligence System

A smart PDF processing system that extracts document structure and provides persona-driven document analysis.

## What This Does

- **Round 1A**: Extracts titles and headings (H1-H3) from PDFs with page numbers
- **Round 1B**: Finds relevant sections based on user persona and tasks
- **Round 2**: Web app with Adobe PDF Embed API

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the main script
python main.py
```

## Round 1A - Document Structure

Extracts structured outlines from PDFs:

```json
{
  "title": "Understanding AI",
  "outline": [
    { "level": "H1", "text": "Introduction", "page": 1 },
    { "level": "H2", "text": "What is AI?", "page": 2 }
  ]
}
```

### Docker Usage

```bash
# Build
docker build --platform linux/amd64 -t pdf-processor .

# Run
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none pdf-processor
```

## Round 1B - Persona Intelligence

Analyzes documents based on:
- User persona (researcher, analyst, student)
- Specific task requirements
- Relevance ranking

## Project Structure

```
├── Challenge - 1(a)/          # Round 1A implementation
│   ├── Datasets/              # Sample PDFs and outputs
│   ├── process_pdfs.py        # Main processing script
│   └── *.md                   # Documentation
├── main.py                    # Entry point
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Technical Approach

- **PDF Processing**: PyMuPDF for text extraction
- **ML Models**: Lightweight transformers (<200MB)
- **Performance**: Multi-threaded processing
- **Languages**: Unicode support for multilingual docs

## Constraints

- 10 seconds max for 50-page PDF
- CPU only, no GPU
- No internet access
- 200MB model size limit