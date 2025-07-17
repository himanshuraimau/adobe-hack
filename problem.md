# Adobe Hackathon: "Connecting the Dots" Challenge

## Table of Contents
1. [Overview](#overview)
2. [Round 1A: Document Structure Extraction](#round-1a-document-structure-extraction)
3. [Round 1B: Persona-Driven Document Intelligence](#round-1b-persona-driven-document-intelligence)
4. [Resources](#resources)

---

## Overview

Build an intelligent PDF processing system that extracts structured information and provides persona-driven document analysis.

**Round 1A**: Extract structured outlines from PDFs (title, headings H1-H3)  
**Round 1B**: Build persona-driven document intelligence for relevant section extraction  
**Round 2**: Create a web application using Adobe's PDF Embed API  

---

## Round 1A: Document Structure Extraction

### Problem Statement

Extract structured outline from PDF documents including title and hierarchical headings (H1, H2, H3) with page numbers.

### Requirements

- Input: PDF file (up to 50 pages)
- Output: JSON file with title and heading structure
- Extract: Title, H1/H2/H3 headings with levels and page numbers

### Output Format

```json
{
  "title": "Understanding AI",
  "outline": [
    { "level": "H1", "text": "Introduction", "page": 1 },
    { "level": "H2", "text": "What is AI?", "page": 2 },
    { "level": "H3", "text": "History of AI", "page": 3 }
  ]
}
```

### Docker Requirements

- AMD64 architecture
- CPU only (no GPU)
- Model size ≤ 200MB
- No internet access
- Base image: `FROM --platform=linux/amd64 <base_image>`

### Execution Commands

```bash
# Build
docker build --platform linux/amd64 -t mysolutionname:identifier .

# Run
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none mysolutionname:identifier
```

### Constraints

| Constraint | Requirement |
|------------|-------------|
| Execution time | ≤ 10 seconds for 50-page PDF |
| Model size | ≤ 200MB |
| Network | No internet access |
| Runtime | CPU (amd64), 8 CPUs, 16 GB RAM |

### Scoring

| Criteria | Points |
|----------|---------|
| Heading Detection Accuracy | 25 |
| Performance | 10 |
| Multilingual Support | 10 |
| **Total** | **45** |

### Deliverables

1. Dockerfile in root directory
2. README.md with approach explanation
3. All dependencies containerized

---

## Round 1B: Persona-Driven Document Intelligence

### Problem Statement

Build a system that extracts and prioritizes relevant sections from multiple documents based on a specific persona and task.

### Input Specification

1. **Documents**: 3-10 related PDFs
2. **Persona**: Role description with expertise areas
3. **Job-to-be-Done**: Specific task to accomplish

### Sample Test Cases

**Academic Research**
- Documents: 4 research papers on "Graph Neural Networks for Drug Discovery"
- Persona: PhD Researcher in Computational Biology
- Job: "Literature review focusing on methodologies, datasets, and benchmarks"

**Business Analysis**
- Documents: 3 annual reports from tech companies (2022-2024)
- Persona: Investment Analyst
- Job: "Analyze revenue trends, R&D investments, and market positioning"

**Educational Content**
- Documents: 5 organic chemistry textbook chapters
- Persona: Undergraduate Chemistry Student
- Job: "Key concepts and mechanisms for reaction kinetics exam prep"

### Output Structure

1. **Metadata**: Input documents, persona, job, timestamp
2. **Extracted Sections**: Document, page, section title, importance rank
3. **Sub-section Analysis**: Document, refined text, page number

### Constraints

- CPU only
- Model size ≤ 1GB
- Processing time ≤ 60 seconds (3-5 documents)
- No internet access

### Scoring

| Criteria | Points | Description |
|----------|---------|-------------|
| Section Relevance | 60 | Match persona + job requirements with ranking |
| Sub-Section Relevance | 40 | Quality of granular extraction and ranking |

### Deliverables

- `approach_explanation.md` (300-500 words)
- Dockerfile and execution instructions
- Sample input/output

---

## Resources

**GitHub Repository**: https://github.com/jhaaj08/Adobe-India-Hackathon25.git

**Provided Materials**:
- Sample input PDF
- Sample output JSON
- Sample Dockerfile
- Sample solution