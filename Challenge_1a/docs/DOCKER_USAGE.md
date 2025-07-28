# Docker Usage Guide for PDF Structure Extractor

## Overview

The PDF Structure Extractor is containerized using Docker with the following specifications:
- **Platform**: AMD64 (linux/amd64)
- **CPU-only operation**: No GPU requirements
- **Network isolation**: Runs without internet access
- **Resource constraints**: Optimized for 8 CPUs and 16GB RAM
- **Model size**: Under 200MB (95MB actual)

## Building the Docker Image

```bash
# Build the Docker image from project root (recommended)
docker build --platform linux/amd64 -f docker/Dockerfile -t pdf-structure-extractor:latest .

# Or use the automated build and test script
cd docker
./docker-build-test.sh

# Check image size
docker images pdf-structure-extractor:latest
```

## Running the Container

### Basic Usage

```bash
docker run --platform linux/amd64 \
  -v /path/to/input:/app/input:ro \
  -v /path/to/output:/app/output \
  --memory=16g --cpus=8 --network=none \
  pdf-structure-extractor:latest
```

### Parameters Explanation

- `--platform linux/amd64`: Ensures AMD64 architecture
- `-v /path/to/input:/app/input:ro`: Mount input directory (read-only)
- `-v /path/to/output:/app/output`: Mount output directory (read-write)
- `--memory=16g`: Limit memory to 16GB
- `--cpus=8`: Limit to 8 CPU cores
- `--network=none`: Disable network access for security

### Processing Single PDF

```bash
# Create input and output directories (or use existing data directories)
mkdir -p data/input data/output

# Copy your PDF file to input directory
cp your-document.pdf data/input/

# Run the container
docker run --platform linux/amd64 \
  -v "$(pwd)/data/input:/app/input:ro" \
  -v "$(pwd)/data/output:/app/output" \
  --memory=16g --cpus=8 --network=none \
  pdf-structure-extractor:latest

# Check the output
ls -la data/output/
cat data/output/your-document.json
```

### Processing Multiple PDFs

```bash
# Place multiple PDF files in input directory
cp *.pdf data/input/

# Run the container (processes all PDFs in input directory)
docker run --platform linux/amd64 \
  -v "$(pwd)/data/input:/app/input:ro" \
  -v "$(pwd)/data/output:/app/output" \
  --memory=16g --cpus=8 --network=none \
  pdf-structure-extractor:latest
```

### Custom Timeout

```bash
# Set custom timeout (default is 10 seconds)
docker run --platform linux/amd64 \
  -v "$(pwd)/data/input:/app/input:ro" \
  -v "$(pwd)/data/output:/app/output" \
  --memory=16g --cpus=8 --network=none \
  pdf-structure-extractor:latest \
  python main.py /app/input --output /app/output --timeout 15
```

### Verbose Logging

```bash
# Enable verbose logging for debugging
docker run --platform linux/amd64 \
  -v "$(pwd)/data/input:/app/input:ro" \
  -v "$(pwd)/data/output:/app/output" \
  --memory=16g --cpus=8 --network=none \
  pdf-structure-extractor:latest \
  python main.py /app/input --output /app/output --verbose
```

## Output Format

The container generates JSON files with the following structure:

```json
{
  "title": "Document Title",
  "outline": [
    {
      "level": "H1",
      "text": "Main Heading",
      "page": 1
    },
    {
      "level": "H2", 
      "text": "Sub Heading",
      "page": 2
    }
  ]
}
```

## Error Handling

### Timeout Errors
If processing exceeds the timeout limit, the container generates:
```json
{
  "title": "Error: Processing timeout for document.pdf",
  "outline": [],
  "error": {
    "type": "TimeoutError",
    "message": "Processing exceeded 10 second limit",
    "details": "PDF processing timed out: /app/input/document.pdf"
  }
}
```

### Processing Errors
For corrupted or unreadable PDFs:
```json
{
  "title": "Error: Failed to process document.pdf",
  "outline": [],
  "error": {
    "type": "ProcessingError",
    "message": "Error details here",
    "details": "Failed to process PDF: /app/input/document.pdf"
  }
}
```

## Resource Monitoring

### Check Container Resource Usage

```bash
# Monitor container resources while running
docker stats pdf-extractor-container

# Run with resource monitoring
docker run --name pdf-extractor-container \
  --platform linux/amd64 \
  -v "$(pwd)/data/input:/app/input:ro" \
  -v "$(pwd)/data/output:/app/output" \
  --memory=16g --cpus=8 --network=none \
  pdf-structure-extractor:latest
```

### Memory and CPU Constraints

The container is optimized for:
- **Memory**: Up to 16GB (typically uses 400-600MB)
- **CPU**: Up to 8 cores (CPU-only PyTorch)
- **Processing time**: Under 10 seconds for 50-page documents
- **Model size**: 95MB (under 200MB limit)

## Testing the Container

Use the provided test script:

```bash
# Run comprehensive tests
./docker-build-test.sh
```

This script tests:
- Docker image build
- Container startup
- PDF processing
- Resource constraints
- Network isolation
- Performance benchmarks

## Troubleshooting

### Common Issues

1. **Permission Denied**
   ```bash
   # Fix volume permissions
   sudo chown -R $(id -u):$(id -g) data/input data/output
   ```

2. **Out of Memory**
   ```bash
   # Increase memory limit
   docker run --memory=32g ...
   ```

3. **Slow Processing**
   ```bash
   # Increase CPU allocation
   docker run --cpus=16 ...
   ```

4. **Network Access Error**
   ```bash
   # Ensure network isolation
   docker run --network=none ...
   ```

### Debug Mode

```bash
# Run with debug information
docker run --platform linux/amd64 \
  -v "$(pwd)/data/input:/app/input:ro" \
  -v "$(pwd)/data/output:/app/output" \
  --memory=16g --cpus=8 --network=none \
  pdf-structure-extractor:latest \
  python main.py /app/input --output /app/output --verbose
```

### Health Check

```bash
# Check container health
docker run --platform linux/amd64 \
  pdf-structure-extractor:latest \
  python -c "import sys; sys.path.append('/app'); from main import main; print('Container is healthy')"
```

## Security Features

- **Non-root user**: Container runs as user 'app'
- **Network isolation**: No internet access required
- **Read-only input**: Input directory mounted read-only
- **Minimal base image**: Python 3.12 slim for reduced attack surface
- **No unnecessary packages**: Only required dependencies included

## Performance Benchmarks

Based on testing with sample PDFs:
- **Small documents (1-5 pages)**: 1-2 seconds
- **Medium documents (10-20 pages)**: 2-5 seconds  
- **Large documents (30-50 pages)**: 5-10 seconds
- **Memory usage**: 400-600MB peak
- **Model loading time**: ~1 second

## Container Specifications

- **Base image**: python:3.12-slim
- **Architecture**: linux/amd64
- **Total size**: ~1.34GB
- **Model size**: 95MB
- **Python version**: 3.12
- **PyTorch**: CPU-only version
- **Dependencies**: PyMuPDF, transformers, numpy, psutil
## Usage
 Instructions

To run the container with your own PDFs:
```bash
docker run --platform linux/amd64 \
  -v "/path/to/your/input:/app/input:ro" \
  -v "/path/to/your/output:/app/output" \
  --memory=16g --cpus=8 --network=none \
  pdf-structure-extractor:latest
```

To run with the sample data:
```bash
docker run --platform linux/amd64 \
  -v "$(pwd)/data/input:/app/input:ro" \
  -v "$(pwd)/data/output:/app/output" \
  --memory=16g --cpus=8 --network=none \
  pdf-structure-extractor:latest
```

## Quick Start

1. **Build the image**:
   ```bash
   docker build --platform linux/amd64 -f docker/Dockerfile -t pdf-structure-extractor .
   ```

2. **Test with sample data**:
   ```bash
   docker run --platform linux/amd64 \
     -v "$(pwd)/data/input:/app/input:ro" \
     -v "$(pwd)/data/output:/app/output" \
     --memory=16g --cpus=8 --network=none \
     pdf-structure-extractor:latest
   ```

3. **Check results**:
   ```bash
   ls -la data/output/
   ```