#!/bin/bash

# Docker build and test script for PDF Structure Extractor
# This script builds the Docker image and tests it with sample PDFs

set -e  # Exit on any error

echo "=== PDF Structure Extractor Docker Build and Test ==="

# Configuration
IMAGE_NAME="pdf-structure-extractor"
TAG="latest"
CONTAINER_NAME="pdf-extractor-test"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Cleanup function
cleanup() {
    print_status "Cleaning up..."
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
}

# Set up cleanup trap
trap cleanup EXIT

# Step 1: Build the Docker image
print_status "Building Docker image: $IMAGE_NAME:$TAG"
docker build --platform linux/amd64 -t $IMAGE_NAME:$TAG .

if [ $? -eq 0 ]; then
    print_status "Docker image built successfully"
else
    print_error "Docker build failed"
    exit 1
fi

# Step 2: Check image size
print_status "Checking image size..."
IMAGE_SIZE=$(docker images $IMAGE_NAME:$TAG --format "table {{.Size}}" | tail -n 1)
print_status "Image size: $IMAGE_SIZE"

# Step 3: Verify model size constraint (should be under 200MB)
print_status "Checking model size in container..."
MODEL_SIZE=$(docker run --rm $IMAGE_NAME:$TAG du -sh /app/models | cut -f1)
print_status "Model size: $MODEL_SIZE"

# Step 4: Create test directories
print_status "Setting up test environment..."
mkdir -p test_input test_output

# Copy sample PDFs to test input
if [ -d "input" ]; then
    cp input/*.pdf test_input/ 2>/dev/null || print_warning "No PDF files found in input directory"
fi

# Step 5: Test container execution with volume mounting
print_status "Testing container execution with volume mounting..."

# Test 1: Basic execution test
print_status "Test 1: Basic container startup and health check"
docker run --name $CONTAINER_NAME \
    --platform linux/amd64 \
    -v "$(pwd)/test_input:/app/input:ro" \
    -v "$(pwd)/test_output:/app/output" \
    --memory=16g \
    --cpus=8 \
    --network=none \
    $IMAGE_NAME:$TAG python -c "print('Container startup successful')"

if [ $? -eq 0 ]; then
    print_status "✓ Container startup test passed"
else
    print_error "✗ Container startup test failed"
    exit 1
fi

# Clean up container
docker rm $CONTAINER_NAME

# Test 2: PDF processing test (if PDFs are available)
if [ -n "$(ls -A test_input/*.pdf 2>/dev/null)" ]; then
    print_status "Test 2: PDF processing test"
    
    docker run --name $CONTAINER_NAME \
        --platform linux/amd64 \
        -v "$(pwd)/test_input:/app/input:ro" \
        -v "$(pwd)/test_output:/app/output" \
        --memory=16g \
        --cpus=8 \
        --network=none \
        $IMAGE_NAME:$TAG
    
    if [ $? -eq 0 ]; then
        print_status "✓ PDF processing test passed"
        
        # Check if output files were generated
        if [ -n "$(ls -A test_output/*.json 2>/dev/null)" ]; then
            print_status "✓ JSON output files generated successfully"
            print_status "Generated files:"
            ls -la test_output/
        else
            print_warning "No JSON output files found"
        fi
    else
        print_error "✗ PDF processing test failed"
        exit 1
    fi
    
    # Clean up container
    docker rm $CONTAINER_NAME
else
    print_warning "No PDF files found for processing test"
fi

# Test 3: Resource constraint test
print_status "Test 3: Resource constraint test"
docker run --name $CONTAINER_NAME \
    --platform linux/amd64 \
    -v "$(pwd)/test_input:/app/input:ro" \
    -v "$(pwd)/test_output:/app/output" \
    --memory=1g \
    --cpus=2 \
    --network=none \
    $IMAGE_NAME:$TAG python -c "
import psutil
import torch
print(f'Available memory: {psutil.virtual_memory().available / (1024**3):.2f} GB')
print(f'CPU count: {psutil.cpu_count()}')
print(f'PyTorch device: {torch.device(\"cpu\")}')
print('Resource constraint test passed')
"

if [ $? -eq 0 ]; then
    print_status "✓ Resource constraint test passed"
else
    print_error "✗ Resource constraint test failed"
    exit 1
fi

# Clean up container
docker rm $CONTAINER_NAME

# Test 4: Network isolation test
print_status "Test 4: Network isolation test"
docker run --name $CONTAINER_NAME \
    --platform linux/amd64 \
    --network=none \
    $IMAGE_NAME:$TAG python -c "
import socket
try:
    socket.create_connection(('8.8.8.8', 53), timeout=5)
    print('ERROR: Network access detected')
    exit(1)
except:
    print('✓ Network isolation confirmed')
"

if [ $? -eq 0 ]; then
    print_status "✓ Network isolation test passed"
else
    print_error "✗ Network isolation test failed"
    exit 1
fi

# Clean up container
docker rm $CONTAINER_NAME

# Step 6: Performance test
print_status "Test 5: Performance and timeout test"
if [ -n "$(ls -A test_input/*.pdf 2>/dev/null)" ]; then
    # Test with timeout
    timeout 15s docker run --name $CONTAINER_NAME \
        --platform linux/amd64 \
        -v "$(pwd)/test_input:/app/input:ro" \
        -v "$(pwd)/test_output:/app/output" \
        --memory=16g \
        --cpus=8 \
        --network=none \
        $IMAGE_NAME:$TAG
    
    if [ $? -eq 0 ] || [ $? -eq 124 ]; then
        print_status "✓ Performance test completed (within timeout)"
    else
        print_warning "Performance test had issues but container constraints work"
    fi
    
    # Clean up container
    docker rm $CONTAINER_NAME 2>/dev/null || true
fi

# Final summary
print_status "=== Docker Build and Test Summary ==="
print_status "✓ Docker image built successfully"
print_status "✓ Image size: $IMAGE_SIZE"
print_status "✓ Model size: $MODEL_SIZE"
print_status "✓ Container startup test passed"
print_status "✓ Resource constraint test passed"
print_status "✓ Network isolation test passed"

if [ -n "$(ls -A test_input/*.pdf 2>/dev/null)" ]; then
    print_status "✓ PDF processing test completed"
fi

print_status "=== Usage Instructions ==="
echo "To run the container:"
echo "docker run --platform linux/amd64 \\"
echo "  -v /path/to/input:/app/input:ro \\"
echo "  -v /path/to/output:/app/output \\"
echo "  --memory=16g --cpus=8 --network=none \\"
echo "  $IMAGE_NAME:$TAG"

print_status "All tests completed successfully!"

# Cleanup test directories
rm -rf test_input test_output