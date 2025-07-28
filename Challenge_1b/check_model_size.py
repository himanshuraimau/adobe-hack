#!/usr/bin/env python3
"""Check model size to verify it meets the <1GB constraint."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pdf_analysis import SemanticRanker
import psutil
import gc

def check_model_size():
    """Check if the model meets the size constraint."""
    print("Checking model size constraint...")
    
    # Get initial memory usage
    process = psutil.Process()
    initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
    
    # Load model
    ranker = SemanticRanker()
    ranker.load_model()
    
    # Get memory after loading model
    final_memory = process.memory_info().rss / (1024 * 1024)  # MB
    model_memory = final_memory - initial_memory
    
    print(f"Initial memory: {initial_memory:.1f} MB")
    print(f"Final memory: {final_memory:.1f} MB")
    print(f"Model memory usage: {model_memory:.1f} MB")
    
    # Check constraint (1GB = 1024 MB)
    if model_memory < 1024:
        print(f"✓ Model size constraint met: {model_memory:.1f} MB < 1024 MB")
        return True
    else:
        print(f"✗ Model size constraint violated: {model_memory:.1f} MB >= 1024 MB")
        return False

if __name__ == "__main__":
    success = check_model_size()
    sys.exit(0 if success else 1)