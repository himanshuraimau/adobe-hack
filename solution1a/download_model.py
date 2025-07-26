#!/usr/bin/env python3
"""
One-time script to download MobileBERT model for PDF outline extraction.
This downloads approximately 200MB of model files.

Usage:
    python download_model.py

The model will be saved to ./models/local_mobilebert/
"""

import os
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

def download_mobilebert():
    """Download MobileBERT model and tokenizer to local directory."""
    
    # Define paths
    models_dir = Path("models")
    model_dir = models_dir / "local_mobilebert"
    
    # Create directories
    models_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)
    
    print("📥 Downloading MobileBERT model and tokenizer...")
    print("📁 Target directory: ./models/local_mobilebert/")
    print("⏳ This may take a few minutes (~200MB download)...")
    
    try:
        # Download tokenizer
        print("\n🔤 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
        tokenizer.save_pretrained(model_dir)
        
        # Download model
        print("🤖 Downloading model...")
        model = AutoModel.from_pretrained("google/mobilebert-uncased")
        model.save_pretrained(model_dir)
        
        print(f"\n✅ Successfully downloaded MobileBERT to {model_dir}")
        print("📁 Model files:")
        for file in sorted(model_dir.iterdir()):
            if file.is_file():
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"   {file.name} ({size_mb:.1f} MB)")
        
        print("\n🚀 Model ready! You can now run: python run.py")
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        print("🔗 Please check your internet connection and try again.")
        sys.exit(1)

if __name__ == "__main__":
    download_mobilebert()
