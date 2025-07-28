#!/usr/bin/env python3
"""
Simple performance testing script for individual components.
"""

import time
import logging
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_pdf_parsing_performance():
    """Test PDF parsing performance."""
    from pdf_parser import PDFParser
    
    input_dir = Path("input")
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.error("No PDF files found for testing")
        return
    
    logger.info("Testing PDF parsing performance...")
    
    for pdf_file in pdf_files:
        logger.info(f"Testing with {pdf_file.name}")
        
        parser = PDFParser()
        start_time = time.time()
        
        try:
            blocks = parser.parse_document(str(pdf_file))
            end_time = time.time()
            
            logger.info(f"  Parsed {len(blocks)} blocks in {end_time - start_time:.3f}s")
            
        except Exception as e:
            logger.error(f"  Error: {e}")
        finally:
            parser.close()


def test_preprocessing_performance():
    """Test preprocessing performance."""
    from pdf_parser import PDFParser
    from preprocessor import TextPreprocessor
    
    input_dir = Path("input")
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.error("No PDF files found for testing")
        return
    
    logger.info("Testing preprocessing performance...")
    
    for pdf_file in pdf_files:
        logger.info(f"Testing with {pdf_file.name}")
        
        # Parse PDF first
        parser = PDFParser()
        try:
            blocks = parser.parse_document(str(pdf_file))
            
            # Test preprocessing
            preprocessor = TextPreprocessor()
            start_time = time.time()
            
            processed_blocks = preprocessor.preprocess_blocks(blocks)
            end_time = time.time()
            
            logger.info(f"  Preprocessed {len(blocks)} -> {len(processed_blocks)} blocks in {end_time - start_time:.3f}s")
            
        except Exception as e:
            logger.error(f"  Error: {e}")
        finally:
            parser.close()


def test_feature_extraction_performance():
    """Test feature extraction performance."""
    from pdf_parser import PDFParser
    from preprocessor import TextPreprocessor
    from feature_extractor import FeatureExtractor
    
    input_dir = Path("input")
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.error("No PDF files found for testing")
        return
    
    logger.info("Testing feature extraction performance...")
    
    for pdf_file in pdf_files:
        logger.info(f"Testing with {pdf_file.name}")
        
        parser = PDFParser()
        try:
            # Parse and preprocess
            blocks = parser.parse_document(str(pdf_file))
            preprocessor = TextPreprocessor()
            processed_blocks = preprocessor.preprocess_blocks(blocks)
            
            # Test feature extraction
            extractor = FeatureExtractor()
            start_time = time.time()
            
            extractor.initialize_document_stats(processed_blocks)
            
            features_extracted = 0
            for block in processed_blocks:
                features = extractor.extract_features(block)
                features_extracted += 1
            
            end_time = time.time()
            
            logger.info(f"  Extracted features for {features_extracted} blocks in {end_time - start_time:.3f}s")
            
        except Exception as e:
            logger.error(f"  Error: {e}")
        finally:
            parser.close()


def test_model_loading_performance():
    """Test model loading performance."""
    from classifier import HeadingClassifier
    from config import config
    
    logger.info("Testing model loading performance...")
    
    classifier = HeadingClassifier()
    start_time = time.time()
    
    try:
        classifier.load_model(str(config.mobilebert_model_path))
        end_time = time.time()
        
        logger.info(f"  Model loaded in {end_time - start_time:.3f}s")
        
    except Exception as e:
        logger.error(f"  Error loading model: {e}")


def main():
    """Run all performance tests."""
    logger.info("Starting component performance tests...")
    
    # Test individual components
    test_pdf_parsing_performance()
    test_preprocessing_performance()
    test_feature_extraction_performance()
    test_model_loading_performance()
    
    logger.info("Component performance tests completed")


if __name__ == "__main__":
    main()