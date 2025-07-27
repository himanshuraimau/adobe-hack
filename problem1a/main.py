"""
Main entry point for the PDF Structure Extractor application.

This module orchestrates the complete PDF processing pipeline from input
to JSON output generation, handling command-line arguments and error cases.
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

from .config import config
from .pdf_parser import PDFParser
from .preprocessor import TextPreprocessor
from .feature_extractor import FeatureExtractor
from .classifier import HeadingClassifier
from .structure_analyzer import StructureAnalyzer
from .json_builder import JSONBuilder


def setup_logging():
    """Set up logging configuration."""
    logging_config = config.get_logging_config()
    logging.basicConfig(
        level=getattr(logging, logging_config["level"]),
        format=logging_config["format"],
        filename=logging_config["file"]
    )


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Extract structured outline from PDF documents"
    )
    parser.add_argument(
        "input_path",
        help="Path to input PDF file or directory"
    )
    parser.add_argument(
        "-o", "--output",
        default=str(config.output_dir),
        help="Output directory for JSON files"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=config.processing_timeout,
        help="Processing timeout in seconds"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def process_pdf(pdf_path: str, output_dir: str) -> Optional[str]:
    """
    Process a single PDF file through the complete pipeline.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save output JSON
        
    Returns:
        Path to generated JSON file or None if processing failed
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Processing PDF: {pdf_path}")
    
    try:
        # Initialize components
        pdf_parser = PDFParser()
        preprocessor = TextPreprocessor()
        feature_extractor = FeatureExtractor()
        classifier = HeadingClassifier()
        structure_analyzer = StructureAnalyzer()
        json_builder = JSONBuilder()
        
        # Load classification model
        classifier.load_model(str(config.mobilebert_model_path))
        
        # Process PDF through pipeline
        # Implementation will be completed in task 8
        logger.info("PDF processing pipeline not yet implemented")
        return None
        
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
        return None
    finally:
        # Clean up resources
        if 'pdf_parser' in locals():
            pdf_parser.close()


def main():
    """Main application entry point."""
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Parse command-line arguments
    args = parse_arguments()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Ensure required directories exist
    config.ensure_directories()
    
    # Process input
    input_path = Path(args.input_path)
    output_dir = Path(args.output)
    
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if input_path.is_file() and input_path.suffix.lower() == '.pdf':
        # Process single PDF file
        result = process_pdf(str(input_path), str(output_dir))
        if result:
            logger.info(f"Successfully processed PDF. Output: {result}")
        else:
            logger.error("Failed to process PDF")
            sys.exit(1)
    
    elif input_path.is_dir():
        # Process all PDF files in directory
        pdf_files = list(input_path.glob("*.pdf"))
        if not pdf_files:
            logger.error(f"No PDF files found in directory: {input_path}")
            sys.exit(1)
        
        success_count = 0
        for pdf_file in pdf_files:
            result = process_pdf(str(pdf_file), str(output_dir))
            if result:
                success_count += 1
        
        logger.info(f"Processed {success_count}/{len(pdf_files)} PDF files successfully")
        
        if success_count == 0:
            sys.exit(1)
    
    else:
        logger.error(f"Invalid input path: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()