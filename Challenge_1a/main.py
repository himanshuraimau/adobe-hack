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

# Import configuration first (no external dependencies)
try:
    from src.pdf_extractor.config.config import config
except ImportError as e:
    print(f"❌ Error: Could not import configuration: {e}")
    print("Please ensure you're running from the project root directory.")
    sys.exit(1)

# Import performance profiler
try:
    from scripts.performance_profiler import get_global_profiler, start_global_monitoring, stop_global_monitoring
except ImportError as e:
    print(f"❌ Error: Could not import performance profiler: {e}")
    print("Performance monitoring will be disabled.")
    
    # Create dummy functions
    def get_global_profiler():
        class DummyProfiler:
            def profile_operation(self, name):
                from contextlib import contextmanager
                @contextmanager
                def dummy_context():
                    yield
                return dummy_context()
            def get_performance_report(self):
                return {}
        return DummyProfiler()
    
    def start_global_monitoring():
        pass
    
    def stop_global_monitoring():
        pass

# Import core modules with better error handling
def import_core_modules():
    """Import core modules with detailed error messages for missing dependencies."""
    missing_deps = []
    
    try:
        from src.pdf_extractor.core.pdf_parser import PDFParser
    except ImportError as e:
        if "fitz" in str(e):
            missing_deps.append("pymupdf (for PDF parsing)")
        else:
            missing_deps.append(f"pdf_parser: {e}")
        PDFParser = None
    
    try:
        from src.pdf_extractor.core.preprocessor import TextPreprocessor
    except ImportError as e:
        missing_deps.append(f"preprocessor: {e}")
        TextPreprocessor = None
    
    try:
        from src.pdf_extractor.core.feature_extractor import FeatureExtractor
    except ImportError as e:
        missing_deps.append(f"feature_extractor: {e}")
        FeatureExtractor = None
    
    try:
        from src.pdf_extractor.core.classifier import HeadingClassifier
    except ImportError as e:
        if "torch" in str(e):
            missing_deps.append("torch (for ML classification)")
        elif "transformers" in str(e):
            missing_deps.append("transformers (for BERT model)")
        else:
            missing_deps.append(f"classifier: {e}")
        HeadingClassifier = None
    
    try:
        from src.pdf_extractor.core.structure_analyzer import StructureAnalyzer
    except ImportError as e:
        missing_deps.append(f"structure_analyzer: {e}")
        StructureAnalyzer = None
    
    try:
        from src.pdf_extractor.core.json_builder import JSONBuilder
    except ImportError as e:
        missing_deps.append(f"json_builder: {e}")
        JSONBuilder = None
    
    if missing_deps:
        print("❌ Missing dependencies detected:")
        for dep in missing_deps:
            print(f"   • {dep}")
        print("\n💡 To install dependencies:")
        print("   uv sync                    # Recommended")
        print("   # OR")
        print("   pip install -e .          # Alternative")
        print("\n📦 Required packages:")
        print("   • pymupdf>=1.26.3         # PDF processing")
        print("   • torch>=2.7.1            # Machine learning")
        print("   • transformers>=4.54.0    # BERT model")
        print("   • numpy>=2.3.2            # Numerical computing")
        print("   • psutil>=7.0.0           # System monitoring")
        sys.exit(1)
    
    return PDFParser, TextPreprocessor, FeatureExtractor, HeadingClassifier, StructureAnalyzer, JSONBuilder

# Import core modules
PDFParser, TextPreprocessor, FeatureExtractor, HeadingClassifier, StructureAnalyzer, JSONBuilder = import_core_modules()


def setup_logging():
    """Set up comprehensive logging configuration."""
    import logging.handlers
    
    logging_config = config.get_logging_config()
    
    # Create logs directory if it doesn't exist
    logs_dir = config.base_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, logging_config["level"]))
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(logging_config["format"])
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Error log handler (for ERROR and CRITICAL messages)
    if logging_config.get("error_log_file"):
        error_handler = logging.handlers.RotatingFileHandler(
            logging_config["error_log_file"],
            maxBytes=logging_config["max_file_size"],
            backupCount=logging_config["backup_count"]
        )
        error_handler.setLevel(logging.ERROR)
        error_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
        error_handler.setFormatter(error_formatter)
        root_logger.addHandler(error_handler)
    
    # Debug log handler (for all messages when in debug mode)
    if logging_config.get("debug_log_file") and logging_config["level"] == "DEBUG":
        debug_handler = logging.handlers.RotatingFileHandler(
            logging_config["debug_log_file"],
            maxBytes=logging_config["max_file_size"],
            backupCount=logging_config["backup_count"]
        )
        debug_handler.setLevel(logging.DEBUG)
        debug_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
        debug_handler.setFormatter(debug_formatter)
        root_logger.addHandler(debug_handler)


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


def process_pdf(pdf_path: str, output_dir: str, timeout: int = None) -> Optional[str]:
    """
    Process a single PDF file through the complete pipeline.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save output JSON
        timeout: Processing timeout in seconds (optional)
        
    Returns:
        Path to generated JSON file or None if processing failed
    """
    import signal
    import time
    from pathlib import Path
    
    logger = logging.getLogger(__name__)
    logger.info(f"Processing PDF: {pdf_path}")
    
    # Set up timeout handling
    if timeout is None:
        timeout = config.processing_timeout
    
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Processing exceeded {timeout} second limit")
    
    # Set up signal handler for timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    
    pdf_parser = None
    profiler = get_global_profiler()
    
    try:
        start_time = time.time()
        
        # Start performance monitoring
        start_global_monitoring()
        
        with profiler.profile_operation("component_initialization"):
            # Initialize components
            pdf_parser = PDFParser()
            preprocessor = TextPreprocessor()
            feature_extractor = FeatureExtractor()
            classifier = HeadingClassifier()
            structure_analyzer = StructureAnalyzer()
            json_builder = JSONBuilder()
        
        # Load classification model
        logger.info("Loading MobileBERT model...")
        with profiler.profile_operation("model_loading"):
            classifier.load_model(str(config.mobilebert_model_path))
        
        # Step 1: Parse PDF document
        logger.info("Parsing PDF document...")
        with profiler.profile_operation("pdf_parsing"):
            try:
                text_blocks = pdf_parser.parse_document(pdf_path)
                if not text_blocks:
                    logger.warning("No text blocks extracted from PDF")
                    return _generate_error_output(pdf_path, output_dir, "No text content found")
            except Exception as e:
                logger.error(f"PDF parsing failed: {e}")
                return _generate_error_output(pdf_path, output_dir, f"PDF parsing failed: {e}")
        
        logger.info(f"Extracted {len(text_blocks)} text blocks")
        
        # Step 2: Preprocess text blocks
        logger.info("Preprocessing text blocks...")
        with profiler.profile_operation("text_preprocessing"):
            try:
                processed_blocks = preprocessor.preprocess_blocks(text_blocks)
                if not processed_blocks:
                    logger.warning("No processed blocks after preprocessing")
                    return _generate_error_output(pdf_path, output_dir, "No content after preprocessing")
            except Exception as e:
                logger.error(f"Text preprocessing failed: {e}")
                return _generate_error_output(pdf_path, output_dir, f"Text preprocessing failed: {e}")
        
        logger.info(f"Preprocessed to {len(processed_blocks)} blocks")
        
        # Step 3: Initialize feature extractor with document statistics
        logger.info("Initializing feature extraction...")
        with profiler.profile_operation("feature_initialization"):
            try:
                feature_extractor.initialize_document_stats(processed_blocks)
            except Exception as e:
                logger.error(f"Feature initialization failed: {e}")
                return _generate_error_output(pdf_path, output_dir, f"Feature initialization failed: {e}")
        
        # Step 4: Extract features and classify blocks
        logger.info("Extracting features and classifying blocks...")
        with profiler.profile_operation("feature_extraction_and_classification"):
            try:
                classification_results = []
                
                # Process in batches for better memory management and potential parallelization
                batch_size = 50  # Process 50 blocks at a time
                
                for i in range(0, len(processed_blocks), batch_size):
                    batch = processed_blocks[i:i + batch_size]
                    
                    # Extract features for the batch with error handling
                    for block in batch:
                        try:
                            features = feature_extractor.extract_features(block)
                            block.features = features
                        except Exception as e:
                            logger.warning(f"Feature extraction failed for block, using defaults: {e}")
                            # Use default features as fallback
                            from src.pdf_extractor.core.error_handler import global_error_handler
                            default_features = global_error_handler.handle_feature_extraction_error(e, block.text)
                            from src.pdf_extractor.models.models import FeatureVector
                            block.features = FeatureVector(**default_features)
                    
                    # Classify the batch
                    texts_and_features = [(block.text, block.features) for block in batch]
                    
                    # Use batch prediction if available and batch size > 1
                    if len(batch) > 1 and hasattr(classifier.model_adapter, 'predict_batch'):
                        try:
                            batch_results = classifier.model_adapter.predict_batch(texts_and_features)
                            
                            for block, (predicted_class, confidence) in zip(batch, batch_results):
                                from src.pdf_extractor.models.models import ClassificationResult
                                result = ClassificationResult(
                                    block=block,
                                    predicted_class=predicted_class,
                                    confidence=confidence
                                )
                                classification_results.append(result)
                        except Exception as e:
                            logger.warning(f"Batch prediction failed, falling back to individual: {e}")
                            # Fallback to individual classification
                            for block in batch:
                                try:
                                    result = classifier.classify_block(block.features, block.text)
                                    result.block = block
                                    classification_results.append(result)
                                except Exception as classify_error:
                                    logger.warning(f"Individual classification failed for block: {classify_error}")
                                    # Create minimal result as ultimate fallback
                                    from src.pdf_extractor.models.models import ClassificationResult
                                    result = ClassificationResult(
                                        block=block,
                                        predicted_class="text",
                                        confidence=0.1
                                    )
                                    classification_results.append(result)
                    else:
                        # Individual classification for small batches or when batch prediction unavailable
                        for block in batch:
                            try:
                                result = classifier.classify_block(block.features, block.text)
                                result.block = block
                                classification_results.append(result)
                            except Exception as classify_error:
                                logger.warning(f"Individual classification failed for block: {classify_error}")
                                # Create minimal result as ultimate fallback
                                from src.pdf_extractor.models.models import ClassificationResult
                                result = ClassificationResult(
                                    block=block,
                                    predicted_class="text",
                                    confidence=0.1
                                )
                                classification_results.append(result)
                
                if not classification_results:
                    logger.error("No classification results generated")
                    return _generate_error_output(pdf_path, output_dir, "Classification failed for all blocks")
                    
            except Exception as e:
                logger.error(f"Feature extraction and classification failed: {e}")
                return _generate_error_output(pdf_path, output_dir, f"Feature extraction and classification failed: {e}")
        
        logger.info(f"Classified {len(classification_results)} blocks")
        
        # Step 5: Analyze document structure
        logger.info("Analyzing document structure...")
        with profiler.profile_operation("structure_analysis"):
            try:
                document_structure = structure_analyzer.analyze_structure(classification_results)
            except Exception as e:
                logger.error(f"Structure analysis failed: {e}")
                # Use error handler to create minimal structure
                from src.pdf_extractor.core.error_handler import global_error_handler
                document_structure = global_error_handler.handle_structure_analysis_error(e, classification_results)
        
        logger.info(f"Found title: {document_structure.title}")
        logger.info(f"Found {len(document_structure.headings)} headings")
        
        # Step 6: Generate JSON output
        logger.info("Generating JSON output...")
        with profiler.profile_operation("json_generation"):
            try:
                output_filename = Path(pdf_path).stem + ".json"
                output_path = Path(output_dir) / output_filename
                
                json_data = json_builder.process_and_write(document_structure, str(output_path))
            except Exception as e:
                logger.error(f"JSON generation failed: {e}")
                # Use error handler to create fallback JSON
                from src.pdf_extractor.core.error_handler import global_error_handler
                error_json = global_error_handler.handle_output_generation_error(e, document_structure)
                
                # Try to write the error JSON
                try:
                    output_filename = Path(pdf_path).stem + ".json"
                    output_path = Path(output_dir) / output_filename
                    json_builder.write_json(error_json, str(output_path))
                    json_data = error_json
                except Exception as write_error:
                    logger.error(f"Failed to write error JSON: {write_error}")
                    return _generate_error_output(pdf_path, output_dir, f"JSON generation and error handling failed: {e}")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        logger.info(f"Processing completed in {processing_time:.2f} seconds")
        
        # Log performance summary
        if processing_time > 5.0:  # Log detailed performance if slow
            performance_report = profiler.get_performance_report()
            logger.warning(f"Slow processing detected. Performance summary:")
            for op_name, stats in performance_report.get('operation_stats', {}).items():
                logger.warning(f"  {op_name}: {stats['avg_duration']:.2f}s avg, {stats['max_memory_peak']:.1f}MB peak")
        
        # Cancel the alarm
        signal.alarm(0)
        
        return str(output_path)
        
    except TimeoutError as e:
        logger.error(f"Processing timeout: {e}")
        signal.alarm(0)  # Cancel alarm
        return _generate_timeout_output(pdf_path, output_dir)
        
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
        signal.alarm(0)  # Cancel alarm
        return _generate_error_output(pdf_path, output_dir, str(e))
        
    finally:
        # Clean up resources
        if pdf_parser:
            pdf_parser.close()
        # Ensure alarm is cancelled
        signal.alarm(0)


def _generate_error_output(pdf_path: str, output_dir: str, error_message: str) -> Optional[str]:
    """
    Generate error JSON output when processing fails.
    
    Args:
        pdf_path: Path to the PDF file that failed
        output_dir: Output directory
        error_message: Error message to include
        
    Returns:
        Path to generated error JSON file
    """
    logger = logging.getLogger(__name__)
    
    try:
        try:
            from src.pdf_extractor.core.json_builder import JSONBuilder
        except ImportError:
            from src.pdf_extractor.core.json_builder import JSONBuilder
        
        json_builder = JSONBuilder()
        error_json = {
            "title": f"Error: Failed to process {Path(pdf_path).name}",
            "outline": [],
            "error": {
                "type": "ProcessingError",
                "message": error_message,
                "details": f"Failed to process PDF: {pdf_path}"
            }
        }
        
        output_filename = Path(pdf_path).stem + ".json"
        output_path = Path(output_dir) / output_filename
        
        json_builder.write_json(error_json, str(output_path))
        logger.info(f"Generated error output: {output_path}")
        
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Failed to generate error output: {e}")
        return None


def _generate_timeout_output(pdf_path: str, output_dir: str) -> Optional[str]:
    """
    Generate timeout JSON output when processing times out.
    
    Args:
        pdf_path: Path to the PDF file that timed out
        output_dir: Output directory
        
    Returns:
        Path to generated timeout JSON file
    """
    logger = logging.getLogger(__name__)
    
    try:
        try:
            from src.pdf_extractor.core.json_builder import JSONBuilder
        except ImportError:
            from src.pdf_extractor.core.json_builder import JSONBuilder
        
        json_builder = JSONBuilder()
        timeout_json = {
            "title": f"Error: Processing timeout for {Path(pdf_path).name}",
            "outline": [],
            "error": {
                "type": "TimeoutError",
                "message": f"Processing exceeded {config.processing_timeout} second limit",
                "details": f"PDF processing timed out: {pdf_path}"
            }
        }
        
        output_filename = Path(pdf_path).stem + ".json"
        output_path = Path(output_dir) / output_filename
        
        json_builder.write_json(timeout_json, str(output_path))
        logger.info(f"Generated timeout output: {output_path}")
        
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Failed to generate timeout output: {e}")
        return None


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
        result = process_pdf(str(input_path), str(output_dir), args.timeout)
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
            result = process_pdf(str(pdf_file), str(output_dir), args.timeout)
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