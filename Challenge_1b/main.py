#!/usr/bin/env python3
"""Main entry point for the PDF Analysis System."""

import os
import sys
import logging
import time
import traceback
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pdf_analysis.collection_processor import CollectionProcessor


def setup_logging(log_level: str = "INFO") -> None:
    """Configure comprehensive logging for the application."""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Set up file and console handlers
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.FileHandler(log_dir / "pdf_analysis.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific log levels for external libraries to reduce noise
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)


def check_system_requirements() -> bool:
    """Check if system has required dependencies and resources."""
    try:
        # Check Python version
        if sys.version_info < (3, 8):
            logging.error("Python 3.8 or higher is required")
            return False
        
        # Check required packages
        required_packages = [
            "fitz",  # PyMuPDF
            "sentence_transformers",
            "sklearn",
            "numpy"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logging.error(f"Missing required packages: {missing_packages}")
            logging.error("Please install with: pip install -r requirements.txt")
            return False
        
        # Check available memory (basic check)
        import psutil
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        if available_memory_gb < 2:
            logging.warning(f"Low available memory: {available_memory_gb:.1f}GB. Processing may be slow.")
        
        logging.info("System requirements check passed")
        return True
        
    except Exception as e:
        logging.error(f"Error checking system requirements: {e}")
        return False


def handle_critical_error(error: Exception, context: str) -> None:
    """Handle critical errors with detailed logging and graceful shutdown."""
    logging.critical(f"Critical error in {context}: {error}")
    logging.critical(f"Error type: {type(error).__name__}")
    logging.critical(f"Traceback:\n{traceback.format_exc()}")
    
    # Provide user-friendly error messages
    if isinstance(error, ImportError):
        logging.critical("Missing required dependencies. Please run: pip install -r requirements.txt")
    elif isinstance(error, FileNotFoundError):
        logging.critical("Required files or directories not found. Check your workspace structure.")
    elif isinstance(error, MemoryError):
        logging.critical("Out of memory. Try processing fewer documents or use a machine with more RAM.")
    else:
        logging.critical("An unexpected error occurred. Check the logs for details.")


def optimize_performance() -> None:
    """Apply performance optimizations for CPU-only processing."""
    try:
        # Set environment variables for CPU optimization
        os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer warnings
        os.environ["OMP_NUM_THREADS"] = str(min(4, os.cpu_count() or 1))  # Limit OpenMP threads
        
        # Import torch and set CPU-only mode if available
        try:
            import torch
            torch.set_num_threads(min(4, os.cpu_count() or 1))
            if torch.cuda.is_available():
                logging.info("CUDA available but using CPU-only mode as per requirements")
        except ImportError:
            pass  # torch not available, continue without optimization
        
        logging.info("Performance optimizations applied")
        
    except Exception as e:
        logging.warning(f"Could not apply performance optimizations: {e}")


def main(base_path: str = ".", log_level: str = "INFO") -> int:
    """
    Main function to process PDF collections with comprehensive error handling.
    
    Args:
        base_path: Base directory to search for collections
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Exit code: 0 for success, 1 for failure
    """
    start_time = time.time()
    
    try:
        # Setup logging
        setup_logging(log_level)
        logging.info("=" * 60)
        logging.info("PDF Analysis System - Starting")
        logging.info("=" * 60)
        
        # Check system requirements
        if not check_system_requirements():
            logging.critical("System requirements not met. Exiting.")
            return 1
        
        # Apply performance optimizations
        optimize_performance()
        
        # Initialize collection processor with error handling
        try:
            processor = CollectionProcessor()
            logging.info("Collection processor initialized successfully")
        except Exception as e:
            handle_critical_error(e, "processor initialization")
            return 1
        
        # Process all collections
        try:
            logging.info(f"Starting collection discovery in: {base_path}")
            results = processor.process_all_collections(base_path)
            
            if not results:
                logging.warning("No collections found to process")
                return 0
            
            # Analyze results
            successful_collections = [r for r in results if r.success]
            failed_collections = [r for r in results if not r.success]
            
            total_sections = sum(r.sections_processed for r in successful_collections)
            total_documents = sum(r.documents_processed for r in successful_collections)
            
            # Log detailed summary
            logging.info("=" * 60)
            logging.info("PROCESSING SUMMARY")
            logging.info("=" * 60)
            logging.info(f"Total collections found: {len(results)}")
            logging.info(f"Successfully processed: {len(successful_collections)}")
            logging.info(f"Failed to process: {len(failed_collections)}")
            logging.info(f"Total documents processed: {total_documents}")
            logging.info(f"Total sections extracted: {total_sections}")
            
            # Log successful collections
            if successful_collections:
                logging.info("\nSuccessful collections:")
                for result in successful_collections:
                    logging.info(f"  ✓ {result.collection_path} - "
                               f"{result.documents_processed} docs, "
                               f"{result.sections_processed} sections")
            
            # Log failed collections
            if failed_collections:
                logging.error("\nFailed collections:")
                for result in failed_collections:
                    logging.error(f"  ✗ {result.collection_path} - {result.error_message}")
            
            # Calculate processing time
            end_time = time.time()
            processing_time = end_time - start_time
            logging.info(f"\nTotal processing time: {processing_time:.2f} seconds")
            
            if total_documents > 0:
                avg_time_per_doc = processing_time / total_documents
                logging.info(f"Average time per document: {avg_time_per_doc:.2f} seconds")
            
            # Determine exit code
            if failed_collections:
                logging.warning("Some collections failed to process. Check logs for details.")
                return 1 if len(failed_collections) == len(results) else 0
            else:
                logging.info("All collections processed successfully!")
                return 0
                
        except KeyboardInterrupt:
            logging.warning("Processing interrupted by user")
            return 1
        except Exception as e:
            handle_critical_error(e, "collection processing")
            return 1
            
    except Exception as e:
        # Handle any unexpected errors in main setup
        try:
            handle_critical_error(e, "main function setup")
        except:
            # If logging fails, print to stderr
            print(f"Critical error: {e}", file=sys.stderr)
            traceback.print_exc()
        return 1
    
    finally:
        # Cleanup logging handlers
        try:
            logging.shutdown()
        except:
            pass


if __name__ == "__main__":
    import argparse
    
    # Add command line argument parsing for flexibility
    parser = argparse.ArgumentParser(description="PDF Analysis System")
    parser.add_argument("--base-path", default=".", 
                       help="Base directory to search for collections (default: current directory)")
    parser.add_argument("--log-level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Logging level (default: INFO)")
    
    args = parser.parse_args()
    
    # Run main function and exit with appropriate code
    exit_code = main(args.base_path, args.log_level)
    sys.exit(exit_code)