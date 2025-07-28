#!/usr/bin/env python3
"""
Performance testing script for the PDF Structure Extractor.

This script tests the application performance with various document sizes
and generates performance reports to identify bottlenecks.
"""

import sys
import time
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from main import process_pdf
from scripts.performance_profiler import get_global_profiler, start_global_monitoring, stop_global_monitoring
from src.pdf_extractor.config.config import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_performance_with_sample_pdfs():
    """Test performance with the provided sample PDF files."""
    input_dir = Path("input")
    output_dir = Path("output")
    
    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)
    
    # Find all PDF files in input directory
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.error("No PDF files found in input directory")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files to test")
    
    # Start global performance monitoring
    profiler = get_global_profiler()
    start_global_monitoring()
    
    results = {}
    
    for pdf_file in pdf_files:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing performance with: {pdf_file.name}")
        logger.info(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Process the PDF
            result_path = process_pdf(str(pdf_file), str(output_dir), timeout=10)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            if result_path:
                logger.info(f"✅ Successfully processed {pdf_file.name} in {processing_time:.2f} seconds")
                results[pdf_file.name] = {
                    'success': True,
                    'processing_time': processing_time,
                    'output_path': result_path
                }
            else:
                logger.error(f"❌ Failed to process {pdf_file.name}")
                results[pdf_file.name] = {
                    'success': False,
                    'processing_time': processing_time,
                    'error': 'Processing failed'
                }
                
        except Exception as e:
            end_time = time.time()
            processing_time = end_time - start_time
            logger.error(f"❌ Error processing {pdf_file.name}: {e}")
            results[pdf_file.name] = {
                'success': False,
                'processing_time': processing_time,
                'error': str(e)
            }
    
    # Stop monitoring and generate report
    stop_global_monitoring()
    
    # Generate performance report
    logger.info(f"\n{'='*60}")
    logger.info("PERFORMANCE SUMMARY")
    logger.info(f"{'='*60}")
    
    total_files = len(results)
    successful_files = sum(1 for r in results.values() if r['success'])
    failed_files = total_files - successful_files
    
    logger.info(f"Total files processed: {total_files}")
    logger.info(f"Successful: {successful_files}")
    logger.info(f"Failed: {failed_files}")
    
    if successful_files > 0:
        successful_times = [r['processing_time'] for r in results.values() if r['success']]
        avg_time = sum(successful_times) / len(successful_times)
        max_time = max(successful_times)
        min_time = min(successful_times)
        
        logger.info(f"Average processing time: {avg_time:.2f} seconds")
        logger.info(f"Fastest processing time: {min_time:.2f} seconds")
        logger.info(f"Slowest processing time: {max_time:.2f} seconds")
        
        # Check if any files exceeded the 10-second limit
        slow_files = [name for name, r in results.items() if r['success'] and r['processing_time'] > 10.0]
        if slow_files:
            logger.warning(f"Files that exceeded 10-second limit: {slow_files}")
        else:
            logger.info("✅ All files processed within 10-second limit")
    
    # Generate detailed performance report
    performance_report = profiler.get_performance_report()
    
    logger.info(f"\n{'='*60}")
    logger.info("DETAILED PERFORMANCE ANALYSIS")
    logger.info(f"{'='*60}")
    
    if 'operation_stats' in performance_report:
        logger.info("Operation Performance:")
        for op_name, stats in performance_report['operation_stats'].items():
            logger.info(f"  {op_name}:")
            logger.info(f"    Average time: {stats['avg_duration']:.3f}s")
            logger.info(f"    Max time: {stats['max_duration']:.3f}s")
            logger.info(f"    Count: {stats['count']}")
            logger.info(f"    Peak memory: {stats['max_memory_peak']:.1f}MB")
    
    if 'bottlenecks' in performance_report and performance_report['bottlenecks']:
        logger.info("\nIdentified Bottlenecks:")
        for bottleneck in performance_report['bottlenecks']:
            logger.warning(f"  ⚠️  {bottleneck}")
    
    if 'recommendations' in performance_report and performance_report['recommendations']:
        logger.info("\nOptimization Recommendations:")
        for recommendation in performance_report['recommendations']:
            logger.info(f"  💡 {recommendation}")
    
    # Save detailed report to file
    report_path = output_dir / "performance_report.json"
    profiler.save_report(str(report_path))
    logger.info(f"\nDetailed performance report saved to: {report_path}")
    
    return results


def create_synthetic_large_document():
    """Create a synthetic large PDF document for stress testing."""
    try:
        import fitz  # PyMuPDF
        
        # Create a new PDF document
        doc = fitz.open()
        
        # Add 50 pages with various content
        for page_num in range(50):
            page = doc.new_page()
            
            # Add title on first page
            if page_num == 0:
                page.insert_text((72, 100), "Large Document Performance Test", 
                                fontsize=24, fontname="helv")
            
            # Add chapter headings every 10 pages
            if page_num % 10 == 0:
                page.insert_text((72, 150), f"Chapter {page_num // 10 + 1}: Performance Testing", 
                                fontsize=18, fontname="helv")
            
            # Add section headings every 5 pages
            if page_num % 5 == 0:
                page.insert_text((72, 200), f"Section {page_num // 5 + 1}.1: Test Content", 
                                fontsize=14, fontname="helv")
            
            # Add subsection headings every 2 pages
            if page_num % 2 == 0:
                page.insert_text((72, 250), f"Subsection {page_num // 2 + 1}.1.1: Details", 
                                fontsize=12, fontname="helv")
            
            # Add body text
            body_text = f"""This is page {page_num + 1} of the synthetic test document. 
This document is designed to test the performance of the PDF Structure Extractor 
with a large document containing multiple pages and various heading levels.

The document contains structured content with titles, headings, and body text
to simulate a real-world document that might be processed by the system.

This content is repeated across multiple pages to create a document of sufficient
size to test performance under load conditions."""
            
            page.insert_text((72, 300), body_text, fontsize=10, fontname="helv")
        
        # Save the synthetic document
        output_path = Path("input") / "synthetic_large_document.pdf"
        output_path.parent.mkdir(exist_ok=True)
        doc.save(str(output_path))
        doc.close()
        
        logger.info(f"Created synthetic large document: {output_path}")
        return str(output_path)
        
    except ImportError:
        logger.warning("PyMuPDF not available, cannot create synthetic document")
        return None
    except Exception as e:
        logger.error(f"Error creating synthetic document: {e}")
        return None


def run_stress_test():
    """Run stress test with large document."""
    logger.info("Creating synthetic large document for stress testing...")
    
    synthetic_doc = create_synthetic_large_document()
    if not synthetic_doc:
        logger.error("Could not create synthetic document for stress testing")
        return
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    logger.info("Running stress test with 50-page document...")
    
    # Start monitoring
    profiler = get_global_profiler()
    start_global_monitoring()
    
    start_time = time.time()
    
    try:
        result_path = process_pdf(synthetic_doc, str(output_dir), timeout=10)
        end_time = time.time()
        processing_time = end_time - start_time
        
        if result_path:
            logger.info(f"✅ Stress test completed successfully in {processing_time:.2f} seconds")
            if processing_time <= 10.0:
                logger.info("✅ Processing time within 10-second requirement")
            else:
                logger.warning(f"⚠️  Processing time exceeded 10-second limit by {processing_time - 10.0:.2f} seconds")
        else:
            logger.error("❌ Stress test failed")
            
    except Exception as e:
        end_time = time.time()
        processing_time = end_time - start_time
        logger.error(f"❌ Stress test failed with error: {e}")
    
    finally:
        stop_global_monitoring()
        
        # Generate stress test report
        performance_report = profiler.get_performance_report()
        
        logger.info("\nStress Test Performance Report:")
        if 'operation_stats' in performance_report:
            for op_name, stats in performance_report['operation_stats'].items():
                logger.info(f"  {op_name}: {stats['avg_duration']:.3f}s avg, {stats['max_memory_peak']:.1f}MB peak")


def main():
    """Main function to run performance tests."""
    logger.info("Starting PDF Structure Extractor Performance Tests")
    logger.info(f"Python path: {sys.path[0]}")
    
    # Test with sample PDFs
    logger.info("\n" + "="*60)
    logger.info("TESTING WITH SAMPLE PDF FILES")
    logger.info("="*60)
    
    sample_results = test_performance_with_sample_pdfs()
    
    # Run stress test with large document
    logger.info("\n" + "="*60)
    logger.info("RUNNING STRESS TEST WITH LARGE DOCUMENT")
    logger.info("="*60)
    
    run_stress_test()
    
    logger.info("\n" + "="*60)
    logger.info("PERFORMANCE TESTING COMPLETED")
    logger.info("="*60)


if __name__ == "__main__":
    main()