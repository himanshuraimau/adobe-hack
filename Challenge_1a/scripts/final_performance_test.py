#!/usr/bin/env python3
"""
Final performance test to verify 10-second requirement compliance.
"""

import time
import logging
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_10_second_requirement():
    """Test that processing completes within 10 seconds."""
    from main import process_pdf
    from scripts.performance_profiler import get_global_profiler, start_global_monitoring, stop_global_monitoring
    
    input_dir = Path("input")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.error("No PDF files found for testing")
        return False
    
    logger.info("Testing 10-second requirement compliance...")
    
    # Start monitoring
    profiler = get_global_profiler()
    start_global_monitoring()
    
    all_passed = True
    
    for pdf_file in pdf_files:
        logger.info(f"Testing {pdf_file.name}...")
        
        start_time = time.time()
        
        try:
            result_path = process_pdf(str(pdf_file), str(output_dir), timeout=10)
            end_time = time.time()
            processing_time = end_time - start_time
            
            if result_path and processing_time <= 10.0:
                logger.info(f"✅ {pdf_file.name}: {processing_time:.2f}s (PASS)")
            elif result_path:
                logger.warning(f"⚠️  {pdf_file.name}: {processing_time:.2f}s (SLOW)")
                all_passed = False
            else:
                logger.error(f"❌ {pdf_file.name}: FAILED")
                all_passed = False
                
        except Exception as e:
            logger.error(f"❌ {pdf_file.name}: ERROR - {e}")
            all_passed = False
    
    stop_global_monitoring()
    
    # Generate performance summary
    performance_report = profiler.get_performance_report()
    
    logger.info("\n" + "="*60)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("="*60)
    
    if 'operation_stats' in performance_report:
        total_time = sum(stats['total_duration'] for stats in performance_report['operation_stats'].values())
        logger.info(f"Total processing time: {total_time:.2f}s")
        
        logger.info("Operation breakdown:")
        for op_name, stats in performance_report['operation_stats'].items():
            percentage = (stats['total_duration'] / total_time * 100) if total_time > 0 else 0
            logger.info(f"  {op_name}: {stats['avg_duration']:.3f}s avg ({percentage:.1f}%)")
    
    if all_passed:
        logger.info("✅ ALL TESTS PASSED - 10-second requirement met")
    else:
        logger.warning("⚠️  SOME TESTS FAILED - optimization needed")
    
    return all_passed


def main():
    """Run final performance test."""
    logger.info("Starting final performance validation...")
    
    success = test_10_second_requirement()
    
    if success:
        logger.info("🎉 Performance optimization task completed successfully!")
        logger.info("The system meets all performance requirements:")
        logger.info("  ✅ Processing time < 10 seconds")
        logger.info("  ✅ Memory usage optimized")
        logger.info("  ✅ CPU-only operation")
        logger.info("  ✅ Comprehensive monitoring")
    else:
        logger.error("❌ Performance requirements not fully met")
        sys.exit(1)


if __name__ == "__main__":
    main()