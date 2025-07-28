"""
Performance profiling and optimization module for the PDF Structure Extractor.

This module provides tools for profiling application performance, identifying
bottlenecks, and implementing optimizations for faster processing.
"""

import time
import logging
import psutil
import os
import gc
from typing import Dict, List, Any, Optional, Callable
from functools import wraps
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import threading
import json

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    operation_name: str
    duration: float
    memory_before: float
    memory_after: float
    memory_peak: float
    cpu_percent: float
    timestamp: float


class PerformanceProfiler:
    """Main performance profiler for monitoring and optimizing application performance."""
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.memory_monitor = MemoryMonitor()
        self.timing_monitor = TimingMonitor()
        self.bottleneck_analyzer = BottleneckAnalyzer()
        self._monitoring_active = False
        self._monitor_thread = None
    
    def start_monitoring(self):
        """Start continuous performance monitoring."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous performance monitoring."""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Continuous monitoring loop."""
        while self._monitoring_active:
            try:
                # Monitor memory usage
                memory_info = psutil.virtual_memory()
                process = psutil.Process()
                cpu_percent = process.cpu_percent()
                
                # Log if memory usage is high
                if memory_info.percent > 80:
                    logger.warning(f"High memory usage: {memory_info.percent:.1f}%")
                
                # Log if CPU usage is high
                if cpu_percent > 90:
                    logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
                
                time.sleep(0.5)  # Monitor every 500ms
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                break
    
    @contextmanager
    def profile_operation(self, operation_name: str):
        """Context manager for profiling individual operations."""
        # Get initial metrics
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        start_time = time.time()
        
        # Start memory peak monitoring
        peak_monitor = PeakMemoryMonitor()
        peak_monitor.start()
        
        try:
            yield
        finally:
            # Get final metrics
            end_time = time.time()
            duration = end_time - start_time
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_peak = peak_monitor.stop()
            cpu_percent = process.cpu_percent()
            
            # Create metrics record
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                duration=duration,
                memory_before=memory_before,
                memory_after=memory_after,
                memory_peak=memory_peak,
                cpu_percent=cpu_percent,
                timestamp=start_time
            )
            
            self.metrics.append(metrics)
            
            # Log performance info
            logger.info(f"Operation '{operation_name}' completed in {duration:.3f}s, "
                       f"memory: {memory_before:.1f}MB -> {memory_after:.1f}MB "
                       f"(peak: {memory_peak:.1f}MB)")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.metrics:
            return {"error": "No performance data collected"}
        
        # Analyze metrics by operation
        operation_stats = {}
        for metric in self.metrics:
            op_name = metric.operation_name
            if op_name not in operation_stats:
                operation_stats[op_name] = {
                    'count': 0,
                    'total_duration': 0.0,
                    'max_duration': 0.0,
                    'min_duration': float('inf'),
                    'total_memory_used': 0.0,
                    'max_memory_peak': 0.0
                }
            
            stats = operation_stats[op_name]
            stats['count'] += 1
            stats['total_duration'] += metric.duration
            stats['max_duration'] = max(stats['max_duration'], metric.duration)
            stats['min_duration'] = min(stats['min_duration'], metric.duration)
            stats['total_memory_used'] += (metric.memory_after - metric.memory_before)
            stats['max_memory_peak'] = max(stats['max_memory_peak'], metric.memory_peak)
        
        # Calculate averages
        for stats in operation_stats.values():
            stats['avg_duration'] = stats['total_duration'] / stats['count']
            stats['avg_memory_used'] = stats['total_memory_used'] / stats['count']
        
        # Identify bottlenecks
        bottlenecks = self.bottleneck_analyzer.identify_bottlenecks(self.metrics)
        
        return {
            'total_operations': len(self.metrics),
            'total_duration': sum(m.duration for m in self.metrics),
            'operation_stats': operation_stats,
            'bottlenecks': bottlenecks,
            'recommendations': self._generate_recommendations(operation_stats, bottlenecks)
        }
    
    def _generate_recommendations(self, operation_stats: Dict, bottlenecks: List[str]) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Check for slow operations
        for op_name, stats in operation_stats.items():
            if stats['avg_duration'] > 2.0:  # Operations taking more than 2 seconds
                recommendations.append(f"Optimize '{op_name}' operation (avg: {stats['avg_duration']:.2f}s)")
        
        # Check for memory-intensive operations
        for op_name, stats in operation_stats.items():
            if stats['max_memory_peak'] > 500:  # More than 500MB peak
                recommendations.append(f"Reduce memory usage in '{op_name}' (peak: {stats['max_memory_peak']:.1f}MB)")
        
        # Add bottleneck-specific recommendations
        for bottleneck in bottlenecks:
            recommendations.append(f"Address bottleneck: {bottleneck}")
        
        return recommendations
    
    def save_report(self, output_path: str):
        """Save performance report to file."""
        report = self.get_performance_report()
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Performance report saved to {output_path}")


class MemoryMonitor:
    """Monitor memory usage and implement memory optimizations."""
    
    def __init__(self):
        self.process = psutil.Process()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        memory_info = self.process.memory_info()
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': self.process.memory_percent()
        }
    
    def optimize_memory(self):
        """Perform memory optimization operations."""
        # Force garbage collection
        collected = gc.collect()
        logger.debug(f"Garbage collection freed {collected} objects")
        
        # Clear any caches that might exist
        self._clear_caches()
    
    def _clear_caches(self):
        """Clear internal caches to free memory."""
        # This would clear any module-level caches
        # Implementation depends on specific caching mechanisms used
        pass
    
    @contextmanager
    def memory_limit(self, limit_mb: int):
        """Context manager to enforce memory limits."""
        initial_memory = self.get_memory_usage()['rss_mb']
        
        try:
            yield
        finally:
            current_memory = self.get_memory_usage()['rss_mb']
            if current_memory > limit_mb:
                logger.warning(f"Memory limit exceeded: {current_memory:.1f}MB > {limit_mb}MB")
                self.optimize_memory()


class TimingMonitor:
    """Monitor timing and implement timing optimizations."""
    
    def __init__(self):
        self.operation_times = {}
    
    def time_function(self, func: Callable) -> Callable:
        """Decorator to time function execution."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                func_name = f"{func.__module__}.{func.__name__}"
                
                if func_name not in self.operation_times:
                    self.operation_times[func_name] = []
                self.operation_times[func_name].append(duration)
                
                logger.debug(f"Function {func_name} took {duration:.3f}s")
        
        return wrapper
    
    def get_timing_report(self) -> Dict[str, Dict[str, float]]:
        """Get timing report for all monitored functions."""
        report = {}
        for func_name, times in self.operation_times.items():
            report[func_name] = {
                'count': len(times),
                'total': sum(times),
                'average': sum(times) / len(times),
                'min': min(times),
                'max': max(times)
            }
        return report


class BottleneckAnalyzer:
    """Analyze performance metrics to identify bottlenecks."""
    
    def identify_bottlenecks(self, metrics: List[PerformanceMetrics]) -> List[str]:
        """Identify performance bottlenecks from metrics."""
        bottlenecks = []
        
        if not metrics:
            return bottlenecks
        
        # Find operations that take more than 30% of total time
        total_time = sum(m.duration for m in metrics)
        operation_times = {}
        
        for metric in metrics:
            op_name = metric.operation_name
            if op_name not in operation_times:
                operation_times[op_name] = 0
            operation_times[op_name] += metric.duration
        
        for op_name, op_time in operation_times.items():
            if op_time / total_time > 0.3:
                bottlenecks.append(f"Operation '{op_name}' consumes {op_time/total_time*100:.1f}% of total time")
        
        # Find memory-intensive operations
        for metric in metrics:
            memory_increase = metric.memory_after - metric.memory_before
            if memory_increase > 200:  # More than 200MB increase
                bottlenecks.append(f"Operation '{metric.operation_name}' uses {memory_increase:.1f}MB memory")
        
        return bottlenecks


class PeakMemoryMonitor:
    """Monitor peak memory usage during an operation."""
    
    def __init__(self):
        self.peak_memory = 0.0
        self.monitoring = False
        self.monitor_thread = None
        self.process = psutil.Process()
    
    def start(self):
        """Start monitoring peak memory usage."""
        self.monitoring = True
        self.peak_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop(self) -> float:
        """Stop monitoring and return peak memory usage."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=0.1)
        return self.peak_memory
    
    def _monitor_loop(self):
        """Monitor memory usage in a loop."""
        while self.monitoring:
            try:
                current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
                self.peak_memory = max(self.peak_memory, current_memory)
                time.sleep(0.01)  # Check every 10ms
            except Exception:
                break


# Decorator for easy profiling
def profile_performance(operation_name: str = None):
    """Decorator to profile function performance."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            # Use global profiler if available
            profiler = getattr(wrapper, '_profiler', None)
            if not profiler:
                profiler = PerformanceProfiler()
                wrapper._profiler = profiler
            
            with profiler.profile_operation(op_name):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Global profiler instance
_global_profiler = None

def get_global_profiler() -> PerformanceProfiler:
    """Get or create global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler


def start_global_monitoring():
    """Start global performance monitoring."""
    profiler = get_global_profiler()
    profiler.start_monitoring()


def stop_global_monitoring():
    """Stop global performance monitoring."""
    profiler = get_global_profiler()
    profiler.stop_monitoring()


def get_global_performance_report() -> Dict[str, Any]:
    """Get global performance report."""
    profiler = get_global_profiler()
    return profiler.get_performance_report()


def save_global_performance_report(output_path: str):
    """Save global performance report to file."""
    profiler = get_global_profiler()
    profiler.save_report(output_path)