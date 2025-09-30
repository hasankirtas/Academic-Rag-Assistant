"""
Performance Monitoring for RAG System

Tracks and logs performance metrics to identify bottlenecks.
"""

import time
import logging
from typing import Dict, Any, Optional
from contextlib import contextmanager
from collections import defaultdict

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class PerformanceMonitor:
    """
    Simple performance monitoring for RAG pipeline components.
    """
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.current_operations = {}
    
    @contextmanager
    def time_operation(self, operation_name: str, **kwargs):
        """
        Context manager to time operations.
        
        Args:
            operation_name: Name of the operation being timed
            **kwargs: Additional metadata to store
        """
        start_time = time.time()
        operation_id = f"{operation_name}_{int(start_time * 1000)}"
        
        try:
            self.current_operations[operation_id] = {
                'name': operation_name,
                'start_time': start_time,
                'metadata': kwargs
            }
            yield operation_id
        finally:
            end_time = time.time()
            duration = (end_time - start_time) * 1000  # Convert to milliseconds
            
            self.metrics[operation_name].append({
                'duration_ms': duration,
                'timestamp': start_time,
                'metadata': kwargs
            })
            
            if operation_id in self.current_operations:
                del self.current_operations[operation_id]
            
            logger.debug(f"{operation_name} completed in {duration:.2f}ms")
    
    def get_stats(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Args:
            operation_name: Specific operation to get stats for, or None for all
            
        Returns:
            Dictionary with performance statistics
        """
        if operation_name:
            if operation_name not in self.metrics:
                return {}
            
            durations = [op['duration_ms'] for op in self.metrics[operation_name]]
            return {
                'operation': operation_name,
                'count': len(durations),
                'avg_duration_ms': sum(durations) / len(durations) if durations else 0,
                'min_duration_ms': min(durations) if durations else 0,
                'max_duration_ms': max(durations) if durations else 0,
                'total_duration_ms': sum(durations)
            }
        else:
            stats = {}
            for op_name in self.metrics:
                stats[op_name] = self.get_stats(op_name)
            return stats
    
    def log_slow_operations(self, threshold_ms: float = 1000):
        """
        Log operations that took longer than threshold.
        
        Args:
            threshold_ms: Threshold in milliseconds
        """
        for operation_name, operations in self.metrics.items():
            slow_ops = [
                op for op in operations 
                if op['duration_ms'] > threshold_ms
            ]
            
            if slow_ops:
                logger.warning(
                    f"Found {len(slow_ops)} slow {operation_name} operations "
                    f"(>{threshold_ms}ms): {[op['duration_ms'] for op in slow_ops]}"
                )
    
    def clear_metrics(self):
        """Clear all performance metrics."""
        self.metrics.clear()
        self.current_operations.clear()
        logger.info("Performance metrics cleared")
    
    def get_current_operations(self) -> Dict[str, Any]:
        """Get currently running operations."""
        current = {}
        for op_id, op_data in self.current_operations.items():
            current[op_id] = {
                'name': op_data['name'],
                'running_time_ms': (time.time() - op_data['start_time']) * 1000,
                'metadata': op_data['metadata']
            }
        return current


# Global performance monitor instance
_performance_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor

def time_operation(operation_name: str, **kwargs):
    """Convenience function for timing operations."""
    return get_performance_monitor().time_operation(operation_name, **kwargs)
