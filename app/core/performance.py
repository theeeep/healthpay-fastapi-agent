import asyncio
import time
from contextlib import asynccontextmanager
from functools import wraps
from typing import Any, Callable

from app.core.logger import logger


def timing_decorator(func: Callable) -> Callable:
    """Decorator to measure function execution time."""

    @wraps(func)
    async def async_wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} completed in {execution_time:.2f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.2f}s: {e}")
            raise

    @wraps(func)
    def sync_wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} completed in {execution_time:.2f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.2f}s: {e}")
            raise

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


@asynccontextmanager
async def performance_timer(operation_name: str):
    """Context manager for timing operations."""
    start_time = time.time()
    try:
        yield
        execution_time = time.time() - start_time
        logger.info(f"{operation_name} completed in {execution_time:.2f}s")
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"{operation_name} failed after {execution_time:.2f}s: {e}")
        raise


class PerformanceMetrics:
    """Simple performance metrics collector."""

    def __init__(self):
        self.metrics = {}

    def record_metric(self, name: str, value: float, unit: str = "seconds"):
        """Record a performance metric."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append({"value": value, "unit": unit, "timestamp": time.time()})

    def get_average(self, name: str) -> float:
        """Get average value for a metric."""
        if name not in self.metrics or not self.metrics[name]:
            return 0.0
        values = [m["value"] for m in self.metrics[name]]
        return sum(values) / len(values)

    def get_summary(self) -> dict:
        """Get summary of all metrics."""
        summary = {}
        for name, measurements in self.metrics.items():
            if measurements:
                values = [m["value"] for m in measurements]
                summary[name] = {
                    "count": len(values),
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "unit": measurements[0]["unit"],
                }
        return summary


# Global performance metrics instance
performance_metrics = PerformanceMetrics()
