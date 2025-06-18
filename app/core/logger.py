import sys

from loguru import logger as _logger


class LoggerGuRu:
    """Custom logger configuration for the application"""

    @staticmethod
    def setup() -> _logger:
        """Configure and return a Loguru logger instance with custom formatting"""

        # Remove default handler
        _logger.remove()

        # Add console handler with custom formatting
        _logger.add(
            sys.stdout,
            colorize=True,
            format=(
                "<blue>{time:MMMM D, YYYY - HH:mm:ss}</blue> | "
                "<green><level>{level: <8}</level></green> | "
                "<cyan>{name}</cyan>:"
                "<cyan>{function}</cyan>:"
                "<cyan>{line}</cyan> - "
                "<green><level>{message}</level></green>"
            ),
            level="INFO",
        )

        return _logger


# Initialize and export logger
logger = LoggerGuRu.setup()
__all__ = ["logger"]
