"""Logging utilities."""

import sys
from loguru import logger


def setup_logger(log_file: str = None, level: str = "INFO"):
    """Setup logger configuration."""
    logger.remove()

    # Console output
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
    )

    # File output
    if log_file:
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=level,
            rotation="10 MB",
        )

    return logger


def get_logger():
    """Get the configured logger."""
    return logger
