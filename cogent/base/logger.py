"""
Global logging configuration for Cogent.
Provides a centralized logging setup with file and console output.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

from cogent.base.config import get_config


def setup_cogent_logger(
    name: str = "cogent",
    log_dir: Optional[Path] = None,
    log_level: Optional[str] = None,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """
    Set up the global Cogent logger with file and console handlers.

    Args:
        name: Logger name, defaults to "cogent"
        log_dir: Directory for log files, defaults to config value
        log_level: Logging level, defaults to config value
        log_format: Log format string, defaults to config value

    Returns:
        Configured logger instance
    """
    # Get configuration
    config = get_config()

    # Use provided values or fall back to config defaults
    log_dir = log_dir or config.log_dir
    log_level = log_level or config.log_level
    log_format = log_format or config.log_format

    # Create log directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)

    # Get logger
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # Set log level
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler with rotation
    log_file = log_dir / "cogent.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"  # 10MB
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Error file handler for ERROR and above
    error_log_file = log_dir / "cogent_error.log"
    error_handler = logging.handlers.RotatingFileHandler(
        error_log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"  # 10MB
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)

    logger.info(f"Cogent logger initialized - Level: {log_level}, Log dir: {log_dir}")

    return logger


def get_cogent_logger(name: str = "cogent") -> logging.Logger:
    """
    Get the global Cogent logger instance.

    Args:
        name: Logger name, defaults to "cogent"

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    # If logger doesn't have handlers, set it up
    if not logger.handlers:
        setup_cogent_logger(name)

    return logger


# Initialize the global logger when module is imported
_cogent_logger = setup_cogent_logger()
