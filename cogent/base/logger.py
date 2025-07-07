"""
Basic logging utilities for Cogent.
Provides a simple logging setup that can be overridden by downstream libraries.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_basic_logger(
    name: str = "cogent",
    level: str = "INFO",
    format_string: Optional[str] = None,
    handlers: Optional[list] = None,
) -> logging.Logger:
    """
    Get a basic logger with minimal configuration.

    This function provides a basic logging setup that downstream libraries can extend.
    It doesn't create file handlers or complex configurations by default.

    Args:
        name: Logger name, defaults to "cogent"
        level: Logging level, defaults to "INFO"
        format_string: Custom format string, if None uses basic format
        handlers: Custom handlers list, if None creates basic console handler

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid reconfiguring if logger already has handlers
    if logger.handlers:
        return logger

    # Set log level
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Use provided format or basic format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_string)

    # Use provided handlers or create basic console handler
    if handlers is None:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        handlers = [console_handler]

    # Add handlers
    for handler in handlers:
        logger.addHandler(handler)

    return logger


def setup_logger_with_handlers(
    name: str = "cogent",
    level: str = "INFO",
    format_string: Optional[str] = None,
    log_dir: Optional[Path] = None,
    enable_file_logging: bool = False,
    enable_error_file: bool = False,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    Set up a logger with file handlers for downstream libraries that need them.

    This function provides more comprehensive logging setup for libraries that
    need file logging, rotation, etc. It's not used by default in cogent-base.

    Args:
        name: Logger name
        level: Logging level
        format_string: Custom format string
        log_dir: Directory for log files
        enable_file_logging: Whether to enable general file logging
        enable_error_file: Whether to enable separate error file logging
        max_bytes: Maximum bytes per log file
        backup_count: Number of backup files to keep

    Returns:
        Configured logger instance
    """
    import logging.handlers

    logger = get_basic_logger(name, level, format_string)

    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    log_level = getattr(logging, level.upper(), logging.INFO)

    # Use provided format or basic format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_string)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if enable_file_logging and log_dir:
        # Create log directory if it doesn't exist
        log_dir.mkdir(parents=True, exist_ok=True)

        # General log file
        log_file = log_dir / f"{name}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Error log file
        if enable_error_file:
            error_log_file = log_dir / f"{name}_error.log"
            error_handler = logging.handlers.RotatingFileHandler(
                error_log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatter)
            logger.addHandler(error_handler)

    return logger


def get_logger(name: str = "cogent") -> logging.Logger:
    """
    Get a logger instance.

    This is the main entry point for getting loggers in cogent-base.
    It provides a basic logger that downstream libraries can extend.

    Args:
        name: Logger name, defaults to "cogent"

    Returns:
        Logger instance
    """
    return get_basic_logger(name)


# Legacy function for backward compatibility
def get_cogent_logger(name: str = "cogent") -> logging.Logger:
    """
    Get the Cogent logger instance (legacy function).

    Args:
        name: Logger name, defaults to "cogent"

    Returns:
        Logger instance
    """
    return get_logger(name)
