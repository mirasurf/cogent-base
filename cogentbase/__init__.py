"""
Cogent package initialization.
Provides basic logging utilities for downstream libraries.
"""

from .logger import get_basic_logger, get_logger, setup_logger_with_handlers

__version__ = "0.1.5"
__all__ = ["get_logger", "get_basic_logger", "setup_logger_with_handlers"]
