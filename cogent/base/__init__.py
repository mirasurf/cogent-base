"""
Cogent package initialization.
Provides basic logging utilities for downstream libraries.
"""

import dotenv

from .logger import get_basic_logger, get_logger, setup_logger_with_handlers

dotenv.load_dotenv()

__all__ = ["get_logger", "get_basic_logger", "setup_logger_with_handlers"]
