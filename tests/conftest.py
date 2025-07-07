"""
Pytest configuration file for Cogent tests.
This file ensures that the src directory is in the Python path for imports.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))
