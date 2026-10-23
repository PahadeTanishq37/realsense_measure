"""
pytest configuration and path setup for realsense_measure test suite.
"""

import sys
from pathlib import Path

# Add project root, realsense_measure package, tools, and tests directories to sys.path
TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
PACKAGE_DIR = REPO_ROOT / "realsense_measure"
TOOLS_DIR = REPO_ROOT / "tools"

for p in [str(PACKAGE_DIR), str(REPO_ROOT), str(TOOLS_DIR), str(TESTS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)
