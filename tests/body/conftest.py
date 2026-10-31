"""
tests/body/conftest.py

pytest configuration for the body sub-package tests.
Adds the project root and the realsense_measure package to sys.path.
"""
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = TESTS_DIR.parent
PACKAGE_DIR = REPO_ROOT / "realsense_measure"

for p in [str(PACKAGE_DIR), str(REPO_ROOT), str(TESTS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

