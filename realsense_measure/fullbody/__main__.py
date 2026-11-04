"""
realsense_measure/fullbody/__main__.py

Package entry point enabling:
    python -m realsense_measure.fullbody <subcommand> [options]
"""

from __future__ import annotations

import sys
from realsense_measure.fullbody.cli import main

if __name__ == "__main__":
    sys.exit(main())

