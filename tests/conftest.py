"""
pytest configuration and path setup for realsense_measure test suite.
"""

from unittest.mock import MagicMock
import sys
from pathlib import Path

# Add project root, tools, and tests directories to sys.path
TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
TOOLS_DIR = REPO_ROOT / "tools"

for p in [str(REPO_ROOT), str(TOOLS_DIR), str(TESTS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# If pyrealsense2 cannot be loaded (e.g. CI or system policy block on hardware DLL),
# install mock so offline camera unit tests can run.
try:
    import pyrealsense2 as _rs
    # Check if DLL loaded successfully
    _ = _rs.pipeline
except (ImportError, AttributeError):
    class MockStream:
        color = 1
        depth = 2
        any = 0

    class MockFormat:
        bgr8 = 1
        z16 = 2
        rgb8 = 3
        any = 0

    class MockRs(MagicMock):
        stream = MockStream
        format = MockFormat
        def align(self, stream_type):
            m = MagicMock()
            m.process = lambda frames: frames
            return m

    mock_rs = MockRs()
    sys.modules["pyrealsense2"] = mock_rs
    import realsense_measure.camera.realsense_capture as rsc
    rsc.rs = mock_rs
