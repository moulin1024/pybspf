"""! @file conftest.py
@brief Test configuration for local package imports.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add both the repository root and the ``src`` layout to ``sys.path`` so tests
# can import the in-repo package without requiring installation first.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

for path in (str(SRC), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)
