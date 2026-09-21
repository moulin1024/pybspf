#!/usr/bin/env python3
"""Run the ISW slope example; see README.md."""

from pathlib import Path
import os
import sys

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")
sys.path.insert(0, str(Path(__file__).resolve().parent / "source"))
from execution import main

if __name__ == "__main__":
    main()
