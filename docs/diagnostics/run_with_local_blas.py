"""Run this Python interpreter with the project-local macOS BLAS build.

Examples (from repository root):
    python docs/diagnostics/run_with_local_blas.py -m jupyterlab
    OMP_NUM_THREADS=8 python docs/diagnostics/run_with_local_blas.py script.py

Library selection happens before importing any numerical package. The shared
conda installation is not modified. This launcher is specific to the validated
macOS conda environment; it is not a general BLAS installer.
"""

import os
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[2]
libdir = root / "build/openblas-atomic/install/lib"
library = libdir / "libopenblas.dylib"
if sys.platform != "darwin" or not library.is_file():
    raise SystemExit(f"Build the local macOS OpenBLAS first: {library}")
if len(sys.argv) < 2:
    raise SystemExit("Usage: python run_with_local_blas.py <Python arguments>")
env = dict(os.environ)
env["DYLD_LIBRARY_PATH"] = str(libdir) + (
    ":" + env["DYLD_LIBRARY_PATH"] if env.get("DYLD_LIBRARY_PATH") else ""
)
env["BSPF_BLAS_LIBRARY"] = str(library)
env.setdefault("OMP_NUM_THREADS", "4")
os.execve(sys.executable, [sys.executable, *sys.argv[1:]], env)
