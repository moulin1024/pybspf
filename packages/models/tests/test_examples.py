"""Run the JAX example notebooks verbatim, including numerical assertions."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
NOTEBOOKS = (
    "operation/differentiate_1d.ipynb",
    "operation/differentiate_1d_noisy.ipynb",
    "operation/differentiate_noisy_1d_3d.ipynb",
    "operation/integrate_1d.ipynb",
    "operation/differentiate_2d.ipynb",
    "operation/differentiate_3d.ipynb",
    "pde/diffusion_2d.ipynb",
    "pde/burgers_1d.ipynb",
    "pde/schroedinger_1d.ipynb",
    "pde/nlse_1d.ipynb",
    "pde/kdv_1d.ipynb",
    "pde/sine_gordon_1d.ipynb",
    "pde/alfven_1d.ipynb",
    "pde/parallel_kinetic_1d.ipynb",
    "pde/landau_open_1d.ipynb",
    "pde/euler_bernoulli_1d.ipynb",
)


@pytest.mark.skipif(importlib.util.find_spec("matplotlib") is None, reason="install the notebook extra")
@pytest.mark.parametrize("name", NOTEBOOKS)
def test_jax_notebook(name, tmp_path):
    path = ROOT / "examples" / name
    if not path.exists():
        pytest.skip("repository examples are not included in an installed wheel")
    notebook = json.loads(path.read_text())
    assert notebook["metadata"]["bspf_backend"] == "jax"
    source = "\n\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    assert "bspf_jax" not in source
    assert "sys.path" not in source
    env = {**os.environ, "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(tmp_path / "mpl"),
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    completed = subprocess.run([sys.executable, "-c", source], cwd=tmp_path,
                               env=env, capture_output=True, text=True,
                               timeout=300 if name in ("pde/landau_open_1d.ipynb", "pde/parallel_kinetic_1d.ipynb", "pde/alfven_1d.ipynb", "pde/kdv_1d.ipynb", "pde/nlse_1d.ipynb",
                                                       "operation/differentiate_2d.ipynb",
                                                       "operation/differentiate_noisy_1d_3d.ipynb",
                                                       "operation/differentiate_3d.ipynb") else 120)
    assert completed.returncode == 0, completed.stdout + completed.stderr
