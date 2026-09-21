"""Package ownership and installed-core contracts for the JAX migration."""
import ast
from pathlib import Path
import subprocess
import sys

import pybspf


def test_core_has_no_physics_or_archive_dependency():
    root = Path(pybspf.__file__).parent
    for source in root.rglob('*.py'):
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or '']
            else:
                continue
            assert not any(name.split('.')[0] in {'bspf_models', 'bspf_sim', 'legacy', 'bspf_jax', 'cupy'} for name in names), source
    assert not (root / 'solvers').exists()
    assert not hasattr(pybspf, 'BSPF1D')
    assert not hasattr(pybspf, 'PressurePoisson2D')


def test_import_preserves_jax_settings_and_does_not_load_models(tmp_path):
    code = '''import sys, jax
before = (jax.config.x64_enabled, jax.config.jax_platforms)
import pybspf
assert before == (jax.config.x64_enabled, jax.config.jax_platforms)
assert not any(n.split('.')[0] in {'bspf_models', 'bspf_sim', 'legacy', 'bspf_jax', 'cupy'} for n in sys.modules)
assert 'pybspf.trial_spaces' not in sys.modules
assert 'pybspf.fourier_extension' not in sys.modules
'''
    subprocess.run([sys.executable, '-c', code], cwd=tmp_path, check=True)
