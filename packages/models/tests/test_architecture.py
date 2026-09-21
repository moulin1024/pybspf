"""Models may depend on numerical packages, never applications or archives."""
import ast
import importlib.util
from pathlib import Path

import bspf_models


def test_dependency_direction():
    for source in Path(bspf_models.__file__).parent.rglob('*.py'):
        module = 'bspf_models.' + '.'.join(source.relative_to(Path(bspf_models.__file__).parent).with_suffix('').parts)
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                name = node.module or ''
                if node.level:
                    name = importlib.util.resolve_name('.' * node.level + name, module.rsplit('.', 1)[0])
                names = [name]
            else:
                continue
            for name in names:
                assert name.split('.')[0] not in {'bspf_sim', 'legacy', 'bspf_jax', 'cupy'}, (source, name)
                if '.elliptic.' in module or '._numerics.' in module:
                    assert not name.startswith(('bspf_models.fluids', 'bspf_models.plasma', 'bspf_models.air_sea')), (source, name)
