"""Validate wheel/sdist ownership after building the three maintained packages."""
from pathlib import Path
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
for project, package in [(ROOT, 'pybspf'), (ROOT / 'packages/models', 'bspf_models'),
                         (ROOT / 'packages/sim', 'bspf_sim')]:
    wheels = list((project / 'dist').glob('*.whl'))
    sdists = list((project / 'dist').glob('*.tar.gz'))
    assert wheels and sdists, f'Build both wheel and sdist: {project}'
    for path in wheels + sdists:
        if path.suffix == '.whl':
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
        else:
            with tarfile.open(path) as archive:
                names = archive.getnames()
        assert any(f'{package}/__init__.py' in name for name in names), path
        for name in names:
            parts = Path(name).parts
            assert not set(parts) & {'legacy', 'experiments', 'examples', 'scratch', 'tests', 'build', '__pycache__', 'bspf_jax'}, (path, name)
            if package == 'pybspf':
                assert not set(parts) & {'solvers', 'bspf_models', 'bspf_sim'}, (path, name)
        if package == 'bspf_models':
            assert any(name.endswith('surface_exchange_LICENSE.txt') for name in names), path
        if package == 'bspf_sim':
            assert any(name.endswith('v1_sources.json') for name in names), path
        print(path.relative_to(ROOT), 'OK')
