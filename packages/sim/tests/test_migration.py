"""A v1 checkpoint produced before migration resumes with a recorded lineage."""
from dataclasses import asdict
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
from bspf_sim.air_sea.platform import provenance, read_checkpoint, resume, run, source_fingerprint


def test_source_fingerprint_covers_installed_packages():
    sources = source_fingerprint()
    for name in ('pybspf', 'bspf_models', 'bspf_sim'):
        assert any(key.startswith(name + '/') for key in sources)
    assert all(len(value) == 64 for value in sources.values())


def prepare_v1(tmp_path):
    source = Path(__file__).parent / 'reference/v1'
    out = tmp_path / 'old'
    shutil.copytree(source, out / 'checkpoints')
    manifest = out / 'checkpoints/step000000002.json'
    data = json.loads(manifest.read_text())
    # The frozen state/source identity are unchanged. Match runtime metadata so
    # this fixture also exercises the migration on CI's Python/JAX environment.
    current = provenance()
    for key in ('python', 'backend', 'float64'):
        data['provenance'][key] = current[key]
    data['provenance']['dependencies'] = {
        key: value for key, value in current['dependencies'].items()
        if key not in ('pybspf', 'bspf-models', 'bspf-sim')
    }
    manifest.write_text(json.dumps(data))
    (out / 'config.json').write_text(json.dumps(data['config']))
    return out, manifest, data


def test_pre_migration_checkpoint_resume(tmp_path):
    out, manifest, data = prepare_v1(tmp_path)
    config = read_checkpoint(manifest).config
    run(config, tmp_path / 'fresh')
    resume(manifest, until=120.0)
    migrated = read_checkpoint(out / 'checkpoints/step000000004.json')
    fresh = read_checkpoint(tmp_path / 'fresh/checkpoints/step000000004.json')
    for actual, expected in zip(migrated.state, fresh.state):
        np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=1e-12)
    np.testing.assert_allclose(migrated.ledger.cumulative, fresh.ledger.cumulative, rtol=2e-10, atol=1e-8)
    assert migrated.provenance['migrated_from'] == data['provenance']
    assert migrated.provenance['source_files'] == source_fingerprint()


def test_unknown_old_source_is_rejected(tmp_path):
    _, manifest, data = prepare_v1(tmp_path)
    data['provenance']['source_files']['air_sea.py'] = '0' * 64
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='source_files'):
        resume(manifest, until=120.0)
