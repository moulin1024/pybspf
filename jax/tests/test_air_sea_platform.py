"""Restart/rollback contracts and corruption protection, actual solver states."""

from dataclasses import asdict, replace
import json

import numpy as np
import pytest
from netCDF4 import Dataset

from bspf_jax.air_sea_platform import (
    RunConfig,
    SpatialConfig,
    TimeConfig,
    OutputConfig,
    run,
    resume,
    read_checkpoint,
    report,
)


@pytest.fixture(scope="module")
def config():
    return RunConfig(
        spatial=SpatialConfig(33, 20),
        time=TimeConfig(30.0, 6.0, 30.0),
        output=OutputConfig(120.0, 30.0, 60.0, 60.0),
    )


def test_configuration_contract(config):
    assert RunConfig.from_dict(asdict(config)) == config
    with pytest.raises(ValueError):
        RunConfig.from_dict({"unknown": 1})
    with pytest.raises(ValueError):
        replace(config, output=replace(config.output, fields_seconds=17.0)).validate()
    with pytest.raises(ValueError):
        RunConfig.from_dict({"surface": {"wind_height": 0}})


def test_exact_restart_and_field_metadata(tmp_path, config):
    full, split = tmp_path / "full", tmp_path / "split"
    run(config, full)
    run(config, split, until=60.0)
    resumed = resume(split / "checkpoints/step000000002.json", until=120.0)
    assert resumed["status"]["status"] == "COMPLETE"
    a = read_checkpoint(full / "checkpoints/step000000004.json")
    b = read_checkpoint(split / "checkpoints/step000000004.json")
    for x, y in zip(a.state, b.state):
        np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(a.ledger.cumulative, b.ledger.cumulative)
    assert (full / "diagnostics.jsonl").read_text() == (
        split / "diagnostics.jsonl"
    ).read_text()
    assert a.time_seconds == b.time_seconds == 120.0
    with Dataset(split / "fields/step000000004.nc") as ds:
        assert ds.variables["humidity"].units == "kg kg-1"
        assert ds.variables["total_heat"].units == "W m-2"
        assert ds.time_seconds == 120.0
    before = {p: p.stat().st_mtime_ns for p in split.rglob("*") if p.is_file()}
    report(split)
    assert before == {p: p.stat().st_mtime_ns for p in split.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        run(config, full)
    # Rollback from an older valid checkpoint reproduces complete history.
    resume(split / "checkpoints/step000000002.json", until=120.0)
    assert (full / "diagnostics.jsonl").read_text() == (
        split / "diagnostics.jsonl"
    ).read_text()
    assert (split / "rollback_fields").exists()
    payload = split / "checkpoints/step000000004.npz"
    with payload.open("ab") as f:
        f.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        read_checkpoint(payload.with_suffix(".json"))


def test_environment_mismatch_is_rejected(tmp_path, config):
    out = tmp_path / "env"
    run(config, out, until=60.0)
    checkpoint = out / "checkpoints/step000000002.json"
    value = json.loads(checkpoint.read_text())
    value["provenance"]["backend"] = "different-device"
    checkpoint.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="environment mismatch"):
        resume(checkpoint, until=120.0)


def test_failure_evidence_and_config_tamper(tmp_path, config, monkeypatch):
    import bspf_jax.air_sea_platform as module

    original = module.initial_air_sea_state
    monkeypatch.setattr(
        module,
        "initial_air_sea_state",
        lambda p: original(p)._replace(humidity=-original(p).humidity),
    )
    out = tmp_path / "failure"
    with pytest.raises(Exception, match="stage"):
        run(config, out)
    stage = json.loads((out / "failure/stage.json").read_text())
    assert stage["code"] == 2 and stage["time_seconds"] == 0
    assert stage["stage_event"] == 1
    assert stage["field_catalog"][stage["field_index"]] == "humidity_quadrature"
    assert json.loads((out / "status.json").read_text())["status"] == "FAILED"
    assert (out / "failure/stage.npz").exists()
    monkeypatch.setattr(module, "initial_air_sea_state", original)
    clean = tmp_path / "clean"
    run(config, clean, until=60.0)
    stored = json.loads((clean / "config.json").read_text())
    stored["physics"]["wind_speed"] += 1
    (clean / "config.json").write_text(json.dumps(stored))
    with pytest.raises(ValueError, match="configuration mismatch"):
        resume(clean / "checkpoints/step000000002.json", until=120.0)


def test_sigint_saves_checkpoint_and_restores_handler(tmp_path, config, monkeypatch):
    import os
    import signal
    import bspf_jax.air_sea_platform as module

    original = module._compiled_advance
    handler = signal.getsignal(signal.SIGINT)

    def interrupted(*args):
        advance = original(*args)

        def first_step(*values):
            result = advance(*values)
            os.kill(os.getpid(), signal.SIGINT)
            return result

        return first_step

    monkeypatch.setattr(module, "_compiled_advance", interrupted)
    out = tmp_path / "interrupted"
    result = run(config, out)
    assert result["status"]["status"] == "PAUSED"
    assert result["status"]["step"] == 1
    assert signal.getsignal(signal.SIGINT) == handler
    monkeypatch.setattr(module, "_compiled_advance", original)
    resume(out / "checkpoints/step000000001.json", until=120.0)
    assert report(out)["status"]["status"] == "COMPLETE"


def test_off_cadence_pause_and_same_time_rollback(tmp_path, config):
    config = replace(config, output=replace(config.output, diagnostics_seconds=60.0))
    full, split = tmp_path / "full", tmp_path / "split"
    run(config, full)
    run(config, split, until=30.0)
    assert not (split / "diagnostics.jsonl").exists()
    resume(split / "checkpoints/step000000001.json", until=120.0)
    assert (full / "diagnostics.jsonl").read_bytes() == (
        split / "diagnostics.jsonl"
    ).read_bytes()
    resume(split / "checkpoints/step000000001.json", until=30.0)
    pointer = json.loads((split / "latest_checkpoint.json").read_text())
    assert pointer["manifest"] == "checkpoints/step000000001.json"
    resume(split / pointer["manifest"], until=120.0)
    assert (full / "diagnostics.jsonl").read_bytes() == (
        split / "diagnostics.jsonl"
    ).read_bytes()
