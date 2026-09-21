"""Versioned, restartable experiments for the audited horizontal air-sea model.

CLI: python -m bspf_sim.air_sea.platform {run,resume,report,validate} ...
Checkpoint NPZ is immutable; its JSON manifest is the atomic commit marker.
Recovery truncates only this experiment's outputs newer than its checkpoint.
"""

from dataclasses import asdict, dataclass, field, fields
import argparse
import hashlib
from functools import lru_cache
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from bspf_models.air_sea.air_sea import AirSeaConfig
from bspf_models.air_sea.air_sea import AirSeaState
from bspf_models.air_sea.air_sea import plan_air_sea
from bspf_models.air_sea.air_sea import plan_air_sea_stepper
from bspf_models.air_sea.air_sea import initial_air_sea_state
from bspf_models.air_sea.air_sea import scalar_mean
from bspf_models.air_sea.air_sea import scalar_values
from bspf_models.air_sea.air_sea import air_sea_fields
from bspf_models.air_sea.air_sea import bulk_flux
from bspf_models.air_sea.air_sea import saturation_specific_humidity
from bspf_models.air_sea.air_sea_audit import PROCESSES
from bspf_models.air_sea.air_sea_audit import QUANTITIES
from bspf_models.air_sea.air_sea_audit import audited_step
from bspf_models.air_sea.air_sea_audit import zero_ledger
from bspf_models.air_sea.air_sea_audit import kinetic_energy
from bspf_models.air_sea.air_sea_audit import scalar_variances
from bspf_models.air_sea.air_sea_audit import spectral_tails
from bspf_models.air_sea.surface_exchange import SurfaceExchangeConfig
from bspf_models.air_sea.surface_exchange import REFERENCE_COMMIT
from bspf_models.air_sea.surface_exchange import coare35


@dataclass(frozen=True)
class SpatialConfig:
    n: int = 81
    quadrature_order: int = 32


@dataclass(frozen=True)
class TimeConfig:
    dt_ocean: float = 300.0
    dt_air: float = 60.0
    window: float = 600.0
    method: str = "mri-gark4"


@dataclass(frozen=True)
class OutputConfig:
    duration_seconds: float = 86400.0
    diagnostics_seconds: float = 3600.0
    fields_seconds: float = 1800.0
    checkpoint_seconds: float = 86400.0


@dataclass(frozen=True)
class ValidationConfig:
    heat_atol: float = 1e-3
    water_atol: float = 1e-9
    linear_rtol: float = 1e-10
    kinetic_atol: float = 1e-6
    kinetic_rtol: float = 1e-5
    spectral_tail_limit: float = 0.02
    space_rms_limit: float = 0.01
    space_peak_limit: float = 0.05
    error_separation_ratio: float = 0.1
    smooth_order_min: float = 3.6
    smooth_order_max: float = 4.4


@dataclass(frozen=True)
class RunConfig:
    schema_version: int = 1
    physics: AirSeaConfig = field(default_factory=AirSeaConfig)
    surface: SurfaceExchangeConfig = field(default_factory=SurfaceExchangeConfig)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    @classmethod
    def from_dict(cls, data):
        allowed = {f.name for f in fields(cls)}
        if set(data) - allowed:
            raise ValueError(f"Unknown configuration keys: {set(data) - allowed}")
        if data.get("schema_version", 1) != 1:
            raise ValueError("Unsupported configuration schema")
        kwargs = {"schema_version": 1}
        for name, kind in (
            ("physics", AirSeaConfig),
            ("surface", SurfaceExchangeConfig),
            ("spatial", SpatialConfig),
            ("time", TimeConfig),
            ("output", OutputConfig),
            ("validation", ValidationConfig),
        ):
            kwargs[name] = kind(**data.get(name, {}))
        config = cls(**kwargs)
        config.validate()
        return config

    def validate(self):
        if self.schema_version != 1:
            raise ValueError("Unsupported configuration schema")
        if self.time.method != "mri-gark4":
            raise ValueError(
                "Research runner requires mri-gark4; legacy comparison uses air_sea_step"
            )
        if (
            not isinstance(self.spatial.n, int)
            or isinstance(self.spatial.n, bool)
            or self.spatial.n < 33
        ):
            raise ValueError("n must be an integer >=33")
        if (
            not isinstance(self.spatial.quadrature_order, int)
            or isinstance(self.spatial.quadrature_order, bool)
            or self.spatial.quadrature_order < 20
        ):
            raise ValueError("quadrature_order must be an integer >=20")
        signed = {"f0", "beta", "radiation", "initial_air_temperature_offset"}
        positive = {
            "length",
            "ocean_depth",
            "mixed_layer_depth",
            "atmosphere_depth",
            "rho_ocean",
            "rho_air",
            "cp_ocean",
            "cp_air",
            "latent_heat",
            "reference_temperature",
            "pressure",
        }
        for name, value in asdict(self.physics).items():
            if (
                not np.isfinite(value)
                or (name in positive and value <= 0)
                or (name not in signed | positive and value < 0)
            ):
                raise ValueError(f"Invalid physics parameter: {name}")
        if (
            not isinstance(self.physics.wind_jet_count, int)
            or isinstance(self.physics.wind_jet_count, bool)
            or self.physics.wind_jet_count < 1
        ):
            raise ValueError("wind_jet_count must be a positive integer")
        if self.physics.mixed_layer_depth > self.physics.ocean_depth:
            raise ValueError("Mixed layer exceeds ocean depth")
        stepper = plan_air_sea_stepper(**asdict(self.time))
        for name, value in asdict(self.output).items():
            if (
                not np.isfinite(value)
                or value <= 0
                or not aligned(value, stepper.window)
            ):
                raise ValueError(f"{name} must be a positive exact multiple of window")
        for name, value in asdict(self.validation).items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid validation tolerance: {name}")
        if self.validation.smooth_order_min >= self.validation.smooth_order_max:
            raise ValueError("Invalid smooth-order interval")
        if self.validation.spectral_tail_limit >= 1:
            raise ValueError("Spectral tail limit must be below one")


@dataclass
class BudgetLedger:
    cumulative: np.ndarray
    initial: dict


@dataclass
class RunCheckpoint:
    state: AirSeaState
    ledger: BudgetLedger
    step: int
    time_seconds: float
    config: RunConfig
    provenance: dict


def aligned(value, step):
    return np.isfinite(value) and abs(value / step - round(value / step)) < 1e-10


def _json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def source_fingerprint():
    import importlib
    fingerprints = {}
    for name in ("pybspf", "bspf_models", "bspf_sim"):
        root = Path(importlib.import_module(name).__file__).parent
        for source in sorted(root.rglob("*.py")):
            key = name + "/" + source.relative_to(root).as_posix()
            fingerprints[key] = hashlib.sha256(source.read_bytes()).hexdigest()
    return fingerprints


def provenance():
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL)
        except (OSError, subprocess.CalledProcessError):
            return b"unavailable"

    return dict(
        source_files=source_fingerprint(),
        git_commit=git("rev-parse", "HEAD").decode().strip(),
        git_diff_sha256=hashlib.sha256(git("diff", "--binary", "HEAD")).hexdigest(),
        git_status=git("status", "--porcelain").decode(),
        python=platform.python_version(),
        dependencies={
            name: importlib.metadata.version(name)
            for name in ("pybspf", "bspf-models", "bspf-sim", "jax", "jaxlib", "numpy", "scipy", "gmpy2", "netCDF4")
        },
        backend=jax.default_backend(),
        float64=bool(jax.config.x64_enabled),
        coare_commit=REFERENCE_COMMIT,
    )


@lru_cache(maxsize=12)
def _cached_plan(physics, surface, spatial):
    return plan_air_sea(**asdict(spatial), config=physics, surface=surface)


def _plan(config):
    return _cached_plan(config.physics, config.surface, config.spatial)


@lru_cache(maxsize=2)
def _compiled_advance(physics, surface, spatial, time_config):
    p = _cached_plan(physics, surface, spatial)
    stepper = plan_air_sea_stepper(**asdict(time_config))
    return jax.jit(lambda s, b, t: audited_step(p, stepper, s, b, t))


def _initial(p, s):
    c = p.config
    return dict(
        heat=[
            float(c.ocean_capacity * scalar_mean(p, s.sst)),
            float(c.air_capacity * scalar_mean(p, s.air_temperature, air=True)),
            float(c.latent_heat * c.air_mass * scalar_mean(p, s.humidity, air=True)),
        ],
        water=float(c.air_mass * scalar_mean(p, s.humidity, air=True)),
        kinetic=np.asarray(kinetic_energy(p, s)).tolist(),
        variance=np.asarray(scalar_variances(p, s)).tolist(),
    )


def write_checkpoint(out, checkpoint):
    folder = Path(out) / "checkpoints"
    folder.mkdir(exist_ok=True)
    stem = f"step{checkpoint.step:09d}"
    target = folder / (stem + ".npz")
    if target.exists() and target.with_suffix(".json").exists():
        existing = read_checkpoint(target.with_suffix(".json"))
        same = (
            existing.config == checkpoint.config
            and existing.time_seconds == checkpoint.time_seconds
            and existing.step == checkpoint.step
            and existing.ledger.initial == checkpoint.ledger.initial
        )
        same = (
            same
            and np.array_equal(existing.ledger.cumulative, checkpoint.ledger.cumulative)
            and all(
                np.array_equal(a, b) for a, b in zip(existing.state, checkpoint.state)
            )
        )
        if not same:
            raise ValueError(
                "Existing immutable checkpoint differs from recomputed state"
            )
        _json(
            Path(out) / "latest_checkpoint.json",
            {"manifest": str(target.with_suffix(".json").relative_to(out))},
        )
        return target.with_suffix(".json")
    tmp = target.with_suffix(".npz.tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(
            f,
            **{k: np.asarray(v) for k, v in checkpoint.state._asdict().items()},
            ledger=checkpoint.ledger.cumulative,
        )
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, target)
    manifest = dict(
        schema_version=1,
        payload=target.name,
        sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        step=checkpoint.step,
        time_seconds=checkpoint.time_seconds,
        config=asdict(checkpoint.config),
        initial=checkpoint.ledger.initial,
        provenance=checkpoint.provenance,
        processes=PROCESSES,
        quantities=QUANTITIES,
    )
    _json(target.with_suffix(".json"), manifest)
    _json(
        Path(out) / "latest_checkpoint.json",
        {"manifest": str(target.with_suffix(".json").relative_to(out))},
    )
    # Keep all checkpoints in v1 (therefore at least two), useful for audit.
    return target.with_suffix(".json")


def read_checkpoint(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1 or "ledger" in manifest:
        raise ValueError("Not a supported complete checkpoint")
    payload = path.parent / manifest["payload"]
    if payload.parent != path.parent or payload.suffix != ".npz":
        raise ValueError("Invalid checkpoint payload path")
    if hashlib.sha256(payload.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("Checkpoint checksum mismatch")
    if (
        tuple(manifest["processes"]) != PROCESSES
        or tuple(manifest["quantities"]) != QUANTITIES
    ):
        raise ValueError("Checkpoint budget schema mismatch")
    with np.load(payload, allow_pickle=False) as data:
        state = AirSeaState(*(jnp.asarray(data[k]) for k in AirSeaState._fields))
        ledger = BudgetLedger(data["ledger"].copy(), manifest["initial"])
    if ledger.cumulative.shape != (len(PROCESSES), len(QUANTITIES)) or not np.all(
        np.isfinite(ledger.cumulative)
    ):
        raise ValueError("Invalid checkpoint ledger")
    if not all(np.all(np.isfinite(v)) for v in state):
        raise ValueError("Nonfinite checkpoint state")
    return RunCheckpoint(
        state,
        ledger,
        manifest["step"],
        manifest["time_seconds"],
        RunConfig.from_dict(manifest["config"]),
        manifest["provenance"],
    )


def diagnose(p, s, cumulative, initial, seconds):
    c = p.config
    vals = _initial(p, s)
    totals = np.sum(np.asarray(cumulative), axis=0)
    field = air_sea_fields(p, s, nodes=False)
    qn = np.asarray(scalar_values(p, s.humidity, air=True, nodes=True))
    q, ta = np.asarray(field["humidity"]), np.asarray(field["air_temperature"])
    rh = q / np.asarray(saturation_specific_humidity(ta, c.pressure))
    tan = (
        scalar_values(p, s.air_temperature, air=True, nodes=True)
        + c.reference_temperature
    )
    rhn = qn / np.asarray(saturation_specific_humidity(tan, c.pressure))
    external = np.asarray(cumulative)[
        [
            PROCESSES.index(k)
            for k in ("radiation", "temperature_restore", "humidity_restore")
        ]
    ]
    external_heat = float(np.sum(external[:, :3]))
    heat_residual = sum(vals["heat"]) - sum(initial["heat"]) - external_heat
    water_residual = (
        vals["water"] - initial["water"] + totals[4] - float(np.sum(external[:, 3]))
    )
    kinetic_residual = np.asarray(vals["kinetic"]) - initial["kinetic"] - totals[13:15]
    row = dict(
        time_seconds=float(seconds),
        heat_residual=float(heat_residual),
        internal_heat_leak=float(sum(totals[:3]) - external_heat),
        water_residual=float(water_residual),
        kinetic_residual=kinetic_residual.tolist(),
        variance_residual=(
            np.asarray(vals["variance"]) - initial["variance"] - totals[15:18]
        ).tolist(),
        heat=vals["heat"],
        kinetic=vals["kinetic"],
        air_water=vals["water"],
        diagnostic_ocean_water=float(totals[4]),
        min_q=float(min(q.min(), qn.min())),
        max_q=float(max(q.max(), qn.max())),
        max_RH=float(max(rh.max(), rhn.max())),
        relative_humidity_ranges={
            "quadrature": [float(rh.min()), float(rh.max())],
            "nodes": [float(rhn.min()), float(rhn.max())],
        },
        temperature_ranges={
            name: [float(np.min(value)), float(np.max(value))]
            for name, value in (
                ("air_quadrature", ta),
                ("sst_quadrature", field["sst"]),
                (
                    "air_nodes",
                    scalar_values(p, s.air_temperature, air=True, nodes=True)
                    + c.reference_temperature,
                ),
                (
                    "sst_nodes",
                    scalar_values(p, s.sst, nodes=True) + c.reference_temperature,
                ),
            )
        },
        supersaturated_fraction=float(np.sum(np.asarray(p.weight) * (rh > 1))),
        max_ocean_speed=float(np.max(np.linalg.norm(field["ocean_velocity"], axis=-1))),
        max_air_speed=float(np.max(np.linalg.norm(field["air_velocity"], axis=-1))),
        spectral_tails=np.asarray(spectral_tails(p, s)).tolist(),
        cumulative_by_process=np.asarray(cumulative).tolist(),
    )
    for name, coeff, air in (
        ("sst", s.sst, False),
        ("temperature", s.air_temperature, True),
        ("humidity", s.humidity, True),
    ):
        x, y = (p.air_flow.x, p.air_scalar_y) if air else (p.scalar, p.scalar)
        gradient = (
            np.stack(
                (
                    np.asarray(x.g) @ coeff @ np.asarray(y.b).T,
                    np.asarray(x.b) @ coeff @ np.asarray(y.g).T,
                ),
                axis=-1,
            )
            / c.length
        )
        row[name + "_gradient_rms"] = float(
            np.sqrt(np.sum(np.asarray(p.weight) * np.sum(gradient**2, axis=-1)))
        )
        row[name + "_gradient_max"] = float(np.max(np.linalg.norm(gradient, axis=-1)))
    flux = bulk_flux(
        c,
        field["ocean_velocity"],
        field["air_velocity"],
        field["sst"],
        ta,
        q,
        surface=p.surface,
    )
    for name, value in zip(("stress", "sensible", "evaporation"), flux):
        value = np.asarray(value)
        if value.ndim == 3:
            value = np.linalg.norm(value, axis=-1)
        row[name + "_mean"] = float(np.sum(np.asarray(p.weight) * value))
        row[name + "_maxabs"] = float(np.max(abs(value)))
    if p.surface.method == "coare35":
        f = coare35(
            field["air_velocity"] - field["ocean_velocity"],
            ta,
            q,
            field["sst"],
            pressure=c.pressure,
            boundary_layer_height=c.atmosphere_depth,
            surface=p.surface,
            rho_air=c.rho_air,
            cp_air=c.cp_air,
        )
        row["surface_diagnostics"] = {
            name: dict(
                min=float(np.min(v)),
                max=float(np.max(v)),
                mean=float(np.sum(np.asarray(p.weight) * v)),
            )
            for name, v in f._asdict().items()
            if name not in ("stress", "sensible", "water")
        }
    return row


def _check_budget(config, row):
    v = config.validation
    totals = np.sum(row["cumulative_by_process"], axis=0)
    if abs(row["heat_residual"]) > v.heat_atol + v.linear_rtol * totals[18]:
        raise RuntimeError("Thermodynamic budget failure")
    if abs(row["water_residual"]) > v.water_atol + v.linear_rtol * totals[19]:
        raise RuntimeError("Water budget failure")
    if (
        np.max(abs(np.asarray(row["kinetic_residual"])))
        > v.kinetic_atol + v.kinetic_rtol * totals[20]
    ):
        raise RuntimeError("Kinetic budget failure")
    if max(row["spectral_tails"]) > v.spectral_tail_limit:
        raise RuntimeError("Unresolved spectral tail")


def write_fields(out, p, s, step, seconds):
    from netCDF4 import Dataset

    folder = Path(out) / "fields"
    folder.mkdir(exist_ok=True)
    target = folder / f"step{step:09d}.nc"
    if target.exists():
        raise FileExistsError(target)
    tmp = target.with_suffix(".nc.tmp")
    values = {k: np.asarray(v) for k, v in air_sea_fields(p, s).items()}
    tau, h, e = bulk_flux(
        p.config,
        values["ocean_velocity"],
        values["air_velocity"],
        values["sst"],
        values["air_temperature"],
        values["humidity"],
        surface=p.surface,
    )
    values.update(
        stress=np.asarray(tau),
        sensible=np.asarray(h),
        evaporation=np.asarray(e),
        total_heat=np.asarray(h + p.config.latent_heat * e),
    )
    units = dict(
        ocean_velocity="m s-1",
        air_velocity="m s-1",
        ocean_streamfunction="m2 s-1",
        air_streamfunction="m2 s-1",
        sst="K",
        air_temperature="K",
        humidity="kg kg-1",
        stress="Pa",
        sensible="W m-2",
        evaporation="kg m-2 s-1",
        total_heat="W m-2",
    )
    if p.surface.method == "coare35":
        exchange = coare35(
            values["air_velocity"] - values["ocean_velocity"],
            values["air_temperature"],
            values["humidity"],
            values["sst"],
            pressure=p.config.pressure,
            boundary_layer_height=p.config.atmosphere_depth,
            surface=p.surface,
            rho_air=p.config.rho_air,
            cp_air=p.config.cp_air,
        )
        for name in (
            "drag",
            "heat",
            "moisture",
            "stability",
            "friction_velocity",
            "gustiness",
            "valid",
            "thin_stable_branch",
            "iteration_change",
        ):
            values["exchange_" + name] = np.asarray(getattr(exchange, name))
            units["exchange_" + name] = (
                "m s-1" if name in ("friction_velocity", "gustiness") else "1"
            )
    with Dataset(tmp, "w") as ds:
        ds.createDimension("x", len(p.scalar.x))
        ds.createDimension("y", len(p.scalar.x))
        ds.createDimension("component", 2)
        ds.time_seconds = seconds
        ds.schema_version = 1
        ds.scope = "horizontal vertically averaged; no condensation or resolved vertical buoyancy"
        ds.surface_proxy = p.surface.state_proxy
        ds.surface_exchange_config = json.dumps(asdict(p.surface))
        ds.model_latent_heat_J_kg = p.config.latent_heat
        ds.flux_sign = "stress toward ocean; heat and water upward toward atmosphere"
        for axis in ("x", "y"):
            v = ds.createVariable(axis, "f8", (axis,))
            v[:] = np.asarray(p.scalar.x) * p.config.length
            v.units = "m"
        for name, value in values.items():
            dims = ("x", "y", "component") if value.ndim == 3 else ("x", "y")
            v = ds.createVariable(name, "f8", dims, zlib=True)
            v[:] = value
            v.units = units[name]
            v.long_name = name.replace("_", " ")
    os.replace(tmp, target)


def _integrate(config, out, checkpoint, until):
    p = _plan(config)
    stepper = plan_air_sea_stepper(**asdict(config.time))
    target = config.output.duration_seconds if until is None else float(until)
    if target < checkpoint.time_seconds or not aligned(target, stepper.window):
        raise ValueError("until must be an aligned absolute time at/after checkpoint")
    advance = _compiled_advance(
        config.physics, config.surface, config.spatial, config.time
    )
    s, ledger = checkpoint.state, jnp.asarray(checkpoint.ledger.cumulative)
    initial, prov = checkpoint.ledger.initial, checkpoint.provenance
    start = time.perf_counter()
    stop = [False]
    previous_handler = signal.getsignal(signal.SIGINT)
    previous_term_handler = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    step, seconds = checkpoint.step, checkpoint.time_seconds
    status = "RUNNING"
    _json(out / "status.json", dict(status=status, step=step, time_seconds=seconds))
    try:
        if step == 0 and not (out / "fields/step000000000.nc").exists():
            write_fields(out, p, s, 0, 0.0)
        for step in range(checkpoint.step + 1, round(target / stepper.window) + 1):
            before, before_ledger = s, ledger
            s, ledger, observation = advance(s, ledger, float(seconds))
            if int(observation.code):
                folder = out / "failure"
                folder.mkdir(exist_ok=True)
                np.savez_compressed(
                    folder / "stage.npz",
                    **{
                        k: np.asarray(v) for k, v in observation.state._asdict().items()
                    },
                )
                _json(
                    folder / "stage.json",
                    dict(
                        code=int(observation.code),
                        stage_event=int(observation.event),
                        field_catalog=list(AirSeaState._fields)
                        + [
                            "humidity_quadrature",
                            "humidity_nodes",
                            "air_temperature_quadrature",
                            "sst_quadrature",
                            "air_temperature_nodes",
                            "sst_nodes",
                            "closure_valid",
                        ],
                        coefficient_ranges={
                            k: [float(np.nanmin(v)), float(np.nanmax(v))]
                            for k, v in observation.state._asdict().items()
                            if np.all(np.isfinite(v))
                        },
                        time_seconds=float(observation.time),
                        field_index=int(observation.field_index),
                        flat_index=int(observation.flat_index),
                        meanings={
                            1: "nonfinite state",
                            2: "negative humidity",
                            3: "humidity >=1",
                            4: "nonpositive temperature",
                            5: "surface closure invalid",
                        },
                    ),
                )
                # Save the last valid synchronization state for exact reproduction.
                write_checkpoint(
                    out,
                    RunCheckpoint(
                        before,
                        BudgetLedger(np.asarray(before_ledger), initial),
                        step - 1,
                        seconds,
                        config,
                        prov,
                    ),
                )
                raise RuntimeError(
                    f"Invalid stage: code={int(observation.code)}, t={float(observation.time)}"
                )
            seconds = step * stepper.window
            row = diagnose(p, s, ledger, initial, seconds)
            try:
                _check_budget(config, row)
            except RuntimeError:
                _json(out / "failed_diagnostics.json", row)
                raise
            if (
                aligned(seconds, config.output.diagnostics_seconds)
                or seconds == config.output.duration_seconds
            ):
                with (out / "diagnostics.jsonl").open("a") as f:
                    f.write(json.dumps(row, allow_nan=False) + "\n")
                    f.flush()
                print(
                    f"t={seconds / 86400:.4f} d heat={row['heat_residual']:.3e} water={row['water_residual']:.3e} KE={row['kinetic_residual']}",
                    flush=True,
                )
            if aligned(seconds, config.output.fields_seconds):
                write_fields(out, p, s, step, seconds)
            if (
                aligned(seconds, config.output.checkpoint_seconds)
                or seconds == target
                or stop[0]
            ):
                write_checkpoint(
                    out,
                    RunCheckpoint(
                        s,
                        BudgetLedger(np.asarray(ledger), initial),
                        step,
                        seconds,
                        config,
                        prov,
                    ),
                )
            if stop[0]:
                status = "PAUSED"
                break
        else:
            status = (
                "COMPLETE" if target >= config.output.duration_seconds else "PAUSED"
            )
    except Exception as error:
        status = "FAILED"
        _json(
            out / "failure.json",
            dict(error=repr(error), step=step, time_seconds=seconds),
        )
        if all(np.all(np.isfinite(v)) for v in s):
            failure = out / "failure"
            failure.mkdir(exist_ok=True)
            np.savez_compressed(
                failure / "synchronization_state.npz",
                **{k: np.asarray(v) for k, v in s._asdict().items()},
                ledger=np.asarray(ledger),
            )
        raise
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        signal.signal(signal.SIGTERM, previous_term_handler)
        _json(
            out / "status.json",
            dict(
                status=status,
                step=step,
                time_seconds=seconds,
                invocation_runtime_seconds=time.perf_counter() - start,
            ),
        )
    return report(out)


def run(config, out, *, until=None):
    config = RunConfig.from_dict(config) if isinstance(config, dict) else config
    config.validate()
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    jax.config.update("jax_enable_x64", True)
    p = _plan(config)
    s = initial_air_sea_state(p)
    prov = provenance()
    initial = _initial(p, s)
    checkpoint = RunCheckpoint(
        s, BudgetLedger(np.asarray(zero_ledger()), initial), 0, 0.0, config, prov
    )
    _json(out / "config.json", asdict(config))
    _json(out / "manifest.json", prov)
    write_checkpoint(out, checkpoint)
    # Reuse setup for run through a private optional plan (see _integrate).
    return _integrate(config, out, checkpoint, until)


def resume(path, until=None):
    jax.config.update("jax_enable_x64", True)
    path = Path(path).resolve()
    checkpoint = read_checkpoint(path)
    current = provenance()
    historical = json.loads(Path(__file__).with_name("v1_sources.json").read_text())
    migrated = checkpoint.provenance["source_files"] == historical["source_files"]
    for key in ("source_files", "python", "dependencies", "backend", "float64"):
        previous, actual = checkpoint.provenance[key], current[key]
        if migrated and key == "source_files":
            continue
        if migrated and key == "dependencies":
            actual = {name: version for name, version in actual.items()
                      if name not in ("pybspf", "bspf-models", "bspf-sim")}
        if previous != actual:
            raise ValueError(f"Exact resume environment mismatch: {key}")
    if migrated:
        current["migrated_from"] = checkpoint.provenance
        current["migration_source_commit"] = historical["source_commit"]
        checkpoint = RunCheckpoint(
            checkpoint.state, checkpoint.ledger, checkpoint.step,
            checkpoint.time_seconds, checkpoint.config, current,
        )
    out = path.parent.parent
    if (
        RunConfig.from_dict(json.loads((out / "config.json").read_text()))
        != checkpoint.config
    ):
        raise ValueError("Exact resume configuration mismatch")
    if (
        checkpoint.step < 0
        or checkpoint.time_seconds != checkpoint.step * checkpoint.config.time.window
    ):
        raise ValueError("Checkpoint step/time mismatch")
    # Keep evidence and remove only future outputs from a rollback, never mix
    # duplicate time records with newly recomputed ones.
    history = out / "diagnostics.jsonl"
    if history.exists():
        lines = history.read_text().splitlines()
        kept = [
            line
            for line in lines
            if json.loads(line)["time_seconds"] <= checkpoint.time_seconds
        ]
        if len(kept) != len(lines):
            archive = out / f"rollback_diagnostics_{time.time_ns()}.jsonl"
            os.replace(history, archive)
            history.write_text("\n".join(kept) + ("\n" if kept else ""))
    for f in (out / "fields").glob("step*.nc"):
        if int(f.stem[4:]) > checkpoint.step:
            rollback = out / "rollback_fields"
            rollback.mkdir(exist_ok=True)
            os.replace(f, rollback / (str(time.time_ns()) + "_" + f.name))
    _json(out / "latest_checkpoint.json", {"manifest": str(path.relative_to(out))})
    return _integrate(checkpoint.config, out, checkpoint, until)


def report(run_directory):
    """Read-only summary, suitable for CLI JSON or downstream visualization."""
    out = Path(run_directory)
    status = json.loads((out / "status.json").read_text())
    rows = (
        [json.loads(x) for x in (out / "diagnostics.jsonl").read_text().splitlines()]
        if (out / "diagnostics.jsonl").exists()
        else []
    )
    return dict(
        status=status,
        records=len(rows),
        final=rows[-1] if rows else None,
        max_heat_residual=max((abs(r["heat_residual"]) for r in rows), default=0.0),
        max_water_residual=max((abs(r["water_residual"]) for r in rows), default=0.0),
        max_kinetic_residual=max(
            (max(abs(x) for x in r["kinetic_residual"]) for r in rows), default=0.0
        ),
        scope="24h/30d transient reliability, not climate equilibrium; no condensation or precipitation",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--until", type=float)
    p = sub.add_parser("resume")
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--until", type=float)
    p = sub.add_parser("report")
    p.add_argument("directory", type=Path)
    p = sub.add_parser("validate")
    p.add_argument("--suite", required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        result = run(json.loads(args.config.read_text()), args.out, until=args.until)
    elif args.command == "resume":
        result = resume(args.checkpoint, args.until)
    elif args.command == "report":
        result = report(args.directory)
    else:
        from bspf_sim.air_sea.validation import validate

        result = validate(
            args.suite,
            RunConfig.from_dict(json.loads(args.config.read_text())),
            args.out,
        )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
