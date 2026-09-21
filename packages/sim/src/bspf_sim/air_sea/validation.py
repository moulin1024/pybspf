"""Reproducible acceptance suites; results contain differences, not truth claims."""

from dataclasses import asdict, replace
import json
from pathlib import Path

import jax
import numpy as np

from bspf_models.air_sea.air_sea import AirSeaState
from bspf_models.air_sea.air_sea import air_sea_fields
from bspf_models.air_sea.air_sea import bulk_flux
from bspf_sim.air_sea.platform import RunConfig
from bspf_sim.air_sea.platform import run
from bspf_sim.air_sea.platform import resume
from bspf_sim.air_sea.platform import report
from bspf_sim.air_sea.platform import read_checkpoint
from bspf_sim.air_sea.platform import _json
from bspf_sim.air_sea.platform import _plan
from bspf_sim.air_sea.platform import source_fingerprint


def _finish(config, directory, *, restart=False):
    directory = Path(directory)
    if directory.exists():
        saved = json.loads((directory / "config.json").read_text())
        if saved != asdict(config):
            raise ValueError(f"Existing run has different config: {directory}")
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["source_files"] != source_fingerprint():
            raise ValueError(f"Existing run has different source: {directory}")
        status = json.loads((directory / "status.json").read_text())
        if status["status"] == "FAILED":
            raise RuntimeError(
                f"Retaining failed evidence; choose a new directory: {directory}"
            )
        if status["status"] != "COMPLETE":
            checkpoint = (
                directory
                / json.loads((directory / "latest_checkpoint.json").read_text())[
                    "manifest"
                ]
            )
            resume(checkpoint, config.output.duration_seconds)
    else:
        run(config, directory, until=86400.0 if restart else None)
        if restart:
            paused = report(directory)["status"]
            if paused["time_seconds"] < 86400.0:
                raise InterruptedError(f"Validation interrupted: {directory}")
            checkpoint = (
                directory
                / json.loads((directory / "latest_checkpoint.json").read_text())[
                    "manifest"
                ]
            )
            resume(checkpoint, config.output.duration_seconds)
    checkpoint = (
        directory
        / json.loads((directory / "latest_checkpoint.json").read_text())["manifest"]
    )
    result = report(directory)
    if result["status"]["status"] != "COMPLETE":
        raise InterruptedError(f"Validation interrupted: {directory}")
    return read_checkpoint(checkpoint), result


def resolution_fields(p, s):
    f = {k: np.asarray(v) for k, v in air_sea_fields(p, s, nodes=False).items()}
    for name, coeff, air in (
        ("sst", s.sst, False),
        ("air_temperature", s.air_temperature, True),
        ("humidity", s.humidity, True),
    ):
        x, y = (p.air_flow.x, p.air_scalar_y) if air else (p.scalar, p.scalar)
        f[name + "_gradient"] = (
            np.stack(
                (
                    np.asarray(x.g) @ coeff @ np.asarray(y.b).T,
                    np.asarray(x.b) @ coeff @ np.asarray(y.g).T,
                ),
                -1,
            )
            / p.config.length
        )
    tau, h, e = bulk_flux(
        p.config,
        f["ocean_velocity"],
        f["air_velocity"],
        f["sst"],
        f["air_temperature"],
        f["humidity"],
        surface=p.surface,
    )
    f.update(
        stress=np.asarray(tau),
        sensible=np.asarray(h),
        evaporation=np.asarray(e),
        total_heat=np.asarray(h + p.config.latent_heat * e),
    )
    return f


def compare(weight, a, b):
    result = {}

    def magnitude(v):
        return np.linalg.norm(v, axis=-1) if v.ndim == 3 else abs(v)

    for name in a:
        delta = magnitude(a[name] - b[name])
        reference = b[name]
        if name in ("sst", "air_temperature"):
            reference = reference - np.sum(weight * reference)
        scale = magnitude(reference)
        rms = float(np.sqrt(np.sum(weight * delta**2)))
        ref = float(np.sqrt(np.sum(weight * scale**2)))
        result[name] = dict(
            rms=rms,
            reference_rms=ref,
            reference_peak=float(scale.max()),
            roundoff_candidate=bool(
                rms
                <= 128 * np.finfo(float).eps * max(float(np.max(abs(b[name]))), 1e-30)
            ),
            relative_rms=rms / max(ref, 1e-30),
            linf=float(delta.max()),
            relative_linf=float(delta.max()) / max(float(scale.max()), 1e-30),
        )
    return result


def transfer_quadrature_basis(source, target, s):
    """Same n and same trial space: exact coefficient basis change via node traces.

    This is not interpolation between spatial resolutions. QR/eigen rotations
    differ when setup quadrature changes, so coefficients cannot be reused raw.
    """

    def map_coeff(a, sx, sy, tx, ty):
        left = np.linalg.lstsq(np.asarray(tx.bn), np.asarray(sx.bn), rcond=None)[0]
        right = np.linalg.lstsq(np.asarray(ty.bn), np.asarray(sy.bn), rcond=None)[0]
        return left @ np.asarray(a) @ right.T

    return AirSeaState(
        map_coeff(s.ocean, source.flow.x, source.flow.y, target.flow.x, target.flow.y),
        map_coeff(
            s.atmosphere,
            source.air_flow.x,
            source.air_flow.y,
            target.air_flow.x,
            target.air_flow.y,
        ),
        map_coeff(s.sst, source.scalar, source.scalar, target.scalar, target.scalar),
        map_coeff(
            s.air_temperature,
            source.air_flow.x,
            source.air_scalar_y,
            target.air_flow.x,
            target.air_scalar_y,
        ),
        map_coeff(
            s.humidity,
            source.air_flow.x,
            source.air_scalar_y,
            target.air_flow.x,
            target.air_scalar_y,
        ),
        s.air_mean_wind,
    )


def _evaluate(config, checkpoint):
    source = _plan(config)
    target = _plan(
        replace(config, spatial=replace(config.spatial, quadrature_order=48))
    )
    state = transfer_quadrature_basis(source, target, checkpoint.state)
    return np.asarray(target.weight), resolution_fields(target, state)


def validate(suite, config, out):
    if suite not in (
        "smoke",
        "resolution",
        "quadrature",
        "sensitivity",
        "reliability",
        "all",
    ):
        raise ValueError("Unknown validation suite")
    config = RunConfig.from_dict(config) if isinstance(config, dict) else config
    config.validate()
    jax.config.update("jax_enable_x64", True)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if suite == "all":
        results = {}
        for method in ("constant", "coare35"):
            c = replace(config, surface=replace(config.surface, method=method))
            for s in ("resolution", "quadrature"):
                results[method + "_" + s] = validate(s, c, out / (method + "_" + s))
        for s in ("sensitivity", "reliability"):
            results[s] = validate(s, config, out / s)
        _json(out / "validation.json", results)
        return results
    if suite == "smoke":
        c = replace(
            config,
            spatial=replace(config.spatial, n=33, quadrature_order=20),
            output=replace(
                config.output,
                duration_seconds=3600.0,
                diagnostics_seconds=600.0,
                fields_seconds=1800.0,
                checkpoint_seconds=1800.0,
            ),
        )
        _, result = _finish(c, out / "smoke")
        _json(out / "validation.json", result)
        return result
    if suite in ("sensitivity", "reliability"):
        cases = {}
        if suite == "reliability":
            for method in ("constant", "coare35"):
                cases[method] = replace(
                    config,
                    surface=replace(config.surface, method=method),
                    output=replace(
                        config.output,
                        duration_seconds=30 * 86400.0,
                        fields_seconds=21600.0,
                        checkpoint_seconds=86400.0,
                    ),
                )
        else:
            for height in (5.0, 10.0, 20.0):
                cases[f"height_{height:g}"] = replace(
                    config,
                    surface=replace(
                        config.surface,
                        method="coare35",
                        wind_height=height,
                        temperature_height=height,
                        humidity_height=height,
                    ),
                )
            cases["neutral"] = replace(
                config,
                surface=replace(config.surface, method="coare35", stability=False),
            )
            cases["weak_wind"] = replace(
                config,
                surface=replace(config.surface, method="coare35"),
                physics=replace(config.physics, wind_speed=0.2),
            )
            cases["stable"] = replace(
                config,
                surface=replace(config.surface, method="coare35"),
                physics=replace(
                    config.physics,
                    initial_air_temperature_offset=2.0,
                    initial_relative_humidity=0.9,
                ),
            )
            cases["closed"] = replace(
                config,
                surface=replace(config.surface, method="constant"),
                physics=replace(
                    config.physics,
                    radiation=0.0,
                    wind_restore_rate=0.0,
                    temperature_restore_rate=0.0,
                    humidity_restore_rate=0.0,
                ),
            )
        result = {
            name: _finish(c, out / name, restart=(suite == "reliability"))[1]
            for name, c in cases.items()
        }
        _json(out / "validation.json", result)
        return result

    # An hourly complete checkpoint gives actual 6/12/24 h states for audit.
    base = replace(
        config,
        output=replace(
            config.output, duration_seconds=86400.0, checkpoint_seconds=3600.0
        ),
    )
    cases = {}
    if suite == "resolution":
        for n in (33, 49, 65, 81):
            cases[f"n{n}_H150"] = replace(
                base,
                spatial=replace(base.spatial, n=n, quadrature_order=32),
                time=replace(base.time, dt_ocean=150.0, dt_air=30.0),
            )
        for n in (33, 81):
            for h in (600.0, 300.0):
                cases[f"n{n}_H{h:g}"] = replace(
                    base,
                    spatial=replace(base.spatial, n=n, quadrature_order=32),
                    time=replace(base.time, dt_ocean=h, dt_air=h / 5),
                )
    else:
        for q in (32, 40, 48):
            cases[f"q{q}"] = replace(
                base,
                spatial=replace(base.spatial, n=81, quadrature_order=q),
                time=replace(base.time, dt_ocean=150.0, dt_air=30.0),
            )
        # Independent spatial comparator, at exactly the same H and q=48.
        cases["n65_q48"] = replace(
            cases["q48"], spatial=replace(cases["q48"].spatial, n=65)
        )
    runs = {name: _finish(c, out / name)[1] for name, c in cases.items()}
    differences, accepted = {}, True
    for hour in (6, 12, 24):
        fields = {}
        for name, c in cases.items():
            step = round(hour * 3600 / c.time.window)
            checkpoint = read_checkpoint(
                out / name / "checkpoints" / f"step{step:09d}.json"
            )
            weight, fields[name] = _evaluate(c, checkpoint)
        metrics = {}
        if suite == "resolution":
            for a, b in ((33, 49), (49, 65), (65, 81)):
                metrics[f"space_{a}_{b}"] = compare(
                    weight, fields[f"n{a}_H150"], fields[f"n{b}_H150"]
                )
            for n in (33, 81):
                for a, b in ((600, 300), (300, 150)):
                    metrics[f"time_n{n}_{a}_{b}"] = compare(
                        weight, fields[f"n{n}_H{a}"], fields[f"n{n}_H{b}"]
                    )
            flags = {}
            for key, last in metrics["space_65_81"].items():
                t = metrics["time_n81_300_150"][key]["rms"]
                flags[key] = (
                    last["relative_rms"] < config.validation.space_rms_limit
                    and last["relative_linf"] < config.validation.space_peak_limit
                    and last["rms"] < metrics["space_49_65"][key]["rms"]
                    and t < config.validation.error_separation_ratio * last["rms"]
                )
        else:
            for a, b in ((32, 40), (40, 48)):
                metrics[f"quadrature_{a}_{b}"] = compare(
                    weight, fields[f"q{a}"], fields[f"q{b}"]
                )
            metrics["space_65_81"] = compare(weight, fields["n65_q48"], fields["q48"])
            flags = {
                k: (
                    metrics["quadrature_40_48"][k]["rms"]
                    < config.validation.error_separation_ratio * v["rms"]
                    and metrics["quadrature_32_40"][k]["rms"]
                    < config.validation.error_separation_ratio * v["rms"]
                )
                for k, v in metrics["space_65_81"].items()
            }
        accepted = accepted and all(flags.values())
        differences[str(hour)] = dict(metrics=metrics, passed=flags)
    result = dict(
        suite=suite,
        passed=accepted,
        config=asdict(config),
        runs=runs,
        differences=differences,
        scope="finite refinement screening, not exact error bounds",
    )
    _json(out / "validation.json", result)
    if not accepted:
        raise RuntimeError(f"{suite} criteria failed; evidence retained in {out}")
    return result
