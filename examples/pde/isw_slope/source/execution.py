"""Portable run/restart driver for the JAX slope adapter.

New recovery tooling, not an original Run01 file. Units at the command line
are metres/seconds; the retained solver uses H=100 m and U=1 m/s.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any
import numpy as np
import scipy.linalg as la
from common import ROOT
from slope_solver import BSPF, backend

SCHEMA = "bspf-slope-recovery-v1"
CASES = {"R0_193x97": (193, 97), "R1_257x129": (257, 129), "R2_321x161": (321, 161)}


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def mathematical_identity() -> str:
    """Fingerprint all code that computes the RHS, bases, geometry or initial."""
    paths = [
        ROOT / "source/common.py",
        ROOT / "source/slope_solver.py",
        ROOT / "source/evaluation.py",
        ROOT / "reference/shared_initial.npz",
    ]
    import importlib
    for name in (
        "pybspf.trial_spaces", "pybspf._qr_trial", "pybspf.tensor",
        "pybspf.time_integration", "bspf_models.fluids.mapped_boussinesq",
        "bspf_models.fluids.isw_slope", "bspf_models._numerics._tensor_pcg",
    ):
        paths.append(Path(importlib.import_module(name).__file__))
    text = "\n".join(
        f"{p.name} {sha_file(p) if p.is_file() else 'missing'}" for p in paths
    )
    return hashlib.sha256(text.encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temp, path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(temp, path)


def state_name(seconds: float) -> str:
    if abs(seconds - round(seconds)) < 1e-7:
        return f"state_{int(round(seconds)):05d}.npz"
    token = f"{seconds:012.6f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"state_{token}.npz"


def save_basis(model: BSPF, path: Path) -> None:
    atomic_npz(path, Qx=model.Qx, Qz=model.Qz, BQx=model.BQx, BQz=model.BQz)


def load_restart(
    model: BSPF, snapshot: Path
) -> tuple[float, np.ndarray, np.ndarray, dict]:
    """Convert saved basis coordinates into the new full basis when necessary.

    Generalized eigenvectors can change signs across LAPACK builds. Simply
    reusing coefficients would then be incorrect. No state is interpolated to
    another grid here, and cross-grid restart is explicitly rejected.
    """
    snapshot = snapshot.resolve()
    cfg_path, basis_path = (
        snapshot.parent / "config.json",
        snapshot.parent / "basis.npz",
    )
    if not cfg_path.is_file() or not basis_path.is_file():
        raise FileNotFoundError(
            "A restart needs config.json and basis.npz beside the snapshot."
        )
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if int(cfg["nx"]) != model.nx or int(cfg["nz"]) != model.nz:
        raise ValueError("Cross-grid restarts are not supported by this driver.")
    if abs(float(cfg.get("quad", 2.0)) - model.quad_factor) > 1e-12:
        raise ValueError("Restart quadrature differs from the source run.")
    expected = cfg.get("mathematical_identity")
    if expected and expected != mathematical_identity():
        raise ValueError(
            "Mathematical source fingerprint changed; refuse an implicit restart."
        )
    old_init = cfg.get("initial_sha256")
    if old_init and old_init != sha_file(ROOT / "reference/shared_initial.npz"):
        raise ValueError(
            "The frozen shared initial file does not match the source run."
        )
    with np.load(snapshot, allow_pickle=False) as f:
        t_s = float(f["time_s"]) if "time_s" in f else float(f["t"]) * 100.0
        a = np.array(f["a"], dtype=float, copy=True)
        b = np.array(f["b"], dtype=float, copy=True)
    if a.shape != model.shape or b.shape != model.bshape:
        raise ValueError("Saved coefficient shapes do not match the model.")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("Checkpoint contains non-finite values.")
    with np.load(basis_path, allow_pickle=False) as f:
        old = {k: f[k].copy() for k in ("Qx", "Qz", "BQx", "BQz")}
    changed = []

    def convert(c: np.ndarray, key_x: str, key_z: str) -> np.ndarray:
        ax, az = getattr(model, key_x), getattr(model, key_z)
        tx = (
            np.eye(ax.shape[0])
            if np.array_equal(ax, old[key_x])
            else la.solve(ax, old[key_x])
        )
        tz = (
            np.eye(az.shape[0])
            if np.array_equal(az, old[key_z])
            else la.solve(az, old[key_z])
        )
        if np.array_equal(ax, old[key_x]) and np.array_equal(az, old[key_z]):
            return c
        changed.append([key_x, key_z])
        return tx @ c @ tz.T

    a, b = convert(a, "Qx", "Qz"), convert(b, "BQx", "BQz")
    return (
        t_s,
        a,
        b,
        dict(
            snapshot=str(snapshot),
            snapshot_sha256=sha_file(snapshot),
            source_config=cfg,
            basis_conversions=changed,
        ),
    )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Recovered slope BSPF with JAX float64 backend."
    )
    p.add_argument("--case", choices=CASES, default="R2_321x161")
    p.add_argument(
        "--nx", type=int, help="Override construct points; use together with --nz."
    )
    p.add_argument("--nz", type=int)
    p.add_argument("--quad", type=float, default=2.0)
    p.add_argument("--dt", type=float, default=2.0, help="Early timestep in seconds.")
    p.add_argument(
        "--dt-after", type=float, default=1.0, help="Late timestep in seconds."
    )
    p.add_argument(
        "--switch", type=float, default=1200.0, help="Switch time in seconds."
    )
    p.add_argument(
        "--tfinal", type=float, default=1800.0, help="Absolute final time in seconds."
    )
    p.add_argument(
        "--save", type=float, default=50.0, help="Snapshot spacing in seconds."
    )
    p.add_argument("--out", type=Path)
    p.add_argument(
        "--resume",
        type=Path,
        help="Snapshot to restart, with config.json and basis.npz beside it.",
    )
    p.add_argument(
        "--nowave",
        action="store_true",
        help="No incident wave; keep total-buoyancy diffusion.",
    )
    p.add_argument(
        "--test",
        action="store_true",
        help="Run only two 2-second RK4 steps, not the physical experiment.",
    )
    return p


def run(args: argparse.Namespace) -> dict:
    if (args.nx is None) != (args.nz is None):
        raise ValueError("--nx and --nz must be specified together.")
    nx, nz = CASES[args.case] if args.nx is None else (args.nx, args.nz)
    if nx < 33 or nz < 33:
        raise ValueError(
            "This restored production basis is tested here only for nx,nz >= 33."
        )
    for name in ("quad", "dt", "dt_after", "save"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    if args.quad < 2:
        raise ValueError(
            "Use quadrature >= 2; lower quadrature is not part of this recovery."
        )
    if (
        not math.isfinite(args.tfinal)
        or args.tfinal <= 0
        or not math.isfinite(args.switch)
        or args.switch < 0
    ):
        raise ValueError("Invalid final/switch time.")
    if args.resume and (args.test or args.nowave):
        raise ValueError("--resume cannot be combined with --test or --nowave.")
    if args.test:
        args.tfinal, args.dt, args.dt_after, args.save = 4.0, 2.0, 2.0, 2.0
    if (
        not args.nowave
        and not args.resume
        and not (ROOT / "reference/shared_initial.npz").is_file()
    ):
        raise FileNotFoundError(
            "Missing frozen reconstructed initial: reference/shared_initial.npz"
        )
    out = (
        args.out or ROOT / "rerun" / (args.case + ("_test" if args.test else ""))
    ).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    tic = time.perf_counter()
    model = BSPF(nx, nz, args.quad)
    model.quad_factor = args.quad
    if args.resume:
        t_s, a, b, restart = load_restart(model, args.resume)
    else:
        t_s, restart = 0.0, None
        a, b = (
            (np.zeros(model.shape), np.zeros(model.bshape))
            if args.nowave
            else model.initial()
        )
    if args.tfinal <= t_s + 1e-9:
        raise ValueError("Final time must be later than checkpoint time.")
    start_s = t_s
    cfg = dict(
        schema=SCHEMA,
        case=args.case,
        nx=nx,
        nz=nz,
        quad=args.quad,
        dt=args.dt,
        dt_after=args.dt_after,
        switch_s=args.switch,
        tfinal=args.tfinal,
        save=args.save,
        start_s=start_s,
        nowave=bool(args.nowave),
        smoke_test=bool(args.test),
        original_Run01_source=False,
        backend=model.backend_name,
        initial_sha256=(
            sha_file(ROOT / "reference/shared_initial.npz")
            if (ROOT / "reference/shared_initial.npz").is_file()
            else None
        ),
        mathematical_identity=mathematical_identity(),
        restart=restart,
        units="CLI seconds; internal H=100m, U=1m/s, time=100s",
        physical=dict(
            L_m=1800.0,
            nu_m2s=0.01,
            kappa_m2s=0.003,
            deep_depth_m=100.0,
            shelf_depth_m=45.0,
            slope_center_m=850.0,
            slope_width_m=90.0,
            wave_center_m=400.0,
            nominal_DJL_amplitude_m=16.0,
            background_center_m=-25.0,
            background_thickness_m=6.0,
            boundary="all walls no-slip impermeable; total buoyancy natural no flux",
            forcing="none",
        ),
    )
    atomic_json(out / "config.json", cfg)
    save_basis(model, out / "basis.npz")
    print(
        f"BUILD {nx}x{nz}; {model.build_seconds:.3f}s; initial={start_s:g}s; output={out}",
        flush=True,
    )
    rows, steps = [], 0

    def write_state() -> None:
        row = model.diagnostics(t_s / 100.0, a, b)
        row["steps_this_invocation"] = steps
        row["wall_seconds_this_invocation"] = time.perf_counter() - tic
        rows.append(row)
        atomic_npz(
            out / state_name(t_s),
            t=t_s / 100.0,
            time_s=t_s,
            a=a,
            b=b,
            steps_this_invocation=steps,
            schema=SCHEMA,
        )
        atomic_json(out / "diagnostics.json", rows)
        atomic_json(
            out / "status.json",
            dict(
                completed=False,
                last_saved_s=t_s,
                steps_this_invocation=steps,
                elapsed_seconds=time.perf_counter() - tic,
            ),
        )
        print(
            f"t={t_s:8.3f}s  steps={steps:5d}  minN2={row['min_N2']:.4e}  "
            f"CG={row['mass_CG_mean']:.2f}  residual={row['mass_residual']:.3e}  "
            f"elapsed={time.perf_counter() - tic:.1f}s",
            flush=True,
        )

    write_state()
    next_save = (math.floor((t_s + 1e-8) / args.save) + 1) * args.save
    try:
        while t_s < args.tfinal - 1e-8:
            dt = args.dt if t_s < args.switch - 1e-8 else args.dt_after
            target = min(t_s + dt, next_save, args.tfinal)
            if t_s < args.switch - 1e-8:
                target = min(target, args.switch)
            h = target - t_s
            if h <= 0:
                raise RuntimeError("Timestep scheduler failed to advance.")
            a1, b1 = model.rk4(a, b, h / 100.0)
            # The JAX adapter validates CG and state finiteness on device before
            # returning; do not copy both coefficient fields to NumPy per step.
            a, b, t_s = a1, b1, target
            steps += 1
            if t_s >= next_save - 1e-8 or t_s >= args.tfinal - 1e-8:
                write_state()
                if t_s >= next_save - 1e-8:
                    next_save += args.save
    except BaseException as exc:
        atomic_npz(
            out / "interrupted_checkpoint.npz",
            t=t_s / 100.0,
            time_s=t_s,
            a=a,
            b=b,
            steps_this_invocation=steps,
            schema=SCHEMA,
        )
        atomic_json(
            out / "status.json",
            dict(
                completed=False,
                last_valid_s=t_s,
                error=repr(exc),
                steps_this_invocation=steps,
            ),
        )
        raise
    summary = dict(
        schema=SCHEMA,
        completed=True,
        original_Run01_source=False,
        solver="BSPF slope with JAX float64 evolution",
        backend=model.backend_name,
        nx=nx,
        nz=nz,
        start_s=start_s,
        final_s=t_s,
        steps=steps,
        stages=4 * steps,
        smoke_test=bool(args.test),
        wall_seconds=time.perf_counter() - tic,
        build_seconds=model.build_seconds,
        mass_residual_max=model.residual_max,
        kinetic_budget_rel_max=max(r["kinetic_budget_rel"] for r in rows),
        convergence_claim="None: completing a run is not a convergence certificate.",
    )
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "status.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    args = parser().parse_args()
    try:
        run(args)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
