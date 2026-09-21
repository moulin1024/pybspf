"""Reproducible synchronized timings for the rational-obstacle GPU workflow."""
import argparse
import cProfile
import json
import os
from pathlib import Path
from time import perf_counter

import jax
import numpy as np
from bspf_models.fluids.immersed_flow import ImmersedFlowPlan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--basis-precision", choices=("mpfr", "float64"), default="mpfr",
                        help="Basis arithmetic: mpfr reference or opt-in GPU float64")
    parser.add_argument("--basis-workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_enable_x64", True)
    device = jax.devices("gpu")[0]
    record = dict(
        device=str(device), device_kind=device.device_kind,
        basis_workers=args.basis_workers if args.basis_precision == "mpfr" else 0,
        basis_precision=args.basis_precision,
        jax_version=jax.__version__, grid=[73,33], steps=args.steps,
        openblas_threads=os.environ.get("OPENBLAS_NUM_THREADS"),
        omp_threads=os.environ.get("OMP_NUM_THREADS"),
    )
    start = perf_counter()
    p = ImmersedFlowPlan(nx=73,ny=33,wall_method="rational",assembly_device=device,basis_workers=args.basis_workers,basis_precision=args.basis_precision)
    record["setup_seconds"] = perf_counter()-start
    record["line_setup_seconds"] = p.line_seconds
    print("SETUP", record, flush=True)
    start=perf_counter()
    step=p.stepper(0.02,device=device)
    jax.block_until_ready(step.data)
    record["upload_and_factor_seconds"]=perf_counter()-start
    record.update(step.setup_timings)
    start=perf_counter()
    state=step.step(step.initial_state,0.0)
    state.block_until_ready()
    record["first_step_compile_seconds"]=perf_counter()-start
    start=perf_counter()
    for k in range(args.steps):
        state=step.step(state,(k+1)*0.02)
    state.block_until_ready()
    record["steps_seconds"]=perf_counter()-start
    record["seconds_per_step"]=record["steps_seconds"]/args.steps
    jax.block_until_ready(step.diagnostics(state))
    start=perf_counter()
    for _ in range(20):
        values=step.diagnostics(state)
    jax.block_until_ready(values)
    record["seconds_per_diagnostic"]=(perf_counter()-start)/20
    host=jax.device_get(state)
    x,y=np.linspace(-1,5,401),np.linspace(-1,1,161)
    profiler=cProfile.Profile()
    start=perf_counter()
    profiler.enable()
    fields=p.grid(host,x,y)
    profiler.disable()
    record["first_grid_seconds"]=perf_counter()-start
    profiler.dump_stats(str(args.out/"grid.prof"))
    start=perf_counter()
    fields=p.grid(host,x,y)
    record["repeat_grid_seconds"]=perf_counter()-start
    start=perf_counter()
    # Forty-one output times, as in the full t=20 run. Repeated state isolates
    # reconstruction cost from evolution and gives an exact parity reference.
    batch=list(p.grid_many(np.repeat(host[None,:],41,axis=0),x,y))
    record["batch_41_grids_seconds"]=perf_counter()-start
    record["batch_grid_max_difference"]=float(np.nanmax(np.abs(batch[-1]["u"]-fields["u"])))
    record["diagnostics"]={k:float(v) for k,v in jax.device_get(values).items()}
    np.savez_compressed(args.out/"fields.npz",state=host,**fields)
    (args.out/"timings.json").write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps(record,indent=2),flush=True)


if __name__=="__main__":
    main()
