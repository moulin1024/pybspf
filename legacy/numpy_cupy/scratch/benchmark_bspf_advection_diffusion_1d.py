"""Preserve BSPF accuracy while eliminating 1D boundary growth: reproducible study."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import scipy.linalg as la

from bspf_weak_advection_diffusion_1d import (
    assemble,
    analytic_field,
    exact_semidiscrete_error,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assembly-bits", type=int)
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[40, 48, 64, 96, 128, 160, 256, 384, 512],
    )
    parser.add_argument("--windows", nargs="+", type=int, default=[16, 32])
    parser.add_argument(
        "--out", type=Path, default=Path("build/bspf_advection_diffusion_1d")
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for window in args.windows:
        for n in args.sizes:
            start = time.perf_counter()
            plan = assemble(n, window=window, assembly_bits=args.assembly_bits)
            weak = plan.generator()
            strong = plan.generator(weak=False)
            lower = la.cholesky(plan.mass, lower=True)
            generators = {"strong": strong, "weak": weak}
            spectra = {}
            for name, a in generators.items():
                symmetric = (plan.mass @ a + a.T @ plan.mass) / 2
                spectra[name] = dict(
                    max_eigenvalue_real=float(la.eigvals(a).real.max()),
                    mass_energy_max_eigenvalue=float(
                        la.eigh(symmetric, plan.mass, eigvals_only=True)[-1]
                    ),
                )
            cases = {}
            for case in ("nonperiodic", "oscillatory", "gaussian"):
                pn, pxn, pxxn = analytic_field(plan.x, case)
                pq, pxq, pxxq = analytic_field(plan.quadrature_x, case)
                interpolation = plan.values @ pn[1:-1] - pq
                loads = {
                    "strong": (-pn + pxn - 0.002 * pxxn)[1:-1],
                    "weak": plan.project_load(-pq + pxq - 0.002 * pxxq),
                }
                case_result = dict(interpolation_linf=float(abs(interpolation).max()))
                for name, a in generators.items():
                    errors = []
                    for t in (1.0, 3.0, 6.0):
                        difference = exact_semidiscrete_error(
                            a, pn[1:-1], loads[name], t
                        )
                        reconstructed_error = (
                            plan.values @ difference + np.exp(-t) * interpolation
                        )
                        errors.append(
                            dict(
                                time=t,
                                nodal_linf=float(abs(difference).max()),
                                continuous_l2=float(
                                    np.sqrt(
                                        np.sum(
                                            plan.quadrature_weights
                                            * reconstructed_error**2
                                        )
                                    )
                                ),
                            )
                        )
                    steady_load = (
                        (pxn - 0.002 * pxxn)[1:-1]
                        if name == "strong"
                        else plan.project_load(pxq - 0.002 * pxxq)
                    )
                    steady = la.solve(-a, steady_load)
                    case_result[name] = dict(
                        transient=errors,
                        steady_nodal_linf=float(abs(steady - pn[1:-1]).max()),
                    )
                cases[case] = case_result
            record = dict(
                n=n,
                assembly_bits=args.assembly_bits,
                window=window,
                q=9,
                modes=12,
                mass_condition=float(np.linalg.cond(plan.mass)),
                mass_min_eigenvalue=float(la.eigvalsh(plan.mass)[0]),
                projector_norm=plan.projector_norm,
                quadrature_order=plan.quadrature_order,
                raw_integration_by_parts_defect=plan.raw_sbp_defect,
                energy_identity_relative=float(
                    la.norm(
                        plan.mass @ weak + weak.T @ plan.mass + 0.004 * plan.stiffness
                    )
                    / la.norm(0.004 * plan.stiffness)
                ),
                spectra=spectra,
                cases=cases,
                elapsed_seconds=time.perf_counter() - start,
            )
            if n == 160 and window == 16:
                times = np.linspace(0, 3, 31)
                norms = {}
                for name, a in generators.items():
                    scaled = la.solve_triangular(lower, (lower.T @ a).T, lower=True).T
                    norms[name] = [
                        float(la.svdvals(la.expm(t * scaled))[0]) for t in times
                    ]
                record["mass_semigroup"] = dict(times=times.tolist(), **norms)
                refined = assemble(
                    n,
                    window=window,
                    quadrature_order=2 * plan.quadrature_order,
                    assembly_bits=args.assembly_bits,
                )
                rgen = refined.generator()
                phi, px, pxx = analytic_field(refined.quadrature_x, "nonperiodic")
                pn = analytic_field(refined.x, "nonperiodic")[0][1:-1]
                baseline = exact_semidiscrete_error(
                    weak,
                    pn,
                    plan.project_load(
                        -analytic_field(plan.quadrature_x, "nonperiodic")[0]
                        + analytic_field(plan.quadrature_x, "nonperiodic")[1]
                        - 0.002 * analytic_field(plan.quadrature_x, "nonperiodic")[2]
                    ),
                    3,
                )
                high = exact_semidiscrete_error(
                    rgen, pn, refined.project_load(-phi + px - 0.002 * pxx), 3
                )
                record["quadrature_double_order"] = dict(
                    solution_difference_linf=float(abs(high - baseline).max()),
                    refined_error_linf=float(abs(high).max()),
                    relative_mass_difference=float(
                        la.norm(refined.mass - plan.mass) / la.norm(plan.mass)
                    ),
                )
                np.savez_compressed(
                    args.out / "operators_n160_w16.npz",
                    x=plan.x,
                    strong=strong,
                    weak=weak,
                    mass=plan.mass,
                    convection=plan.convection,
                    stiffness=plan.stiffness,
                    load=plan.project_load(
                        -analytic_field(plan.quadrature_x, "nonperiodic")[0]
                        + analytic_field(plan.quadrature_x, "nonperiodic")[1]
                        - 0.002 * analytic_field(plan.quadrature_x, "nonperiodic")[2]
                    ),
                    initial=pn,
                    reference=np.exp(-3) * pn + baseline,
                )
            records.append(record)
            (args.out / "results.json").write_text(json.dumps(records, indent=2))
            print(
                n,
                window,
                record["spectra"],
                {k: v["weak"]["transient"][1]["nodal_linf"] for k, v in cases.items()},
                flush=True,
            )


if __name__ == "__main__":
    main()
