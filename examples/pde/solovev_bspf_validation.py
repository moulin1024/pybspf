"""BSPF fixed-boundary GS convergence on exact Solov'ev flux surfaces."""

from importlib import import_module

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import jax
import numpy as np
import scipy
import scipy.linalg as la
from scipy.optimize import root

from bspf_models.plasma.grad_shafranov import FixedBoundaryGSPlan
from bspf_models.plasma.grad_shafranov import evaluate_gs_factors
from bspf_models.plasma.solovev import SolovevEquilibrium
from bspf_models.plasma.solovev import SolovevFluxDomain

jax.config.update("jax_enable_x64", True)


def validation_grid(domain, nx=127, nz=131):
    low, high = domain.bounds[:, 0], domain.bounds[:, 1]
    x = low[0]+(high[0]-low[0])*(np.arange(nx)+0.613)/nx
    z = low[1]+(high[1]-low[1])*(np.arange(nz)+0.371)/nz
    xx, zz = np.meshgrid(x, z, indexing="ij")
    points = np.column_stack((xx.ravel()+domain.major_radius, zz.ravel()))
    inside = domain.equilibrium.jets(points)[0] > 0
    return points[inside], inside.reshape(nx, nz), x+domain.major_radius, z


def metrics(solution, eq, points, boundary_count):
    psi, grad, h = solution.jets(points)
    exact, eg, eh = eq.jets(points)
    r = points[:, 0]
    delta = h[:, 0, 0]+h[:, 1, 1]-grad[:, 0]/r
    source = eq.source(points)
    bp = np.column_stack((-grad[:, 1], grad[:, 0]))/r[:, None]
    ebp = np.column_stack((-eg[:, 1], eg[:, 0]))/r[:, None]
    f = eq.toroidal_function(psi)
    b = np.column_stack((bp[:, 0], f/r, bp[:, 1]))
    exact_b = eq.magnetic_field(points)
    j = np.column_stack((-eq.ff_prime/f*grad[:, 1], -delta,
                         eq.ff_prime/f*grad[:, 0]))/(eq.mu0*r[:, None])
    pressure_gradient = np.column_stack((eq.p_prime*grad[:, 0], np.zeros(len(r)), eq.p_prime*grad[:, 1]))
    balance = np.cross(j, b)-pressure_gradient
    def h2(v, g, hess):
        return np.sqrt(np.sum(v*v)+np.sum(g*g)+np.sum(hess*hess))
    edge, parameter = solution.plan.arc.sample(boundary_count, offset=0.371)
    normals = solution.plan.domain.normal(parameter)
    edge = edge+[eq.major_radius, 0]
    edge_flux, edge_grad, _ = solution.jets(edge)
    edge_bp = np.column_stack((-edge_grad[:, 1], edge_grad[:, 0]))/edge[:, :1]
    result = dict(
        independent_points=len(points),
        flux_relative_l2=float(la.norm(psi-exact)/la.norm(exact)),
        flux_max_over_axis=float(np.max(np.abs(psi-exact))/eq.axis_flux),
        poloidal_field_relative_l2=float(la.norm(bp-ebp)/la.norm(ebp)),
        total_field_relative_l2=float(la.norm(b-exact_b)/la.norm(exact_b)),
        h2_relative=float(h2(psi-exact, grad-eg, h-eh)/h2(exact, eg, eh)),
        gs_relative_residual=float(la.norm(-delta-source)/la.norm(source)),
        force_balance_relative=float(la.norm(balance)/la.norm(pressure_gradient)),
        boundary_flux_max=float(np.max(np.abs(edge_flux))),
        boundary_normal_field_max=float(np.max(np.abs(np.sum(edge_bp*normals, axis=1)))),
    )
    # Root location is an independent derived equilibrium diagnostic.
    def axis_equations(x):
        _, g, hess = solution.jets(np.asarray(x)[None, :])
        return g[0], hess[0]
    axis = root(axis_equations, [eq.major_radius+0.015, 0.012], jac=True, tol=1e-10)
    axis_gradient = la.norm(axis_equations(axis.x)[0])
    if axis_gradient > 1e-8:
        raise RuntimeError("Numerical magnetic-axis search failed")
    axis_value = solution.jets(axis.x[None, :])[0][0]
    result.update(axis_r=float(axis.x[0]), axis_z=float(axis.x[1]),
                  axis_position_error=float(la.norm(axis.x-[eq.major_radius, 0])),
                  axis_flux_relative_error=float(abs(axis_value-eq.axis_flux)/eq.axis_flux),
                  axis_gradient_norm=float(axis_gradient))
    return result, dict(points=points, flux=psi, exact_flux=exact,
                        poloidal_field=bp, exact_poloidal_field=ebp,
                        boundary_points=edge, boundary_flux=edge_flux)


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    cases = {"polynomial": SolovevEquilibrium(), "logarithmic": SolovevEquilibrium(logarithmic=0.08)}
    domains = {name: SolovevFluxDomain(eq) for name, eq in cases.items()}
    grids = {name: validation_grid(domain) for name, domain in domains.items()}
    source_paths = [Path(__file__), Path(import_module("bspf_models.plasma.solovev").__file__), Path(import_module("bspf_models.plasma.grad_shafranov").__file__)]
    report = dict(settings=vars(args) | {"out": str(args.out)},
                  profiles={name: asdict(eq) for name, eq in cases.items()},
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
                  environment=dict(numpy=np.__version__, scipy=scipy.__version__,
                                   blas_threads=os.environ.get("OPENBLAS_NUM_THREADS"),
                                   omp_threads=os.environ.get("OMP_NUM_THREADS")), runs=[])
    for n in args.nodes:
        for name, eq in cases.items():
            domain = domains[name]
            plan = FixedBoundaryGSPlan(domain, major_radius=eq.major_radius, nodes=n,
                                       volume_order=args.volume_order, boundary_count=args.boundary_count,
                                       rcond=args.rcond)
            print("PLAN", n, name, json.dumps(plan.diagnostics), flush=True)
            solution = plan.solve_solovev(p_prime=eq.p_prime, ff_prime=eq.ff_prime, mu0=eq.mu0)
            # Warm solve and timing exclude user input sampling and output evaluation.
            source = eq.source(plan.points)
            times = []
            for _ in range(5):
                start = perf_counter()
                plan.solve(source, 0.0)
                times.append(perf_counter()-start)
            points, mask, raxis, zaxis = grids[name]
            error, arrays = metrics(solution, eq, points, 2*args.boundary_count)
            validation = None
            if n == max(args.nodes):
                validation = solution.validate(eq.source, volume_order=args.validation_order,
                                               boundary_count=2*args.boundary_count)
                qp, qw, qb, _, _ = plan.validation_cache[(args.validation_order, 2*args.boundary_count)]
                qpsi, qg, qh = evaluate_gs_factors(qb, solution.coefficients)
                qexact, qeg, qeh = eq.jets(qp)
                delta = qh[:, 0, 0]+qh[:, 1, 1]-qg[:, 0]/qp[:, 0]
                jp, je = -delta/(eq.mu0*qp[:, 0]), eq.source(qp)/(eq.mu0*qp[:, 0])
                validation.update(plasma_current=float(qw @ jp), exact_plasma_current=float(qw @ je),
                                  plasma_current_relative_error=float(abs(qw @ (jp-je))/abs(qw @ je)),
                                  flux_relative_l2=float(np.sqrt(qw @ (qpsi-qexact)**2/(qw @ qexact**2))),
                                  h2_relative=float(np.sqrt(
                                      qw @ ((qpsi-qexact)**2+np.sum((qg-qeg)**2, axis=1)+np.sum((qh-qeh)**2, axis=(1, 2))) /
                                      (qw @ (qexact**2+np.sum(qeg*qeg, axis=1)+np.sum(qeh*qeh, axis=(1, 2)))))))
            row = dict(nodes=n, case=name, plan=plan.diagnostics, solve=solution.diagnostics,
                       repeated_rhs_seconds=float(np.median(times)), errors=error, refined_validation=validation)
            report["runs"].append(row)
            np.savez(args.out/f"{name}_n{n}.npz", coefficients=solution.coefficients,
                     singular_values=plan.spectrum, mask=mask, raxis=raxis, zaxis=zaxis, **arrays)
            print("RESULT", json.dumps(row), flush=True)
            (args.out/"results.json").write_text(json.dumps(report, indent=2)+"\n")
            summarize(report, args.out)


def summarize(report, out):
    lines = ["# Fixed-boundary BSPF Grad–Shafranov: Solov'ev validation", "",
             "Exact smooth psi=0 boundaries, physical R>0, nonzero constant p' and FF'.",
             "Fields use BSPF; geometry is analytic. All PDE solves receive only source and zero boundary flux.",
             "Independent offset grids and boundary samples; discrete errors are not certified continuous bounds.", "",
             "| Case | N | Flux rel L2 | Poloidal B rel L2 | H2 rel | GS residual rel | Boundary flux max | Setup s | RHS ms |",
             "|---|---|---|---|---|---|---|---|---|"]
    for row in report["runs"]:
        e = row["errors"]
        lines.append(f"| {row['case']} | {row['nodes']} | {e['flux_relative_l2']:.3e} | {e['poloidal_field_relative_l2']:.3e} | {e['h2_relative']:.3e} | {e['gs_relative_residual']:.3e} | {e['boundary_flux_max']:.3e} | {row['plan']['setup_seconds']:.2f} | {row['repeated_rhs_seconds']*1000:.3f} |")
    lines += ["", "BSPF 1D line construction is cached across geometries at equal N; setup times therefore have different cache states.",
              "Reported RHS excludes output evaluation. Factor storage is not peak memory.", "",
              "## Refined independent quadrature", ""]
    for row in report["runs"]:
        if row["refined_validation"]:
            lines += [f"### {row['case']}, N={row['nodes']}", "", "```json", json.dumps(row["refined_validation"], indent=2), "```", ""]
    lines += ["Solov'ev profiles make this a linear GS problem; this benchmark does not validate general nonlinear profile iteration or free-boundary equilibria.", ""]
    (out/"summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, nargs="+", default=[17, 25, 33, 49])
    parser.add_argument("--volume-order", type=int, default=16)
    parser.add_argument("--validation-order", type=int, default=24)
    parser.add_argument("--boundary-count", type=int, default=512)
    parser.add_argument("--rcond", type=float, default=1e-13)
    parser.add_argument("--out", type=Path, default=Path("build/solovev_bspf"))
    run(parser.parse_args())
