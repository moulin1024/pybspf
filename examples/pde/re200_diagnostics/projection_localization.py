"""Locate remote curl leakage in the BSPF GPU velocity projection.

All projected forces and mass solves reside on GPU. Field operators use the
original corrected BSPF space. This is a frozen-operator diagnostic, not a
change to the Navier--Stokes equations or an output filter.
"""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jl
import numpy as np

from bspf_jax.immersed_flow import ImmersedFlowPlan
from bspf_jax.immersed_flow_gpu import _explicit


def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a)**2)))


def metrics(a):
    return dict(rms=rms(a), maximum=float(np.max(np.abs(a))),
                d4y=rms(np.diff(a, n=4, axis=0)), d4x=rms(np.diff(a, n=4, axis=1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quadrature', type=float, default=2.5)
    parser.add_argument('--nx', type=int, default=73)
    parser.add_argument('--ny', type=int, default=33)
    parser.add_argument('--out', type=Path, default=Path('build/immersed_flow/re200_ripple_study/projection_localization'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jax.config.update('jax_enable_x64', True)
    device = jax.devices('gpu')[0]
    p = ImmersedFlowPlan(assembly_device=device, basis_precision='float64',
                        nx=args.nx, ny=args.ny, reynolds=200, wall_method='rational',
                        rational_preprocessing=False, quadrature_factor=args.quadrature)
    step = p.stepper(.01, device=device)
    d = step.data
    print('SETUP', p.setup_seconds, p.dofs, flush=True)
    transform = jax.device_put(p.transform, device)
    @jax.jit
    def solve(d, rhs):
        return jl.cho_solve((d['mass_factor'], True), rhs)
    @jax.jit
    def load(d, force):
        return d['ops'][0].T @ (d['weights']*force[:, 0])+d['ops'][1].T @ (d['weights']*force[:, 1])
    def field_ops(points):
        raw = p.operators(points, device_output=True)
        return tuple(op @ transform for op in (raw[1], raw[2], raw[5]-raw[4]))
    x, y = np.linspace(-.85, -.45, 61), np.linspace(-.8, .8, 161)
    xx, yy = np.meshgrid(x, y)
    probe = np.column_stack((xx.ravel(), yy.ravel()))
    probe_u, probe_v, probe_curl = field_ops(probe)
    # Independent, offset wall points are separate from rational fitting nodes.
    hole, _ = p.arc.sample(256, offset=.371)
    wx, wy = np.linspace(*p.bounds[:2], 197), np.linspace(-1, 1, 157)
    wall = np.vstack((hole, np.column_stack((wx, -np.ones_like(wx))),
                      np.column_stack((wx, np.ones_like(wx))),
                      np.column_stack((np.full_like(wy, p.bounds[0]), wy))))
    wall_u, wall_v, _ = field_ops(wall)
    points = jax.device_put(p.points, device)
    @jax.jit
    def bump(z):
        return jnp.maximum(1-((z[0]-.2)/.2)**2, 0.)**4*jnp.maximum(1-((z[1]-.45)/.1)**2, 0.)**4
    @jax.jit
    def compact_forces(points):
        b = jax.vmap(bump)(points)
        grad = jax.vmap(jax.grad(bump))(points)
        return (jnp.stack((jnp.zeros_like(b), b), axis=-1),
                jnp.stack((grad[:, 1], -grad[:, 0]), axis=-1), grad)
    force_cases = dict(zip(('vertical_bump', 'divergence_free_bump', 'gradient_bump'), compact_forces(points)))
    outputs = dict(x=x, y=y)
    compact = []
    for name, force in force_cases.items():
        force = force/jnp.sqrt(jnp.sum(d['weights']*jnp.sum(force**2, axis=-1)))
        with jax.transfer_guard('disallow'):
            rhs = load(d, force)
            a = solve(d, rhs)
            curl = probe_curl @ a
            projected = jnp.stack((d['ops'][0] @ a, d['ops'][1] @ a), axis=-1)
            residual = jnp.linalg.norm(d['mass'] @ a-rhs)/jnp.linalg.norm(rhs)
            boundary = jnp.max(jnp.hypot(wall_u @ a, wall_v @ a))
            norm = jnp.sqrt(a @ d['mass'] @ a)
            error = jnp.sqrt(jnp.sum(d['weights']*jnp.sum((projected-force)**2, axis=-1)))
            jax.block_until_ready((curl, residual, boundary, norm, error))
        omega = np.asarray(curl).reshape(xx.shape)
        outputs[name] = omega
        record = dict(force=name, force_l2=1., support=[0., .4, .35, .55],
                      exact_force_and_curl_zero_in_probe=True, projected_curl=metrics(omega),
                      projected_velocity_l2=float(norm), mass_relative_residual=float(residual),
                      homogeneous_wall_error=float(boundary))
        if name == 'divergence_free_bump':
            record['relative_velocity_projection_error'] = float(error)
        compact.append(record)
        print('COMPACT', json.dumps(record), flush=True)

    # Nested smooth masks form a partition; near-body mask vanishes in probe.
    @jax.jit
    def smooth(s):
        s = jnp.clip(s, 0., 1.)
        return s**3*(10+s*(-15+6*s))
    center, axes = jax.device_put((np.asarray(p.hole.center), np.asarray(p.hole.axes)), device)
    q = jnp.sqrt(jnp.sum(((points-center)/axes)**2, axis=-1))
    near = smooth(2-q)
    inlet = (1-near)*(1-smooth((points[:, 0]+.45)/.35))
    buffer = (1-near-inlet)*smooth((points[:, 0]-p.buffer_start)/.5)
    remaining = 1-near-inlet-buffer
    masks = dict(near_body=near, inlet_region=inlet, buffer_region=buffer, remaining=remaining)
    mask_error = float(jnp.max(jnp.abs(sum(masks.values())-1)))
    @jax.jit
    def nonlinear(d, a):
        u, v, ux, uy, vx = [o @ a+b for o, b in zip(d['ops'], d['lift'])]
        return -jnp.stack((u*ux+v*uy, u*vx-v*ux), axis=-1)
    states = [('stokes', step.initial_state)]
    if (args.nx, args.ny) == (73, 33):
        saved = np.load('build/immersed_flow/re200_gpu_dt001/fields.npz')['coefficients']
        coeff = jax.device_put(saved-p.lift_coefficients, device)
        # A GPU least-squares solve also supports an energy-truncated space.
        recovered = jnp.linalg.lstsq(transform, coeff, rcond=None)[0]
        reconstruction_error = float(jnp.linalg.norm(transform @ recovered-coeff)/jnp.linalg.norm(coeff))
        states.append(('t20', recovered))
    else:
        reconstruction_error = None
    actual = []
    for name, a in states:
        force = nonlinear(d, a)
        bulk = load(d, force)
        # Retain the exact production outlet backflow load, independent of the partition.
        convection = _explicit(d, a)+d['linear_lift']
        loads = {key: load(d, force*mask[:, None]) for key, mask in masks.items()}
        loads['outlet_backflow'] = convection-bulk
        loads['total_convection'] = convection
        loads['viscous_sponge'] = -d['linear'] @ a-d['linear_lift']
        loads['total_acceleration'] = convection+loads['viscous_sponge']
        keys = list(loads)
        rhs = jnp.stack([loads[key] for key in keys], axis=1)
        with jax.transfer_guard('disallow'):
            rates = solve(d, rhs)
            omega = probe_curl @ rates
            jax.block_until_ready(omega)
        responses = {key: np.asarray(omega[:, i]).reshape(xx.shape) for i, key in enumerate(keys)}
        record = dict(state=name, metrics={key: metrics(value) for key, value in responses.items()})
        total = np.diff(responses['total_convection'], n=4, axis=0).ravel()
        record['d4y_correlations_with_total_convection'] = {}
        for key in masks:
            value = np.diff(responses[key], n=4, axis=0).ravel()
            record['d4y_correlations_with_total_convection'][key] = dict(
                norm_ratio=rms(value)/rms(total), correlation=float(np.corrcoef(value, total)[0, 1]))
        partition = sum(loads[key] for key in masks)
        record['bulk_partition_relative_error'] = float(jnp.linalg.norm(partition-bulk)/jnp.linalg.norm(bulk))
        outputs.update({name+'_'+key: value for key, value in responses.items()})
        actual.append(record)
        print('ACTUAL', json.dumps(record), flush=True)
    report = dict(device=str(device), nx=args.nx, ny=args.ny, quadrature_factor=args.quadrature,
                  setup_seconds=p.setup_seconds, dofs=p.dofs, probe_bounds=[-.85, -.45, -.8, .8],
                  compact=compact, actual=actual, mask_partition_max_error=mask_error,
                  saved_coefficient_reconstruction_relative_error=reconstruction_error)
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    np.savez_compressed(args.out/'fields.npz', **outputs)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes_plot = plt.subplots(2, 3, figsize=(12, 7), layout='constrained')
    keys = ['vertical_bump', 'divergence_free_bump', 'gradient_bump',
            'stokes_near_body', 'stokes_inlet_region', 'stokes_total_convection']
    for ax, key in zip(axes_plot.flat, keys):
        values = outputs[key]
        limit = max(float(np.max(np.abs(values))), 1e-16)
        im = ax.pcolormesh(x, y, values, cmap='RdBu_r', vmin=-limit, vmax=limit, shading='auto')
        ax.set(title=key, xlabel='x', ylabel='y')
        fig.colorbar(im, ax=ax)
    fig.savefig(args.out/'localization.png', dpi=150)


if __name__ == '__main__':
    main()
