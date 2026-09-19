"""GPU mapped spline Navier--Stokes: unsteady MMS or Re=200 obstacle flow."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from bspf_jax.immersed_poisson import EllipticHole
from bspf_jax.mapped_navier_stokes import MappedNavierStokesPlan, channel_streamfunction


def manufactured(bounds=(-1., 3., -1., 1.), hole=None, viscosity=.02):
    """Smooth no-slip exact velocity and continuous NS forcing (pressure zero)."""
    hole = hole or EllipticHole()
    left, right, bottom, top = bounds
    cx, cy = hole.center
    a, b = hole.axes
    scale = (right-left)**2*(top-bottom)**2
    def stream(z):
        x, y = z
        q = ((x-cx)/a)**2+((y-cy)/b)**2-1
        return .05*(((x-left)*(right-x)*(y-bottom)*(top-y)*q)/scale)**2
    def velocity(z):
        g = jax.grad(stream)(z)
        return jnp.array((g[1], -g[0]))
    def exact(z, t):
        return jnp.exp(-t)*velocity(z)
    def force(z, t):
        u = velocity(z)
        grad = jax.jacfwd(velocity)(z)
        lap = jnp.trace(jax.jacfwd(jax.jacfwd(velocity))(z), axis1=1, axis2=2)
        alpha = jnp.exp(-t)
        return -alpha*u+alpha**2*(grad @ u)-viscosity*alpha*lap
    return stream, velocity, exact, force


def independent_error(plan, state, exact, time):
    q, w = np.polynomial.legendre.leggauss(plan.degree+4)
    def axis(n):
        return ((np.arange(n)[:, None]+(q+1)/2)/n).ravel(), np.tile(w/(2*n), n)
    r, wr = axis(plan.elements[0]); t, wt = axis(plan.elements[1])
    result = plan.evaluate(state, r, t)
    points = result['points']
    values = jax.jit(jax.vmap(exact, in_axes=(0, None)))(points.reshape(-1, 2), time).reshape(points.shape)
    weights = result['determinant']*jax.device_put(wr[:, None]*wt[None, :], plan.device)
    error = result['velocity']-values
    l2 = jnp.sqrt(jnp.sum(weights*jnp.sum(error**2, axis=-1)))
    norm = jnp.sqrt(jnp.sum(weights*jnp.sum(values**2, axis=-1)))
    return dict(velocity_l2=float(l2), velocity_relative_l2=float(l2/norm),
                velocity_max_sample_error=float(jnp.max(jnp.linalg.norm(error, axis=-1))))


def host_stats(plan, state):
    return {k: np.asarray(v).tolist() for k, v in jax.device_get(plan.diagnostics(state)).items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case', choices=('mms', 'channel'), default='mms')
    ap.add_argument('--elements', nargs='+', type=int, default=[4, 8, 12])
    ap.add_argument('--degree', type=int, default=3)
    ap.add_argument('--boundary', choices=('strong', 'nitsche'), default='strong')
    ap.add_argument('--time', type=float)
    ap.add_argument('--dt', type=float)
    ap.add_argument('--reynolds', type=float, default=200.)
    ap.add_argument('--out', type=Path, default=Path('build/mapped_spline_flow'))
    args = ap.parse_args()
    if args.case == 'channel' and len(args.elements) != 1:
        ap.error('Select one --elements resolution for the channel run')
    end = args.time if args.time is not None else (.1 if args.case == 'mms' else 20.)
    dt = args.dt if args.dt is not None else (.002 if args.case == 'mms' else .005)
    if end <= 0 or dt <= 0 or args.reynolds <= 0:
        ap.error('time, dt, and Reynolds number must be positive')
    steps = int(np.ceil(end/dt)); dt = end/steps
    args.out.mkdir(parents=True, exist_ok=True)
    jax.config.update('jax_enable_x64', True)
    device = jax.devices('gpu')[0]
    print('DEVICE', device, flush=True)
    bounds, hole = (-1., 3., -1., 1.), EllipticHole()
    nu = .02 if args.case == 'mms' else (2/3)*(2*hole.axes[1])/args.reynolds
    lift = None if args.case == 'mms' else channel_streamfunction(bounds, hole)
    records = []
    for n in args.elements:
        p = MappedNavierStokesPlan(elements=(n, n), degree=args.degree, device=device,
                                  viscosity=nu, dt=dt, lift=lift, boundary=args.boundary)
        if args.case == 'mms':
            _, velocity, exact, forcing = manufactured(viscosity=nu)
            initial = p.project(velocity)
            initial_errors = independent_error(p, initial, exact, 0.)
        else:
            initial, forcing, initial_errors = p.stokes_state, None, {}
        print('SETUP', json.dumps(dict(elements=n, dofs=p.dofs, seconds=p.setup_seconds,
                                       timings=p.setup_timings, initial=host_stats(p, initial))), flush=True)
        # Compile the same batch shape once, without advancing the saved state.
        batch = min(steps, 50)
        start = perf_counter()
        jax.block_until_ready(p.advance(initial, steps=batch, forcing=forcing))
        compile_and_batch = perf_counter()-start
        state, done, history = initial, 0, [dict(time=0., **host_stats(p, initial))]
        start = perf_counter()
        while done < steps:
            count = min(batch, steps-done)
            state = p.advance(state, time=done*dt, steps=count, forcing=forcing)
            jax.block_until_ready(state)
            done += count
            stats = dict(time=done*dt, **host_stats(p, state))
            if not np.isfinite(stats['kinetic_energy']):
                raise RuntimeError(f'Non-finite solution at t={done*dt}; reduce dt or refine the mesh')
            history.append(stats)
            if args.case == 'channel':
                print('STEP', json.dumps(stats), flush=True)
        elapsed = perf_counter()-start
        record = dict(elements=n, degree=p.degree, boundary=p.boundary, dofs=p.dofs, viscosity=nu, dt=dt, steps=steps,
                      setup_seconds=p.setup_seconds, setup_timings=p.setup_timings,
                      compile_and_batch_seconds=compile_and_batch, evolution_seconds=elapsed,
                      milliseconds_per_step=1000*elapsed/steps,
                      initial_errors=initial_errors, final=history[-1], history=history)
        if args.case == 'mms':
            record.update(independent_error(p, state, exact, end))
            if records:
                record['velocity_l2_rate'] = float(np.log(records[-1]['velocity_l2']/record['velocity_l2'])/np.log(n/records[-1]['elements']))
        records.append(record)
        print('RESULT', json.dumps({k: v for k, v in record.items() if k != 'history'}), flush=True)
        (args.out/'report.json').write_text(json.dumps(dict(case=args.case, device=str(device),
            gpu=device.device_kind, reynolds=args.reynolds if args.case == 'channel' else None,
            end_time=end, cases=records), indent=2)+'\n')
    result = jax.device_get(p.evaluate(state, np.linspace(0, 1, 65), np.linspace(0, 1, 97)))
    np.savez_compressed(args.out/'solution.npz', **result, coefficients=jax.device_get(state))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    for ax, field, title, cmap in zip(axes, (np.linalg.norm(result['velocity'], axis=-1), result['vorticity']),
            ('Velocity magnitude', 'Vorticity'), ('viridis', 'RdBu_r')):
        limit = max(abs(float(field.min())), abs(float(field.max())))
        lo, hi = (0., limit) if cmap == 'viridis' else (-limit, limit)
        for k in range(4):
            xy = result['points'][k]
            im = ax.pcolormesh(xy[..., 0], xy[..., 1], field[k], shading='gouraud', vmin=lo, vmax=hi, cmap=cmap)
            ax.plot(xy[:, 0, 0], xy[:, 0, 1], 'k-', lw=.4, alpha=.3)
        ax.set(title=title, aspect='equal', xlabel='x', ylabel='y')
        fig.colorbar(im, ax=ax)
    fig.suptitle(f'Mapped spline NS: {args.case}, t={end:g}, {p.dofs} unknowns')
    fig.savefig(args.out/'solution.png', dpi=160)


if __name__ == '__main__':
    main()
