"""Reproducible 1z2v magnetic-mirror validation and convergence report.

python examples/pde/drift_kinetic_mirror.py
The default evolves log(f), retaining its exponential continuous representation.
Use --quick for one grid only or --linear to reproduce the old linear-f run.
"""

import bspf_models.kinetic.drift_kinetic as bspf_drift_kinetic
import pybspf.plans as bspf_plans
import argparse
import json
from pathlib import Path

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from pybspf.fast_axis import axis_values
from pybspf.fast_axis import axis_adjoint
from pybspf.fast_axis import sample_aligned_knots


def log_exact(t, z, v, mu):
    omega = jnp.sqrt(mu)
    c, s = jnp.cos(omega*t), jnp.sin(omega*t)
    z0 = z*c-v*t*jnp.sinc(omega*t/jnp.pi)
    v0 = v*c+omega*z*s
    return -4*z0*z0-6*(v0-.8)**2-12*(mu-1.)**2


def exact(t, z, v, mu):
    return jnp.exp(log_exact(t, z, v, mu))


def modulated_log_exact(t, z, v, mu):
    omega = jnp.sqrt(mu)
    z0 = z*jnp.cos(omega*t)-v*t*jnp.sinc(omega*t/jnp.pi)
    v0 = v*jnp.cos(omega*t)+omega*z*jnp.sin(omega*t)
    return log_exact(t, z, v, mu)+jnp.log1p(.2*jnp.cos(1.3*z0+.7*v0))


def run(n, n_mu=12, substeps=40, quadrature_order=12, *, representation="log", profile="gaussian", backend="matrix_free"):
    if representation not in ('log', 'linear') or profile not in ('gaussian', 'modulated'):
        raise ValueError('unsupported representation or profile')
    z = jnp.linspace(-4., 4., n)
    v = jnp.linspace(-3., 3., n)
    def knots(x):
        return sample_aligned_knots(x,degree=7,n_basis=min(21,max(12,n//3)))
    zp = bspf_plans.plan_1d(z, degree=7, knots=knots(z), boundary_points=9)
    vp = bspf_plans.plan_1d(v, degree=7, knots=knots(v), boundary_points=9)
    p = bspf_drift_kinetic.plan_drift_kinetic(zp, vp, magnetic_field=lambda z: 1+z*z/2,
        magnetic_gradient=lambda z: z, mu_max=2., n_mu=n_mu, quadrature_order=quadrature_order, backend=backend)
    def z_values(f):
        return axis_values(p.z_axis,f) if backend == 'matrix_free' else jnp.tensordot(p.z_values,f,axes=(1,0))
    def v_values(f):
        return axis_values(p.v_axis,f) if backend == 'matrix_free' else jnp.tensordot(p.v_values,f,axes=(1,0))
    def evaluate_tensor(f):
        return jnp.moveaxis(v_values(jnp.moveaxis(z_values(f),1,0)),0,1)
    times = jnp.linspace(0., 4., 81)
    log_reference = log_exact if profile == 'gaussian' else modulated_log_exact
    initial_log = log_reference(0., z[:, None, None], v[None, :, None], p.mu[None, None, :])
    reference_log = log_reference(times[:, None, None, None], z[None, :, None, None],
                                  v[None, None, :, None], p.mu[None, None, None, :])
    reference = jnp.exp(reference_log)
    if representation == 'log':
        logs, transfers = bspf_drift_kinetic.integrate_log_drift_kinetic(p, initial_log, times,
            log_inflow=log_reference, substeps=substeps)
        history = jnp.exp(logs)
        moments, _, qminimum = bspf_drift_kinetic.log_drift_kinetic_diagnostics(p, logs)
        ref_moments, _, _ = bspf_drift_kinetic.log_drift_kinetic_diagnostics(p, reference_log)
        assert bool(jnp.all(jnp.isfinite(logs)))
        assert float(jnp.min(history)) >= 0 and float(jnp.min(qminimum)) >= 0
    else:
        history, transfers = bspf_drift_kinetic.integrate_drift_kinetic(p, jnp.exp(initial_log), times,
            inflow=lambda t,z,v,mu: jnp.exp(log_reference(t,z,v,mu)), substeps=substeps)
        moments = bspf_drift_kinetic.drift_kinetic_moments(p, history)
        ref_moments = bspf_drift_kinetic.drift_kinetic_moments(p, reference)
        qminimum = jax.lax.map(lambda f: jnp.min(evaluate_tensor(f)), history)
    nh = moments[:, jnp.array([0, 3])]
    balance = nh-nh[0]-transfers.sum(axis=-1)
    rel_balance = jnp.max(jnp.abs(balance), axis=0)/jnp.abs(nh[0])
    # Mirror orbit of the centroid in the mu slice nearest one. The chosen
    # Gaussian is localized enough that finite-domain centroid truncation is tiny.
    k = int(jnp.argmin(jnp.abs(p.mu-1)))
    zq, vq = p.transport.z_points, p.transport.v_points
    if backend == 'matrix_free':
        zw, vw = axis_adjoint(p.z_axis,p.z_weights*zq),axis_adjoint(p.v_axis,p.v_weights*vq)
        onez, onev = p.number_z,p.number_v
    else:
        zw = (p.z_weights*zq)@p.z_values
        vw = (p.v_weights*vq)@p.v_values
        onez, onev = p.z_weights@p.z_values, p.v_weights@p.v_values
    if representation == 'log':
        def centroid(g):
            values = jnp.exp(v_values(z_values(g).T).T)
            weighted = p.z_weights[:, None]*p.v_weights[None, :]*values
            norm = jnp.sum(weighted)
            return jnp.sum(weighted*zq[:, None])/norm, jnp.sum(weighted*vq[None, :])/norm
        mean_z, mean_v = jax.lax.map(centroid, logs[..., k])
    else:
        slice_f = history[..., k]
        norm = jnp.einsum('tij,i,j->t', slice_f, onez, onev)
        mean_z = jnp.einsum('tij,i,j->t', slice_f, zw, onev)/norm
        mean_v = jnp.einsum('tij,i,j->t', slice_f, onez, vw)/norm
    omega = jnp.sqrt(p.mu[k])
    mean_z0, mean_v0 = 0., .8
    if profile == 'modulated':
        attenuation = jnp.exp(-.5*(1.3**2/8+.7**2/12))
        denominator = 1+.2*attenuation*jnp.cos(.7*.8)
        mean_z0 = -.2*1.3/8*attenuation*jnp.sin(.7*.8)/denominator
        mean_v0 = .8-.2*.7/12*attenuation*jnp.sin(.7*.8)/denominator
    orbit_z = mean_z0*jnp.cos(omega*times)+mean_v0/omega*jnp.sin(omega*times)
    orbit_v = mean_v0*jnp.cos(omega*times)-omega*mean_z0*jnp.sin(omega*times)
    metrics = dict(knots="sample_aligned", backend=backend, representation=representation, profile=profile, n=n, n_mu=n_mu, dt=float(times[1]/substeps), quadrature_order=quadrature_order,
        max_distribution_error=float(jnp.max(jnp.abs(history-reference))),
        min_distribution=float(jnp.min(history)),
        min_quadrature_distribution=float(jnp.min(qminimum)),
        relative_particle_balance=float(rel_balance[0]), relative_energy_balance=float(rel_balance[1]),
        relative_particle_reference_error=float(jnp.max(jnp.abs(moments[:, 0]-ref_moments[:, 0]))/nh[0, 0]),
        relative_energy_reference_error=float(jnp.max(jnp.abs(moments[:, 3]-ref_moments[:, 3]))/nh[0, 1]),
        mirror_mu=float(p.mu[k]), analytic_turn_time=float(jnp.pi/(2*omega)),
        max_centroid_z_error=float(jnp.max(jnp.abs(mean_z-orbit_z))),
        max_centroid_v_error=float(jnp.max(jnp.abs(mean_v-orbit_v))),
        final_centroid_v=float(mean_v[-1]),
        parallel_energy_change=float(moments[-1, 1]-moments[0, 1]),
        perpendicular_energy_change=float(moments[-1, 2]-moments[0, 2]),
        inward_particle_transfer_by_face=np.asarray(transfers[-1, 0]).tolist(),
        inward_energy_transfer_by_face=np.asarray(transfers[-1, 1]).tolist())
    if profile != 'gaussian':
        # This convergence check targets the distribution and physical balances.
        metrics.pop('max_centroid_z_error')
        metrics.pop('max_centroid_v_error')
    if representation == 'log':
        metrics['minimum_log'] = float(jnp.min(logs))
        metrics['zero_nodal_samples'] = int(jnp.count_nonzero(history == 0))
    print(json.dumps(metrics), flush=True)
    arrays = dict(times=np.asarray(times), z=np.asarray(z), v=np.asarray(v), mu=np.asarray(p.mu),
        history=np.asarray(history), moments=np.asarray(moments), balance=np.asarray(balance),
        transfers=np.asarray(transfers), mean_z=np.asarray(mean_z), mean_v=np.asarray(mean_v),
        orbit_z=np.asarray(orbit_z), orbit_v=np.asarray(orbit_v))
    if representation == 'log':
        arrays['log_history'] = np.asarray(logs)
    # Avoid retaining compiled executables for all refinement grids.
    jax.clear_caches()
    return metrics, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('build/drift_kinetic_mirror'))
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--dense', action='store_true', help='Use original dense axes for comparison')
    parser.add_argument('--linear', action='store_true', help='Reproduce the original linear-f method')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    options = dict(backend='dense' if args.dense else 'matrix_free')
    if args.linear:
        fine, arrays = run(65, representation='linear', **options)
        report = {'linear': fine}
    elif args.quick:
        fine, arrays = run(49, **options)
        report = {'fine': fine}
    else:
        coarse, _ = run(49, **options)
        medium, _ = run(57, **options)
        fine, arrays = run(65, **options)
        half_step, half = run(65, substeps=80, **options)
        mu_refined, mu_arrays = run(65, n_mu=24, **options)
        quadrature, gauss = run(65, quadrature_order=16, **options)
        report = dict(coarse=coarse, medium=medium, fine=fine, half_step=half_step,
                      mu_refined=mu_refined, quadrature=quadrature)
        report['step_difference'] = float(np.max(np.abs(half['history']-arrays['history'])))
        report['quadrature_difference'] = float(np.max(np.abs(gauss['history']-arrays['history'])))
        report['mu_relative_moment_difference'] = float(np.max(np.abs(
            mu_arrays['moments']-arrays['moments'])/np.abs(arrays['moments'][0])))
        # A Gaussian has a quadratic logarithm already resolved on all grids;
        # use a non-polynomial modulated profile to test spatial convergence.
        mod_coarse, _ = run(49, profile='modulated', **options)
        mod_fine, _ = run(65, profile='modulated', **options)
        report['modulated_coarse'], report['modulated_fine'] = mod_coarse, mod_fine
        assert mod_fine['max_distribution_error'] < mod_coarse['max_distribution_error']
        assert mod_fine['relative_particle_balance'] < 1e-7
        assert mod_fine['relative_energy_balance'] < 1e-7
        for result in (coarse, medium, fine, half_step, mu_refined, quadrature, mod_coarse, mod_fine):
            assert result['min_distribution'] >= 0
            assert result['min_quadrature_distribution'] >= 0
        assert coarse['max_distribution_error'] < 1e-7
        assert fine['max_distribution_error'] < 1e-5
        assert fine['relative_particle_balance'] < 1e-7
        assert fine['relative_energy_balance'] < 1e-7
        assert fine['max_centroid_z_error'] < 1e-5
        assert fine['max_centroid_v_error'] < 1e-5
        assert fine['final_centroid_v'] < 0
        assert report['step_difference'] < 1e-7
        assert report['quadrature_difference'] < 1e-7
        assert report['mu_relative_moment_difference'] < 1e-4
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    np.savez_compressed(args.out/'solution.npz', **arrays)
    render(arrays, args.out)
    print(f'Report and figures: {args.out.resolve()}')


def render(arrays, out):
    """Render stored numerical states without recomputing the simulation."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t, moments = arrays['times'], arrays['moments']
    fig, ax = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    fig.set_constrained_layout_pads(h_pad=.18, w_pad=.06, hspace=.12, wspace=.05)
    ax[0, 0].plot(t, arrays['mean_z'], label='BSPF mean z')
    ax[0, 0].plot(t, arrays['orbit_z'], '--', label='Analytic z')
    ax[0, 0].plot(t, arrays['mean_v'], label='BSPF mean v_parallel')
    ax[0, 0].plot(t, arrays['orbit_v'], '--', label='Analytic v_parallel')
    ax[0, 0].set(title='Magnetic mirror reflection', xlabel='Time'); ax[0, 0].legend()
    for k, label in [(1, 'Parallel'), (2, 'Perpendicular'), (3, 'Total')]:
        ax[0, 1].plot(t, moments[:, k], label=label)
    ax[0, 1].set(title='Energy exchange', xlabel='Time'); ax[0, 1].legend()
    for k, label in enumerate(['Particle balance', 'Energy balance']):
        ax[1, 0].plot(t, arrays['balance'][:, k]/moments[0, (0, 3)[k]], label=label)
    ax[1, 0].set(title='Relative balance residual including all faces', xlabel='Time'); ax[1, 0].legend()
    k = int(np.argmin(abs(arrays['mu']-1)))
    mesh = ax[1, 1].pcolormesh(arrays['z'], arrays['v'], arrays['history'][-1, :, :, k].T,
                              shading='auto', cmap='viridis')
    ax[1, 1].set(title=f"Final f, mu={arrays['mu'][k]:.3f}", xlabel='z', ylabel='v_parallel')
    fig.colorbar(mesh, ax=ax[1, 1])
    fig.savefig(out/'validation.png', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
