"""Accuracy-gated pole selection before rational flow volume assembly.

The audit never changes rank cutoffs. Field gains are sampled operator norms,
not global continuum error bounds. Boundary probes span all retained flow trace
directions without dividing by their tiny singular values.
"""
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from .rational_stokes import RationalStokesExtension


class RationalPreprocessingError(RuntimeError):
    """No candidate met the requested accuracy/stability limits."""

    def __init__(self, report):
        self.report = report
        super().__init__("Rational preprocessing found no admissible pole configuration; "
                         "inspect exception.report['candidates'] for failed checks")


@jax.jit
def _products(rows, coefficients, right, singular, left):
    prediction = tuple(row @ coefficients for row in rows)
    # Keep the solve ordering; this influence matrix is ONLY used for an audit.
    influence = tuple(((row @ right)/singular) @ left for row in rows)
    gains = tuple(jnp.linalg.norm(a, axis=1) for a in influence)
    # u_y = v_x - omega: form this before taking the operator norm.
    uy_gain = jnp.linalg.norm(influence[5]-influence[3], axis=1)
    return prediction, gains, uy_gain


def _cpu_products(rows, coefficients, right, singular, left):
    prediction = tuple(row @ coefficients for row in rows)
    influence = tuple(((row @ right)/singular) @ left for row in rows)
    return prediction, tuple(np.linalg.norm(a, axis=1) for a in influence), np.linalg.norm(influence[5]-influence[3], axis=1)


def _probes(bounds, hole, samples, clustering):
    left, right, height = bounds
    # Midpoints differ from every candidate's collocation grid; include corners
    # and a uniform component to detect gaps between clustered samples.
    t = np.unique(np.r_[-1., 1., np.tanh(np.linspace(-clustering, clustering, 2*samples+1)[1::2]),
                        np.linspace(-1, 1, max(33, samples//4))])
    n = len(t)
    theta = 2*np.pi*(np.arange(samples)+.5)/samples
    a, b = hole.axes
    hz = complex(*hole.center)+a*np.cos(theta)+1j*b*np.sin(theta)
    walls = (left+1j*height*t, (left+right)/2+(right-left)*t/2+1j*height,
             (left+right)/2+(right-left)*t/2-1j*height, right+1j*height*t)
    xx, yy = np.meshgrid(np.linspace(left, right, 25)[1:-1], np.linspace(-height, height, 17)[1:-1])
    interior = np.column_stack((xx.ravel(), yy.ravel()))
    interior = interior[hole.level(interior)>1.]
    z = np.r_[*walls, hz, interior[:, 0]+1j*interior[:, 1]]
    tangent = np.column_stack((-a*np.sin(theta), b*np.cos(theta)))
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    return z, n, tangent


def audit_extension(extension, data, probes, *, device=None, batch_size=256):
    """Audit response accuracy and all retained singular directions.

    data(points) returns u,v,u_x,u_y,v_x for the hole data, each (N,K).
    Column zero is the lift; remaining columns are scaled flow trace probes.
    Gains map unit RMS perturbations in the two sampled hole-velocity components
    (restricted to zero flux) to individual physical field components.
    """
    z, side_count, tangent = probes
    count = len(tangent)
    fields = data(extension.hole_points)
    coefficients = extension.response(np.vstack(fields[:2]))
    n = len(extension.hole_points)
    theta = 2*np.pi*np.arange(n)/n
    a, b = extension.hole.axes
    # Normal times arc-length element for an ellipse. Projection removes the
    # incompatible flux mode, without removing valid velocity perturbations.
    flux = np.r_[b*np.cos(theta), a*np.sin(theta)]
    flux /= np.linalg.norm(flux)
    left = extension.left_hole-(extension.left_hole @ flux)[:, None]*flux
    left *= np.sqrt(2*n)
    factors = (coefficients, extension.right_scaled, extension.singular_values, left)
    product = _cpu_products
    if device is not None:
        factors = jax.device_put(factors, device)
        product = _products
    predictions = [[] for _ in range(6)]
    max_gains = np.zeros(7)
    for start in range(0, len(z), batch_size):
        batch = z[start:start+batch_size]
        size = len(batch)
        if size < batch_size:
            batch = np.pad(batch, (0, batch_size-size), mode='edge')
        rows = tuple(row[:, extension.columns] for row in extension.rows(batch))
        if device is not None:
            rows = jax.device_put(rows, device)
        pred, gains, uy = jax.device_get(product(rows, *factors))
        for values, v in zip(predictions, pred):
            values.append(v[:size])
        max_gains = np.maximum(max_gains, [np.max(g[:size]) for g in (*gains, uy)])
    u, v, pressure, omega, ux, vx = [np.concatenate(parts) for parts in predictions]
    uy = vx-omega
    h = extension.bounds[2]
    end_wall = 3*side_count
    outlet = slice(end_wall, 4*side_count)
    boundary = slice(4*side_count, 4*side_count+count)
    expected = data(np.column_stack((z[boundary].real, z[boundary].imag)))
    rownorm = lambda x: float(np.max(np.linalg.norm(x, axis=1)))
    velocity_error = max(rownorm(u[:end_wall]), rownorm(v[:end_wall]),
                         rownorm(u[boundary]-expected[0]), rownorm(v[boundary]-expected[1]))
    traction_error = h*max(rownorm(ux[outlet]-pressure[outlet]), rownorm(vx[outlet]))
    # Along inlet use y derivatives; along top/bottom use x derivatives.
    tangent_error = h*max(rownorm(uy[:side_count]), rownorm(ux[:side_count]),
                          rownorm(ux[side_count:end_wall]), rownorm(vx[side_count:end_wall]))
    tx, ty = tangent.T
    tangent_error = max(tangent_error, h*rownorm((ux[boundary]-expected[2])*tx[:, None]+(uy[boundary]-expected[3])*ty[:, None]),
                        h*rownorm((vx[boundary]-expected[4])*tx[:, None]-(ux[boundary]-expected[2])*ty[:, None]))
    detail = {}
    for name, section in [('inlet', slice(0, side_count)), ('top', slice(side_count, 2*side_count)),
                          ('bottom', slice(2*side_count, 3*side_count))]:
        detail[name] = max(rownorm(u[section]), rownorm(v[section]))
    detail['hole'] = max(rownorm(u[boundary]-expected[0]), rownorm(v[boundary]-expected[1]))
    metrics = dict(boundary_errors=detail, velocity_error=velocity_error, traction_error=traction_error,
                   tangent_error=tangent_error, velocity_gain=float(max(max_gains[:2])),
                   gradient_gain=float(h*max(max_gains[4], max_gains[5], max_gains[6])),
                   vorticity_gain=float(h*max_gains[3]), pressure_gain=float(h*max_gains[2]),
                   probe_points=len(z), trace_directions=coefficients.shape[1]-1)
    return metrics


def prepare_extension(baseline, data, *, device=None, options=None):
    """Sweep poles/sampling once and return the already-factored selected fit.

    options: candidates=[(pole_count, nearest_distance/half_height, samples)],
    velocity_tolerance, traction_tolerance, tangent_tolerance, gain_limit,
    validation_samples. No admissible fit raises RationalPreprocessingError.
    """
    started = perf_counter()
    options = dict(options or {})
    count = baseline.info['corner_poles']
    samples = baseline.info['samples_per_side']
    supplied = options.pop('candidates', None)
    candidates = supplied if supplied is not None else [
        (max(4, count//2), 1.e-4, samples),
        (max(4, 3*count//4), 1.e-4, samples),
        (count, 1.e-4, samples),
        (max(4, 3*count//4), 1.e-6, samples),
        (max(4, count//2), 1.e-4, (3*samples+1)//2),
        (max(4, 3*count//4), 1.e-4, (3*samples+1)//2),
    ]
    limits = dict(velocity_error=options.pop('velocity_tolerance', 1.e-9),
                  traction_error=options.pop('traction_tolerance', 1.e-9),
                  tangent_error=options.pop('tangent_tolerance', 1.e-7))
    gain_limit = options.pop('gain_limit', np.inf)
    validation_samples = options.pop('validation_samples', max(257, samples+1))
    if options:
        raise ValueError(f"Unknown rational preprocessing options: {sorted(options)}")
    if any(not np.isfinite(v) or v <= 0 for v in limits.values()) or gain_limit <= 0 or np.isnan(gain_limit):
        raise ValueError('Audit tolerances and gain_limit must be positive')
    if not isinstance(validation_samples, int) or validation_samples < 32:
        raise ValueError('validation_samples must be an integer >=32')
    baseline_key = (count, baseline.info['min_pole_distance'], samples)
    keys = list(dict.fromkeys([baseline_key]+[tuple(c) for c in candidates]))
    probes = _probes(baseline.bounds, baseline.hole, validation_samples, 2*np.sqrt(count)+1)
    report = dict(candidates=[], limits=limits, gain_limit=None if np.isinf(gain_limit) else float(gain_limit),
                  gain_norm='unit RMS zero-flux hole data to maximum sampled scalar field',
                  baseline=0, selected=None)
    best = None
    for index, (poles, distance, n) in enumerate(keys):
        t = perf_counter()
        ext = baseline if index == 0 else RationalStokesExtension(
            baseline.bounds, baseline.hole, degree=baseline.info['degree'],
            laurent=baseline.info['laurent'], rcond=baseline.info['rcond'],
            corner_poles=poles, min_pole_distance=distance, samples=n,
            basis_construction=baseline.info['basis_construction'], assembly_device=device)
        audit = audit_extension(ext, data, probes, device=device)
        failed = [name for name, limit in limits.items() if not np.isfinite(audit[name]) or audit[name]>limit]
        gain = max(audit['gradient_gain'], audit['vorticity_gain'])
        if not np.isfinite(gain) or gain > gain_limit:
            failed.append('gain_limit')
        record = dict(corner_poles=poles, min_pole_distance=distance, samples=n,
                      rank=ext.info['rank'], actual_min_pole_distance=ext.info['actual_min_pole_distance'],
                      **audit, accepted=not failed, failed=failed, seconds=perf_counter()-t)
        report['candidates'].append(record)
        if not failed and (best is None or gain < best[0]):
            best = (gain, index, ext)
    report['seconds'] = perf_counter()-started
    if best is None:
        raise RationalPreprocessingError(report)
    report['selected'] = best[1]
    best[2].info['preprocessing'] = report
    return best[2]
