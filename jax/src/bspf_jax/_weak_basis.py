"""Optional MPFR assembly of BSPF cardinal values, returned in float64.

Only setup uses high precision, to avoid cancelling huge spline/Fourier
components of endpoint cardinal functions. Runtime matrices remain float64.
Requires gmpy2 only during host setup.
"""

import numpy as np


def _splines(knots, points, degree, mp, second=False):
    k = np.array([mp(float(v)) for v in knots], dtype=object)
    t = np.asarray(points, dtype=object)
    b = np.array(((t[:, None] >= k[:-1]) & (t[:, None] < k[1:])), dtype=object)
    # Evaluate the right endpoint by its left limit, including derivatives.
    last_interval = np.flatnonzero(k[1:] > k[:-1])[-1]
    b[t == k[-1], last_interval] = mp(1)
    lower_two = None
    for p in range(1, degree + 1):
        if p == degree - 1:
            lower_two = b
        previous = b
        size = len(k) - p - 1
        b = np.full((len(t), size), mp(0), dtype=object)
        for i in range(size):
            if k[i + p] != k[i]:
                b[:, i] += (t - k[i]) / (k[i + p] - k[i]) * previous[:, i]
            if k[i + p + 1] != k[i + 1]:
                b[:, i] += (
                    (k[i + p + 1] - t) / (k[i + p + 1] - k[i + 1]) * previous[:, i + 1]
                )
    derivative = np.full_like(b, mp(0))
    for i in range(b.shape[1]):
        if k[i + degree] != k[i]:
            derivative[:, i] += degree / (k[i + degree] - k[i]) * previous[:, i]
        if k[i + degree + 1] != k[i + 1]:
            derivative[:, i] -= (
                degree / (k[i + degree + 1] - k[i + 1]) * previous[:, i + 1]
            )
    if not second:
        return b, derivative
    if degree < 2:
        raise ValueError("Second derivatives require spline degree >= 2")
    lower_derivative = np.full_like(previous, mp(0))
    for i in range(previous.shape[1]):
        if k[i + degree - 1] != k[i]:
            lower_derivative[:, i] += (
                (degree - 1) / (k[i + degree - 1] - k[i]) * lower_two[:, i]
            )
        if k[i + degree] != k[i + 1]:
            lower_derivative[:, i] -= (
                (degree - 1) / (k[i + degree] - k[i + 1]) * lower_two[:, i + 1]
            )
    second_derivative = np.full_like(b, mp(0))
    for i in range(b.shape[1]):
        if k[i + degree] != k[i]:
            second_derivative[:, i] += (
                degree / (k[i + degree] - k[i]) * lower_derivative[:, i]
            )
        if k[i + degree + 1] != k[i + 1]:
            second_derivative[:, i] -= (
                degree / (k[i + degree + 1] - k[i + 1]) * lower_derivative[:, i + 1]
            )
    return b, derivative, second_derivative


def mp_trial_values(
    line, spline, points, *, bits=113, second=False, transform=None, layers=(),
    values_only=False,
):
    """Evaluate at MPFR precision; values_only skips unused derivative work."""
    import gmpy2 as g

    if values_only and second:
        raise ValueError("values_only cannot request second derivatives")
    with g.context(precision=bits):
        mp = g.mpfr
        x0, length = mp(float(line.x[0])), mp(float(line.x[-1] - line.x[0]))
        m = len(line.x) - 1
        pi = g.const_pi()
        frequency = pi / length
        frequency2 = frequency**2
        near_node = mp("1e-12")
        fractions = [mp(j) / m for j in range(m)]
        t = np.array([mp(float(v)) for v in points], dtype=object)
        nodes = np.array([x0 + length * j / m for j in range(m)], dtype=object)
        bn, _ = _splines(spline.t, nodes, spline.k, mp)
        spline_values = _splines(spline.t, t, spline.k, mp, second=second)
        b, b1 = spline_values[:2]
        if second:
            b2 = spline_values[2]
        f = np.empty((len(t), m), dtype=object)
        f1 = None if values_only else np.empty_like(f)
        f2 = np.empty_like(f) if second else None
        for i, ti in enumerate(t):
            position = (ti - x0) / length
            for j in range(m):
                delta = position - fractions[j]
                delta -= g.rint(delta)
                u = pi * delta
                su, cu = g.sin_cos(u)
                if abs(delta) < near_node:
                    value = 1 - (m * m - 1) * u * u / 6
                    if not values_only:
                        derivative = -(m * m - 1) * pi * pi * delta / (3 * length)
                else:
                    sm, cm = g.sin_cos(m * u)
                    value = sm / (m * su)
                    if not values_only:
                        derivative = (cm / su - sm * cu / (m * su * su)) * frequency
                if second:
                    if abs(delta) < near_node:
                        second_derivative = (
                            -(m * m - 1) / mp(3)
                            + (3 * m**4 - 10 * m * m + 7) * u * u / 30
                        ) * frequency2
                    else:
                        second_derivative = (
                            -(m * m - 1) * frequency2 * value
                            - 2 * cu / su * frequency * derivative
                        )
                if m % 2 == 0:
                    if second:
                        second_derivative = (
                            second_derivative * cu
                            - 2 * derivative * su * frequency
                            - value * cu * frequency2
                        )
                    if not values_only:
                        derivative = derivative * cu - value * su * frequency
                    value *= cu
                f[i, j] = value
                if not values_only:
                    f1[i, j] = derivative
                if second:
                    f2[i, j] = second_derivative
        projector = np.array(
            [[mp(float(v)) for v in row] for row in line.P], dtype=object
        )
        # Perform the cancellation before multiplying by the sensitive jet map.
        values = (b - f @ bn) @ projector
        values[:, :m] += f
        result = [values]
        if not values_only:
            gradients = (b1 - f1 @ bn) @ projector
            gradients[:, :m] += f1
            result.append(gradients)
        if second:
            curvature = (b2 - f2 @ bn) @ projector
            curvature[:, :m] += f2
            result.append(curvature)
        # Near-dependent enrichment must be combined before casting to float64.
        for width in layers:
            width = mp(float(width))
            for endpoint, sign in ((x0, -1), (x0 + length, 1)):
                exponential = np.array(
                    [g.exp(sign * (ti - endpoint) / width) for ti in t], dtype=object
                )
                for order in range(len(result)):
                    result[order] = np.column_stack(
                        (result[order], exponential * (sign / width) ** order)
                    )
        if transform is not None:
            transform = np.array(
                [[mp(float(v)) for v in row] for row in transform], dtype=object
            )
            result = [value @ transform for value in result]
        return tuple(np.asarray(value, dtype=float) for value in result)


def _mp_chunk(arguments):
    line, spline, points, options = arguments
    return mp_trial_values(line, spline, points, **options)


def evaluate_mp_chunks(line, spline, points, *, executor=None, **options):
    """Evaluate independent point rows with an optional spawn-based CPU pool."""
    points = np.asarray(points)
    if executor is None or len(points) < 128:
        return mp_trial_values(line, spline, points, **options)
    # More chunks than workers balance variable endpoint/spline costs. Each
    # worker retains the original dot-product order within every point row.
    chunks = np.array_split(points, min(16, max(1, len(points)//128)))
    results = list(executor.map(
        _mp_chunk, ((line, spline, chunk, options) for chunk in chunks)
    ))
    return tuple(np.concatenate(items, axis=0) for items in zip(*results))


def evaluate_basis(line, spline, points, *, precision="mpfr", device=None,
                   executor=None, **options):
    """Select reference MPFR or explicitly requested GPU float64 evaluation."""
    if precision == "mpfr":
        return evaluate_mp_chunks(line, spline, points, executor=executor, **options)
    if precision == "float64":
        from ._gpu_basis import gpu_trial_values
        return gpu_trial_values(line, spline, points, device=device, **options)
    raise ValueError("basis_precision must be mpfr or float64")


def with_basis_workers(constructor):
    """Bound pool lifetime to setup; never fork a process with initialized CUDA."""
    from functools import wraps
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    @wraps(constructor)
    def setup(self, *args, **kwargs):
        workers = kwargs.get("basis_workers", 1)
        if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
            raise ValueError("basis_workers must be a positive integer")
        self._basis_executor = None
        if workers == 1 or kwargs.get("basis_precision", "mpfr") == "float64":
            return constructor(self, *args, **kwargs)
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
            self._basis_executor = pool
            try:
                return constructor(self, *args, **kwargs)
            finally:
                self._basis_executor = None
    return setup
