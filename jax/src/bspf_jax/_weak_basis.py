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
    line, spline, points, *, bits=113, second=False, transform=None, layers=()
):
    import gmpy2 as g

    with g.context(precision=bits):
        mp = g.mpfr
        x0, length = mp(float(line.x[0])), mp(float(line.x[-1] - line.x[0]))
        m = len(line.x) - 1
        pi = g.const_pi()
        t = np.array([mp(float(v)) for v in points], dtype=object)
        nodes = np.array([x0 + length * j / m for j in range(m)], dtype=object)
        bn, _ = _splines(spline.t, nodes, spline.k, mp)
        spline_values = _splines(spline.t, t, spline.k, mp, second=second)
        b, b1 = spline_values[:2]
        if second:
            b2 = spline_values[2]
        f = np.empty((len(t), m), dtype=object)
        f1 = np.empty_like(f)
        f2 = np.empty_like(f) if second else None
        for i, ti in enumerate(t):
            position = (ti - x0) / length
            for j in range(m):
                delta = position - mp(j) / m
                delta -= g.rint(delta)
                u = pi * delta
                if abs(delta) < mp("1e-12"):
                    value = 1 - (m * m - 1) * u * u / 6
                    derivative = -(m * m - 1) * pi * pi * delta / (3 * length)
                else:
                    su, cu = g.sin_cos(u)
                    sm, cm = g.sin_cos(m * u)
                    value = sm / (m * su)
                    derivative = (cm / su - sm * cu / (m * su * su)) * pi / length
                if second:
                    if abs(delta) < mp("1e-12"):
                        second_derivative = (
                            -(m * m - 1) / mp(3)
                            + (3 * m**4 - 10 * m * m + 7) * u * u / 30
                        ) * (pi / length) ** 2
                    else:
                        second_derivative = (
                            -(m * m - 1) * (pi / length) ** 2 * value
                            - 2 * cu / su * pi / length * derivative
                        )
                if m % 2 == 0:
                    if second:
                        second_derivative = (
                            second_derivative * g.cos(u)
                            - 2 * derivative * g.sin(u) * pi / length
                            - value * g.cos(u) * (pi / length) ** 2
                        )
                    derivative = derivative * g.cos(u) - value * g.sin(u) * pi / length
                    value *= g.cos(u)
                f[i, j], f1[i, j] = value, derivative
                if second:
                    f2[i, j] = second_derivative
        projector = np.array(
            [[mp(float(v)) for v in row] for row in line.P], dtype=object
        )
        # Perform the cancellation before multiplying by the sensitive jet map.
        values = (b - f @ bn) @ projector
        gradients = (b1 - f1 @ bn) @ projector
        values[:, :m] += f
        gradients[:, :m] += f1
        result = [values, gradients]
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
