"""Optional MPFR assembly of BSPF cardinal values, returned in float64.

Only setup uses high precision, to avoid cancelling huge spline/Fourier
components of endpoint cardinal functions. Runtime matrices remain float64.
Requires gmpy2, used here only for the 1D accuracy experiment.
"""

import numpy as np


def _splines(knots, points, degree, mp):
    k = np.array([mp(float(v)) for v in knots], dtype=object)
    t = np.asarray(points, dtype=object)
    b = np.array(((t[:, None] >= k[:-1]) & (t[:, None] < k[1:])), dtype=object)
    for p in range(1, degree + 1):
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
    return b, derivative


def mp_trial_values(line, spline, points, *, bits=113):
    import gmpy2 as g

    with g.context(precision=bits):
        mp = g.mpfr
        x0, length = mp(float(line.x[0])), mp(float(line.x[-1] - line.x[0]))
        m = len(line.x) - 1
        pi = g.const_pi()
        t = np.array([mp(float(v)) for v in points], dtype=object)
        nodes = np.array([x0 + length * j / m for j in range(m)], dtype=object)
        bn, _ = _splines(spline.t, nodes, spline.k, mp)
        b, b1 = _splines(spline.t, t, spline.k, mp)
        f = np.empty((len(t), m), dtype=object)
        f1 = np.empty_like(f)
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
                if m % 2 == 0:
                    derivative = derivative * g.cos(u) - value * g.sin(u) * pi / length
                    value *= g.cos(u)
                f[i, j], f1[i, j] = value, derivative
        projector = np.array(
            [[mp(float(v)) for v in row] for row in line.P], dtype=object
        )
        # Perform the cancellation before multiplying by the sensitive jet map.
        values = (b - f @ bn) @ projector
        gradients = (b1 - f1 @ bn) @ projector
        values[:, :m] += f
        gradients[:, :m] += f1
        return np.asarray(values, dtype=float), np.asarray(gradients, dtype=float)
