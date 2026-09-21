"""Local polynomial continuation in exact spline-normal coordinates.

Uses values only in an interior collar. This isolates continuation accuracy;
it is not yet a blended-to-zero Fourier continuation or a PDE solver.
Tangential polynomials are piecewise on the original spline spans.
"""

import numpy as np
from numpy.polynomial import chebyshev as ch


def chart(domain, t, r):
    t, r = np.broadcast_arrays(t, r)
    v, a, b = (domain.curve(t, j) for j in (1, 2, 3))
    speed = np.linalg.norm(v, axis=-1)
    sp = np.sum(v * a, axis=-1) / speed
    spp = (np.sum(a * a + v * b, axis=-1) - sp * sp) / speed
    tangent = v / speed[..., None]
    tp = a / speed[..., None] - v * (sp / speed**2)[..., None]
    tpp = (
        b / speed[..., None]
        - 2 * a * (sp / speed**2)[..., None]
        - v * (spp / speed**2)[..., None]
        + 2 * v * (sp**2 / speed**3)[..., None]
    )

    def rotate(x):
        return np.stack((x[..., 1], -x[..., 0]), axis=-1)

    n, nt, ntt = map(rotate, (tangent, tp, tpp))
    point = domain.curve(t) + r[..., None] * n
    jac = np.stack((v + r[..., None] * nt, n), axis=-1)
    second = np.zeros(t.shape + (2, 2, 2))  # physical component, chart axes
    second[..., :, 0, 0] = a + r[..., None] * ntt
    second[..., :, 0, 1] = nt
    second[..., :, 1, 0] = nt
    return point, jac, second


class NormalContinuation:
    def __init__(self, domain, values, *, width, normal_degree=10, tangent_degree=24):
        if width <= 0 or normal_degree < 2 or tangent_degree < 2:
            raise ValueError("Positive width and degrees >= 2 required")
        self.domain, self.width = domain, width
        z = ch.chebpts2(normal_degree + 1)
        s = ch.chebpts2(tangent_degree + 1)
        self.coefficients = []
        for span in range(domain.period):
            t, r = np.meshgrid(span + (s + 1) / 2, width * (z - 1) / 2, indexing="ij")
            points = chart(domain, t, r)[0]
            data = np.asarray(values(points.reshape(-1, 2))).reshape(t.shape)
            # Separable Chebyshev interpolation; no exact derivative input.
            c = ch.chebfit(s, data, tangent_degree)
            c = ch.chebfit(z, c.T, normal_degree).T
            self.coefficients.append(c)

    def evaluate(self, t, r):
        t, r = np.broadcast_arrays(np.asarray(t, float), np.asarray(r, float))
        shape = t.shape
        t, r = t.ravel(), r.ravel()
        # Floating-point modulo can round a tiny negative parameter to period.
        wrapped = np.minimum(
            t % self.domain.period, np.nextafter(float(self.domain.period), 0.0)
        )
        index = np.floor(wrapped).astype(int)
        result = np.empty((len(t), 6))
        for span, c in enumerate(self.coefficients):
            pick = index == span
            s = 2 * (wrapped[pick] - span) - 1
            z = 1 + 2 * r[pick] / self.width
            for col, (i, j) in enumerate(
                ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))
            ):
                cc = ch.chebder(ch.chebder(c, m=i, axis=0), m=j, axis=1)
                result[pick, col] = (
                    ch.chebval2d(s, z, cc) * 2**i * (2 / self.width) ** j
                )
        _, jac, second = chart(self.domain, t, r)
        inv = np.linalg.inv(jac)
        grad = np.einsum("nai,na->ni", inv, result[:, 1:3])
        hq = np.empty((len(t), 2, 2))
        hq[:, 0, 0], hq[:, 0, 1] = result[:, 3], result[:, 4]
        hq[:, 1, 0], hq[:, 1, 1] = result[:, 4], result[:, 5]
        corrected = hq - np.einsum("nk,nkab->nab", grad, second)
        hess = np.einsum("nai,nab,nbj->nij", inv, corrected, inv)
        return (
            result[:, 0].reshape(shape),
            grad.reshape(shape + (2,)),
            np.trace(hess, axis1=-2, axis2=-1).reshape(shape),
        )
