"""Oversampled normal continuation with spectral cutoff regularization.

Seam control uses a shared Cartesian BSPF field, not independent chart fields.
The continuation relation is imposed on that field as a soft linear constraint.
"""

import numpy as np
from numpy.polynomial import chebyshev as ch

from pybspf.normal_continuation import chart
from bspf_models.elliptic.smooth_extension import factors
from pybspf.tensor import tensor_product


def normal_weights(ratios, *, samples=25, degree=14, retained=6, derivative=0):
    """Map interior Lobatto samples to exterior values; truncate modal tail.

    Fit degree `degree` on [-1,0] using `samples` values, then retain only
    degrees 0..retained. This is a spectral cutoff, not singular-value truncation.
    No analytic input derivatives are needed. Value weights preserve constants.
    Derivative weights differentiate in normalized distance r/width; divide
    by width**derivative to obtain physical normal derivatives.
    """
    if not 0 <= retained <= degree < samples:
        raise ValueError("Require 0 <= retained <= degree < samples")
    z = ch.chebpts2(samples)
    fit = np.linalg.pinv(ch.chebvander(z, degree), rcond=1e-14)
    if derivative not in (0, 1, 2):
        raise ValueError("Derivative must be 0, 1 or 2")
    polynomial = ch.chebder(fit[: retained + 1], m=derivative, axis=0)
    weights = ch.chebval(1 + 2 * np.asarray(ratios), polynomial).T * 2**derivative
    return (z - 1) / 2, weights


def value_matrix(line, points):
    (x, _, _), (y, _, _) = factors(line, points)
    return tensor_product(x, y, paired=True)


def collar_matrices(geometry, *, width=0.01, retained=6, boundary_order=4):
    domain = geometry.domain
    q, _ = np.polynomial.legendre.leggauss(boundary_order)
    t = (np.arange(domain.period)[:, None] + (q + 1) / 2).ravel()
    ri, weights = normal_weights([0.15, 0.3, 0.45], retained=retained)
    inside = chart(domain, t[:, None], width * ri[None, :])[0]
    outside = chart(domain, t[:, None], width * np.array([0.15, 0.3, 0.45])[None, :])[0]
    b_in = value_matrix(geometry.line, inside.reshape(-1, 2))
    b_out = value_matrix(geometry.line, outside.reshape(-1, 2))
    extension = np.einsum(
        "es,tsn->ten", weights, b_in.reshape(len(t), len(ri), -1)
    ).reshape(b_out.shape)
    return dict(
        inside=inside,
        outside=outside,
        weights=weights,
        exterior_basis=b_out,
        relation=b_out - extension,
    )
