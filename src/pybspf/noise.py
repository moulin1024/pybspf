"""Joint spline/Fourier regularization, using FFTs and reduced augmented QR.

All kernels are JAX-native. Noise is homoscedastic, independent across samples,
and measured as sqrt(E|epsilon|**2), including for complex fields.
"""

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jl

from pybspf.basis import basis_matrix


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["sigma", "alphas", "transfer", "projector"],
    meta_fields=[],
)
@dataclass(frozen=True)
class NoisePlan:
    sigma: jax.Array
    alphas: jax.Array
    transfer: jax.Array
    projector: jax.Array


class NoiseSelection(NamedTuple):
    """One selection per physical axis; ratios compare fit residual to expected noise."""

    index: jax.Array
    alpha: jax.Array
    residual_ratio: jax.Array
    at_search_edge: jax.Array
    noise_std: jax.Array


def assemble_noise(x, knots, basis, omega, *, degree, order, sigma, alphas):
    # Gauss-Legendre quadrature integrates squared spline roughness exactly.
    count = degree - order + 1
    k = jnp.arange(1, count, dtype=x.dtype)
    off = k / jnp.sqrt(4 * k * k - 1)
    nodes, vectors = (
        jnp.linalg.eigh(jnp.diag(off, 1) + jnp.diag(off, -1))
        if count > 1
        else (jnp.zeros(1), jnp.ones((1, 1)))
    )
    weights = 2 * vectors[0] ** 2
    breaks = knots[degree:-degree]
    width = jnp.diff(breaks)
    points = (
        (breaks[:-1, None] + breaks[1:, None]) / 2 + width[:, None] * nodes / 2
    ).ravel()
    weights = (width[:, None] * weights / 2 * x.size / (x[-1] - x[0])).ravel()
    roughness = jnp.sqrt(weights[:, None]) * basis_matrix(
        knots, points, degree=degree, derivative=order
    )
    spectrum = jnp.fft.fft(basis, axis=0, norm="ortho")

    def candidate(alpha):
        penalty = alpha * jnp.abs(omega) ** (2 * order)
        H = (1 / (1 + penalty)).at[0].set(0.0)
        S = (penalty / (1 + penalty)).at[0].set(1.0)
        augmented = jnp.concatenate(
            (jnp.sqrt(S[:, None]) * spectrum, jnp.sqrt(alpha) * roughness)
        )
        Q, R = jnp.linalg.qr(augmented, mode="reduced")
        projector = jl.solve_triangular(R, Q[: x.size].conj().T * jnp.sqrt(S)[None, :])
        return H, projector

    H, projector = jax.lax.map(candidate, alphas)
    return NoisePlan(jnp.asarray(sigma, dtype=x.dtype), alphas, H, projector)


def _coefficients(p, spectrum, index, real):
    c = p.noise.projector[index] @ spectrum
    return c.real if real else c


def _fit(p, spectrum, index, real, order=0):
    c = _coefficients(p, spectrum, index, real)
    spline = p.basis[0] @ c
    residual_hat = p.noise.transfer[index, :, None] * (
        spectrum - jnp.fft.fft(spline, axis=0, norm="ortho")
    )
    tail = jnp.fft.ifft(
        residual_hat * (1j * p.omega[:, None]) ** order, axis=0, norm="ortho"
    )
    return p.basis[order] @ c + (tail.real if real else tail)


def select_noise(p, samples):
    """One alpha for the complete field/batch, selected from original samples.

    Scan candidates sequentially to avoid an alpha-by-volume temporary in 3D.
    Selection is discrete. Autodiff differentiates the chosen operator locally,
    not the argmin. No reference derivative is used.
    """
    values = samples.reshape((samples.shape[0], -1))
    real = not jnp.iscomplexobj(values)
    spectrum = jnp.fft.fft(values, axis=0, norm="ortho")
    sigma = p.noise.sigma
    target = sigma * jnp.sqrt(values.size)

    def error(index):
        fit = _fit(p, spectrum, index, real)
        return jnp.linalg.norm(fit - values) / target

    ratios = jax.lax.map(error, jnp.arange(p.noise.alphas.size))
    index = jnp.argmin(jnp.abs(ratios - 1))
    return NoiseSelection(
        index,
        p.noise.alphas[index],
        ratios[index],
        (index == 0) | (index == p.noise.alphas.size - 1),
        sigma,
    )


def apply_noise(p, samples, index, order):
    shape = samples.shape
    spectrum = jnp.fft.fft(samples.reshape((shape[0], -1)), axis=0, norm="ortho")
    return _fit(p, spectrum, index, not jnp.iscomplexobj(samples), order).reshape(shape)
