"""Smooth scalar random-wave manufactured solutions, with analytic Poisson data.

The prescribed k^-5/3 shell variance is a synthetic spectral envelope, not
evidence of turbulent dynamics. Continuous wavevectors do not enforce box
periodicity. Phase-ensemble variance is one; spatial variance need not be one.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class RandomWaveMMS:
    wavevectors: np.ndarray
    amplitudes: np.ndarray
    phases: np.ndarray
    shell_edges: np.ndarray
    shell_indices: np.ndarray
    seed: int

    @classmethod
    def create(
        cls,
        *,
        seed=20260918,
        kmin=np.pi,
        kmax=8 * np.pi,
        shells=8,
        per_shell=8,
        exponent=5 / 3,
    ):
        if not 0 < kmin < kmax or shells < 1 or per_shell < 1:
            raise ValueError(
                "Require positive ordered wave-number limits and mode counts"
            )
        rng = np.random.default_rng(seed)
        edges = np.geomspace(kmin, kmax, shells + 1)
        shell = np.repeat(np.arange(shells), per_shell)
        lo, hi = edges[shell], edges[shell + 1]
        magnitude = lo + (hi - lo) * rng.random(len(shell))
        angle = rng.uniform(0, 2 * np.pi, len(shell))
        wavevectors = magnitude[:, None] * np.column_stack(
            (np.cos(angle), np.sin(angle))
        )
        amplitudes = np.sqrt(2 * magnitude ** (-exponent) * (hi - lo) / per_shell)
        amplitudes /= np.sqrt(np.sum(amplitudes**2) / 2)
        return cls(
            wavevectors,
            amplitudes,
            rng.uniform(0, 2 * np.pi, len(shell)),
            edges,
            shell,
            seed,
        )

    def evaluate(self, points):
        """Return u, grad(u), and f=-Delta(u) from closed-form expressions."""
        phase = np.asarray(points) @ self.wavevectors.T + self.phases
        cosine, sine = np.cos(phase), np.sin(phase)
        value = cosine @ self.amplitudes
        gradient = -(sine * self.amplitudes) @ self.wavevectors
        forcing = cosine @ (self.amplitudes * np.sum(self.wavevectors**2, axis=1))
        return value, gradient, forcing

    def save(self, path):
        np.savez(
            path,
            wavevectors=self.wavevectors,
            amplitudes=self.amplitudes,
            phases=self.phases,
            shell_edges=self.shell_edges,
            shell_indices=self.shell_indices,
            seed=self.seed,
        )
