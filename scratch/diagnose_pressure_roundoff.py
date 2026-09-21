"""Recompute identical spline corrections with two FFT implementations."""

import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from pybspf.basis import basis_matrix
from pybspf.basis import open_knots
from bspf_models._numerics.trial_spaces import _fourier

jax.config.update("jax_enable_x64", True)
rows = []
for n in [512, 1024, 2048]:
    data = np.load(f"build/pressure_accuracy_diagnosis/line_{n}.npz")
    grid = jnp.linspace(0, 1, n)
    knots = open_knots(0.0, 1.0, degree=13, n_basis=32)
    B = np.asarray(basis_matrix(knots, grid, degree=13))
    B1 = np.asarray(basis_matrix(knots, grid, degree=13, derivative=1))
    mult = 2j * np.pi * np.fft.fftfreq(n - 1, d=1 / (n - 1))
    fn = np.fft.ifft(np.fft.fft(B[:-1], axis=0) * mult[:, None], axis=0).real
    low_numpy = B1 - np.concatenate([fn, fn[:1]])
    low_jax = B1 - np.asarray(_fourier(jnp.asarray(B), jnp.asarray(mult)))
    delta = low_numpy - low_jax
    P = data["P"]
    D = data["D"]
    response = delta @ P
    random_delta = np.random.default_rng(11).normal(size=low_jax.shape) * 1e-14
    row = dict(
        N=n,
        fft_low_difference_linf=float(abs(delta).max()),
        resulting_D_difference_relative=float(
            np.linalg.norm(response) / np.linalg.norm(D)
        ),
        resulting_D_difference_linf=float(abs(response).max()),
        random_low_perturbation_rms=1e-14,
        random_perturbation_D_relative=float(
            np.linalg.norm(random_delta @ P) / np.linalg.norm(D)
        ),
        saved_low_difference_linf=float(abs(low_jax - data["low"]).max()),
        per_column=[
            dict(
                index=i,
                low_linf=float(abs(data["low"][:, i]).max()),
                P_row_norm=float(np.linalg.norm(P[i])),
            )
            for i in range(32)
        ],
    )
    rows.append(row)
    print(json.dumps({k: v for k, v in row.items() if k != "per_column"}), flush=True)
Path("build/pressure_accuracy_diagnosis/roundoff.json").write_text(
    json.dumps(rows, indent=2)
)
