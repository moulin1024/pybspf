"""GPU setup decompositions with explicit host results for geometry assembly.

No CPU fallback: device failures or non-finite factors are reported to the caller.
The device kernels remain separate so their placement and transfer behavior can
be verified independently of the host-plan interface.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


@partial(jax.jit, static_argnames=("full_matrices",))
def _svd(matrix, *, full_matrices=False):
    # Explicit QR-based cuSOLVER SVD supports both tall and wide matrices.
    # Avoid forming A.T @ A: the boundary matrices are nearly rank deficient.
    return jax.lax.linalg.svd(
        matrix, full_matrices=full_matrices,
        algorithm=jax.lax.linalg.SvdAlgorithm.QR,
    )


@jax.jit
def _eigh(matrix):
    return jnp.linalg.eigh((matrix + matrix.T) / 2, symmetrize_input=False)


def _upload(matrix, device):
    if device is None or device.platform != "gpu":
        raise ValueError("GPU decomposition requires a GPU device")
    if not jax.config.x64_enabled:
        raise ValueError("Enable jax_enable_x64 for GPU setup decompositions")
    return jax.device_put(np.asarray(matrix, dtype=np.float64), device)


def _checked_host(factors, name):
    result = jax.device_get(factors)
    if not all(np.all(np.isfinite(a)) for a in result):
        raise np.linalg.LinAlgError(f"GPU {name} returned non-finite factors")
    return result


def gpu_svd(matrix, *, device, full_matrices=False):
    return _checked_host(_svd(_upload(matrix, device), full_matrices=full_matrices), "SVD")


def gpu_eigh(matrix, *, device):
    return _checked_host(_eigh(_upload(matrix, device)), "eigendecomposition")
