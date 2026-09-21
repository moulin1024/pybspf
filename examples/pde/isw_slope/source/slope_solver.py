"""ISW slope case assembled with the reusable pybspf/JAX model."""

import jax

jax.config.update("jax_enable_x64", True)
from bspf_jax.mapped_boussinesq import MappedBoussinesq
from bspf_jax import isw_slope as backend
from common import geometry, background, SharedInitial, NU, KAPPA


class BSPF(MappedBoussinesq):
    def __init__(self, nx, nz, quad=2.0, nu=NU, kappa=KAPPA):
        super().__init__(
            nx,
            nz,
            geometry=geometry,
            background=background,
            quad=quad,
            nu=nu,
            kappa=kappa,
        )

    def initial(self):
        return self.project_initial(SharedInitial())
