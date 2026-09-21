# Architecture

The numerical core is `pybspf`, implemented with JAX. Applications depend on
models, and models depend on the core. The core never imports models, apps or
archived backends. NumPy/SciPy host preprocessing does not constitute a second
runtime backend.

Core modules own bases, plans, operators, calculus, Galerkin spaces, extension,
tensor algebra and generic time integration. `trial_spaces.ClosedBSPFLine`
retains the QR-based host construction needed by mapped Boussinesq models.

`bspf_models` groups physical equations by domain. Its internal `_numerics`
contains shared experimental trial spaces and assembly routines; elliptic
solvers do not depend on fluid equations. Fields, boundaries, physics and
model diagnostics remain with their models.

`bspf_sim` owns configuration, checkpointing, CLI, outputs and validation
workflows. Package initializers are lightweight. Host-only core facilities are
imported explicitly and never change JAX's global device/precision configuration.

See [migration](migration.md) for exact module ownership and API changes.
