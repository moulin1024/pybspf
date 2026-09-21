# ISW slope numerical regression fixtures

These arrays were captured before removing the independent legacy NumPy solver
from the example. Source: `BSPF_ISW_Slope_Run01_Rebuilt/source/slope_numpy.py`
and its retained ClosedLine basis, before migration on 2026-09-21.

Grids: 33×33 and 65×33, quadrature factor 2. RNG default_rng(214), stream
coefficients normal*1e-7 and buoyancy coefficients normal*1e-6. RK4 dt=0.0001
in nondimensional units. Analytic initial projection uses
u=1e-4 sin(pi*q) sin(pi*s), w=0, b=1e-5 sin(pi*q) sin(pi*s).

Files contain input coefficients, mass action, six velocity/derivative fields,
RHS, RK4 result and analytic initial projection. They are regression evidence,
not a convergence reference or another maintained solver implementation.
