"""Frozen physical model. H=100 m, U=1 m/s; time unit 100 s, B unit .01 m/s²."""

from pathlib import Path
import numpy as np
from scipy.special import erf
from scipy.interpolate import CubicSpline, RectBivariateSpline, BSpline

ROOT = Path(__file__).resolve().parents[1]
NU = 1e-4
KAPPA = 3e-5
L = 18.0


def depth(x):
    r = (np.asarray(x) - 8.5) / 0.9
    t = np.tanh(r)
    return (
        1 - 0.275 * (1 + t),
        -0.275 / 0.9 * (1 - t * t),
        0.55 / 0.9**2 * t * (1 - t * t),
    )


def integral(x):
    return x + 4.5 * np.sqrt(np.pi) * (erf((x - 9.5) / 1.5) - erf(-9.5 / 1.5))


_grid = np.linspace(0, L, 40001)
IT = float(integral(L))
_inv = CubicSpline(integral(_grid) / IT, _grid)


def mapping(q):
    x = _inv(q)
    a = (x - 9.5) / 1.5
    e = np.exp(-a * a)
    ww = 1 + 6 * e
    return x, IT / ww, 12 * a * e / 1.5 * IT**2 / ww**3


def geometry(q, s):
    x, g, gq = mapping(np.asarray(q))
    d, dx, dxx = depth(x)
    ss = np.asarray(s)[None, :]
    return dict(
        x=x,
        g=g[:, None],
        gq=gq[:, None],
        d=d[:, None],
        dx=dx[:, None],
        dxx=dxx[:, None],
        k=dx[:, None] * (ss - 1),
        z=d[:, None] * (ss - 1),
    )


def background(z):
    tt = np.tanh((z + 0.25) / 0.06)
    return 0.5 * tt, 0.5 / 0.06 * (1 - tt * tt)


class SharedInitial:
    """Recovered shared C4 quintic field, NOT the missing original continuous DJL."""

    def __init__(self):
        with np.load(ROOT / "reference/shared_initial.npz") as f:
            self.q = f["q"]
            self.s = f["s"]
            self.c = f["c"]
            tq = f["tq"]
            ts = f["ts"]
            bp = f["bprime"]
        self.bq = BSpline(tq, np.eye(self.c.shape[0]), 5)
        self.bs = BSpline(ts, np.eye(self.c.shape[1]), 5)
        self.b = RectBivariateSpline(self.q, self.s, bp, kx=5, ky=5, s=0)

    def p(self, q, s, dx=0, dy=0):
        return self.bq(q, nu=dx) @ self.c @ self.bs(s, nu=dy).T

    def velocity(self, q, s):
        g = geometry(q, s)
        ps = self.p(q, s, dy=1)
        pq = self.p(q, s, dx=1)
        return ps / g["d"], -pq / g["g"] + g["k"] / g["d"] * ps
