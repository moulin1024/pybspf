"""Evaluate double-precision BSPF checkpoints, without a finite-difference dependency.

New recovery I/O code. Spatial evaluation uses the retained basis, not an
interpolation of pictures or a translation of a solitary-wave profile.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from common import ROOT, geometry, background
from pybspf.trial_spaces import ClosedBSPFLine


def t2(x, a, z):
    return x @ a @ z.T


class SnapshotReader:
    def __init__(self, path: str | Path, q: np.ndarray, s: np.ndarray):
        self.path = Path(path).resolve()
        self.config = json.loads(
            (self.path / "config.json").read_text(encoding="utf-8")
        )
        nx, nz = int(self.config["nx"]), int(self.config["nz"])
        self.geo = geometry(q, s)
        self.s = np.asarray(s)
        lx, lz = ClosedBSPFLine(nx), ClosedBSPFLine(nz)
        with np.load(self.path / "basis.npz", allow_pickle=False) as f:
            Qx, Qz, BQx, BQz = (f[k] for k in ("Qx", "Qz", "BQx", "BQz"))
        vx = lx.values(q, 2)
        g = self.geo["g"]
        self.Px = [
            vx[0] @ Qx,
            (vx[1] / g) @ Qx,
            (vx[2] / g**2 - vx[1] * self.geo["gq"] / g**3) @ Qx,
        ]
        self.Pz = [v @ Qz for v in lz.values(s, 2)]
        self.Tx = [v @ BQx for v in lx.scalar_values(q, 1)]
        self.Tz = [v @ BQz for v in lz.scalar_values(s, 1)]
        self.bg, self.n0 = background(self.geo["z"])
        self.snapshots = {}
        for p in sorted(self.path.glob("state_*.npz")):
            with np.load(p, allow_pickle=False) as f:
                time_s = float(f["time_s"]) if "time_s" in f else float(f["t"]) * 100.0
            key = round(time_s, 7)
            if key in self.snapshots:
                raise ValueError(f"Duplicate saved time {key}: {p}")
            self.snapshots[key] = p
        if not self.snapshots:
            raise FileNotFoundError(f"No actual snapshots in {self.path}")

    @property
    def times(self):
        return sorted(self.snapshots)

    def get(self, time_s: float):
        key = round(float(time_s), 7)
        if key not in self.snapshots:
            raise ValueError(
                f"{time_s:g}s not saved; available: {self.times}. No time interpolation is performed."
            )
        with np.load(self.snapshots[key], allow_pickle=False) as f:
            a, b = f["a"], f["b"]
        px, pz = self.Px, self.Pz
        d, dx, dxx, k = (self.geo[n] for n in ("d", "dx", "dxx", "k"))
        ss = self.s[None, :] - 1
        p = t2(px[0], a, pz[0])
        p_x = t2(px[1], a, pz[0])
        p_s = t2(px[0], a, pz[1])
        p_ss = t2(px[0], a, pz[2])
        p_xs = t2(px[1], a, pz[1])
        p_xx = t2(px[2], a, pz[0])
        u = p_s / d
        w = -p_x + k * p_s / d
        uz = p_ss / d**2
        wx = (
            -p_xx
            + 2 * k * p_xs / d
            + (dxx * ss / d - 2 * k * dx / d**2) * p_s
            - k * k * p_ss / d**2
        )
        bp = t2(self.Tx[0], b, self.Tz[0])
        np_ = t2(self.Tx[0], b, self.Tz[1]) / d
        return dict(
            u=u,
            w=w,
            psi=p,
            uz=uz,
            vorticity=wx - uz,
            bprime=bp,
            N2prime=np_,
            B=self.bg + bp,
            N2=self.n0 + np_,
        )


def field_errors(a, b, W):
    out = {}
    for key in ("velocity", "w", "bprime", "N2prime", "uz", "vorticity"):
        if key == "velocity":
            diff = (a["u"] - b["u"]) ** 2 + (a["w"] - b["w"]) ** 2
            ref = b["u"] ** 2 + b["w"] ** 2
        else:
            diff = (a[key] - b[key]) ** 2
            ref = b[key] ** 2
        out[key + "_relL2"] = float(
            np.sqrt(np.sum(W * diff) / max(np.sum(W * ref), 1e-30))
        )
        out[key + "_absL2"] = float(np.sqrt(np.sum(W * diff)))
    return out
