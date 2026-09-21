"""Prepare a 2-D MITgcm nonhydrostatic counterpart of the BSPF slope case.

321 wet x columns plus two dry wall columns, one periodic y column, 161
z levels. This is the same nominal resolution, not an equal-DOF or
identical-discretization comparison to the terrain-following BSPF basis.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'examples/pde/isw_slope/source'))
from common import SharedInitial, depth, integral, IT, mapping, background


def prepare(out, nx=321, nz=161, dt=0.5, tfinal=50., threads=1):
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(tfinal) or tfinal <= 0 or not np.isclose(tfinal/dt, round(tfinal/dt)):
        raise ValueError("dt must divide the positive simulation interval")
    steps = round(tfinal/dt)
    if threads not in (1,8):
        raise ValueError('Supported builds use 1 or 8 threads')
    # Pad with dry columns to permit eight equal x tiles, preserving all 321
    # fluid columns, their coordinates and the physical initial state.
    ncol = nx+2 if threads == 1 else ((nx+2+7)//8)*8
    right_walls = ncol-nx-1
    out.mkdir(parents=True, exist_ok=True)
    initial = SharedInitial()
    qf = np.linspace(0., 1., nx+1)
    xf = 100*mapping(qf)[0]
    xc = (xf[:-1]+xf[1:])/2
    qc = integral(xc/100)/IT
    depths = 100*depth(xc/100)[0]
    dz = 100/nz
    zf = -np.arange(nz+1)*dz
    zc = (zf[:-1]+zf[1:])/2
    # MITgcm rounds partial cells below half hFacMin down to dry, and
    # raises other thin bottom cells to hFacMin. Use the same mask for U.
    hfac = np.clip((depths[None, :]+zf[:-1, None])/dz, 0., 1.)
    hfac = np.where(hfac < .1, 0., np.maximum(hfac, .2))
    effective_depth = dz*hfac.sum(axis=0)
    sigma = np.clip(1+zc[:, None]/depths[None, :], 0., 1.)
    bprime = initial.b.ev(np.broadcast_to(qc, sigma.shape), sigma)
    buoyancy = (background(zc[:, None]/100)[0]+bprime)/100
    gravity, alpha = 9.81, 1e-3
    temperature = np.where(hfac > 0, buoyancy/(gravity*alpha), 0.)
    theta = np.zeros((nz, 1, ncol))
    theta[:, 0, 1:nx+1] = temperature
    u = np.zeros_like(theta)
    # Cell-mean face velocity from streamfunction differences. A single zero
    # streamfunction on both top/bottom makes each face transport exactly zero.
    # W is then initialized by MITgcm's discrete continuity equation.
    hfac_w = np.minimum(hfac[:, :-1], hfac[:, 1:])
    q_edges = qf[1:-1]
    depth_edges = 100*depth(xf[1:-1]/100)[0]
    face_depth = dz*hfac_w.sum(axis=0)
    psi = np.zeros((nz+1, nx-1))
    q_basis = initial.bq(q_edges)
    coefficients = q_basis @ initial.c
    for i in range(nx-1):
        s = np.clip(1+zf/depth_edges[i], 0., 1.)
        psi[:, i] = 100*(initial.bs(s) @ coefficients[i])
        psi[(zf <= -face_depth[i]) | (zf >= 0), i] = 0.
    u_face = np.divide(psi[:-1]-psi[1:], dz*hfac_w,
                       out=np.zeros_like(hfac_w), where=hfac_w > 0)
    u[:, 0, 2:nx+1] = u_face
    topo = np.zeros((1, ncol))
    topo[0, 1:nx+1] = -depths
    dx = np.r_[xf[1]-xf[0], np.diff(xf), np.repeat(xf[-1]-xf[-2],right_walls)]
    def binary(name, a):
        np.asarray(a, dtype='>f8').tofile(out/name)
    binary('topography.bin', topo)
    binary('theta_initial.bin', theta)
    binary('u_initial.bin', u)
    binary('delX.bin', dx)
    np.savez(out/'initial_mapping.npz', x_faces_m=xf, x_centers_m=xc,
             z_faces_m=zf, z_centers_m=zc, hfac=hfac,
             effective_depth_m=effective_depth, buoyancy=buoyancy)
    (out/'data').write_text(f'''# Physical counterpart of the BSPF ISW slope run
 &PARM01
 viscAh=0.01,
 viscAz=0.01,
 viscA4=0.,
 diffKhT=0.003,
 diffKzT=0.003,
 diffKhS=0.,
 diffKzS=0.,
 no_slip_sides=.TRUE.,
 no_slip_bottom=.TRUE.,
 bottomVisc_pCell=.TRUE.,
 rigidLid=.TRUE.,
 implicitFreeSurface=.FALSE.,
 nonHydrostatic=.TRUE.,
 hFacMin=0.2,
 hFacMinDz=0.,
 useCoriolis=.FALSE.,
 f0=0.,
 beta=0.,
 gravity=9.81,
 rhoConst=1025.,
 rhonil=1025.,
 eosType='LINEAR',
 tAlpha=0.001,
 sBeta=0.,
 tRef={nz}*0.,
 sRef={nz}*0.,
 saltStepping=.FALSE.,
 implicitDiffusion=.FALSE.,
 implicitViscosity=.FALSE.,
 tempAdvScheme=2,
 readBinaryPrec=64,
 writeBinaryPrec=64,
 debugLevel=1,
 &
 &PARM02
 cg2dMaxIters=1000,
 cg2dTargetResidual=1.E-11,
 cg3dMaxIters=10000,
 cg3dTargetResidual=1.E-11,
 &
 &PARM03
 nIter0=0,
 nTimeSteps={steps},
 deltaT={dt:.17g},
 abEps=0.1,
 pChkptFreq=0.,
 chkptFreq=0.,
 dumpFreq=50.,
 monitorFreq={dt:.17g},
 monitorSelect=2,
 &
 &PARM04
 usingCartesianGrid=.TRUE.,
 xgOrigin={-dx[0]:.17g},
 delXFile='delX.bin',
 delY=1.,
 delZ={nz}*{dz:.17g},
 &
 &PARM05
 bathyFile='topography.bin',
 hydrogThetaFile='theta_initial.bin',
 uVelInitFile='u_initial.bin',
 &
''')
    (out/'data.pkg').write_text(' &PACKAGES\n &\n')
    (out/'eedata').write_text(f' &EEPARMS\n nTx={threads},\n nTy=1,\n &\n')
    meta = dict(wet_x_columns=nx, allocated_x_columns=ncol, y_columns=1,
        threads=threads, dry_left_columns=1, dry_right_columns=right_walls,
        z_levels=nz, wet_tracer_cells=int((hfac > 0).sum()), dx_min_m=float(dx.min()),
        dx_max_m=float(dx.max()), dz_m=dz, dt_s=dt, steps=steps, final_time_s=tfinal,
        initial_sha256=hashlib.sha256((initial_path := ROOT/'examples/pde/isw_slope/reference/shared_initial.npz').read_bytes()).hexdigest(),
        initial_source=str(initial_path),max_face_transport_m2s=float(np.max(np.abs(np.sum(u_face*dz*hfac_w,axis=0)))),
        differences=['MITgcm finite-volume z levels with partial bottom cells versus BSPF terrain-following spectral basis',
                     'MITgcm Adams-Bashforth/pressure projection versus BSPF classical RK4',
                     'MITgcm W initialized from discrete continuity; U is layer-averaged from the supplied streamfunction',
                     f'{1+right_walls} dry x columns close the otherwise periodic domain; wet grid unchanged',
                     'Local viscous-flux overrides enforce a no-slip upper rigid lid, matching BSPF'],
        gravity=gravity,thermal_expansion=alpha)
    (out/'case.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--dt', type=float, default=0.5)
    p.add_argument('--tfinal', type=float, default=50.)
    p.add_argument('--threads', type=int, choices=(1,8), default=1)
    args = p.parse_args()
    prepare(args.out, dt=args.dt, tfinal=args.tfinal, threads=args.threads)
