"""CPU audit of ripple entry from a smooth, unforced rational Stokes lift.

No production operators are changed. Local PDE derivatives use independent
fourth-order finite differences, with a halved diagnostic spacing control.
"""
import argparse
from functools import lru_cache
import json
from pathlib import Path
from time import perf_counter

import jax
import numpy as np
import scipy.linalg as la
from bspf_models.fluids.immersed_flow import ImmersedFlowPlan, channel_lift


def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a)**2)))


def cosine(a, b):
    a, b = np.ravel(a), np.ravel(b)
    return float(a @ b / (la.norm(a)*la.norm(b)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--baseline', type=Path,
        default=Path('build/immersed_flow/rational_ns_nosponge_20260922'))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    baseline = np.load(args.baseline/'fields.npz')
    meta = json.loads((args.baseline/'summary.json').read_text())
    for key, value in dict(nx=73, ny=33, reynolds=20., dt=.02, final_time=1.,
            wall_method='rational', buffer_strength=0., quadrature_factor=4.).items():
        if meta[key] != value:
            raise ValueError(f'Unexpected baseline {key}: {meta[key]}')
    jax.config.update('jax_enable_x64', True)
    p = ImmersedFlowPlan(nx=73, ny=33, reynolds=20, wall_method='rational',
                        buffer_strength=0, quadrature_factor=4, basis_workers=4)
    print('SETUP', p.setup_seconds, flush=True)
    @lru_cache(maxsize=24)
    def factors(axis, data):
        return p._line_values(axis, np.frombuffer(data, dtype=np.float64))

    def parts(coeff, points, affine=False):
        bf = np.zeros((6, len(points)))
        if np.any(coeff):
            fac = []
            for axis in (0, 1):
                coords, idx = np.unique(points[:, axis], return_inverse=True)
                fac.append(tuple(a[idx] for a in factors(axis, coords.tobytes())))
            bx, by = fac
            c = coeff.reshape(p.shape)
            def pair(i, j):
                return np.sum((bx[i] @ c)*by[j], axis=1)
            bf = np.array([pair(0,0), pair(0,1), -pair(1,0),
                           pair(1,1), pair(0,2), -pair(2,0)])
        rc = p.rational_modes @ (p.rational_map @ coeff)
        if affine:
            bf += np.array(channel_lift(points, p.bounds[2], p.peak))
            rc += p.rational_lift
        rf = np.array(p.rational.evaluate(points, rc))
        return bf, rf

    x, y = np.linspace(-.85, -.35, 31), np.linspace(-.8, .8, 161)
    xx, yy = np.meshgrid(x,y); shape = xx.shape
    points = np.column_stack((xx.ravel(), yy.ravel()))
    a0 = p.stokes_state.copy()
    c0 = p.coefficients(a0)
    b, r = parts(c0, points, True); initial = b+r
    omega0 = (initial[5]-initial[4]).reshape(shape)
    da = p.rhs(a0)
    db, dr = parts(p.transform @ da, points)
    rb, rr = (db[5]-db[4]).reshape(shape), (dr[5]-dr[4]).reshape(shape)
    rate = rb+rr
    arrays = dict(x=x, y=y, omega0=omega0, discrete_rate=rate,
                  bspf_rate=rb, rational_rate=rr)
    report = dict(parameters=dict(nx=73, ny=33, reynolds=20, wall_method='rational',
                    buffer_strength=0, quadrature_factor=4, basis_precision='mpfr'),
                  setup_seconds=p.setup_seconds, local_pde=[])
    def save():
        (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        np.savez_compressed(args.out/'fields.npz', **arrays)
    for h in (.002, .001):
        shifts = [(0,0),(-2*h,0),(-h,0),(h,0),(2*h,0),
                  (0,-2*h),(0,-h),(0,h),(0,2*h)]
        probe = np.vstack([points+shift for shift in shifts])
        bb, rf = parts(c0, probe, True)
        w = ((bb[5]-bb[4])+(rf[5]-rf[4])).reshape(9,*shape)
        wx = (w[1]-8*w[2]+8*w[3]-w[4])/(12*h)
        wy = (w[5]-8*w[6]+8*w[7]-w[8])/(12*h)
        lap = (-w[1]+16*w[2]-30*w[0]+16*w[3]-w[4]
               -w[5]+16*w[6]-30*w[0]+16*w[7]-w[8])/(12*h*h)
        physical = -initial[1].reshape(shape)*wx-initial[2].reshape(shape)*wy+p.nu*lap
        residual = rate-physical
        arrays[f'physical_rate_{h}'] = physical
        arrays[f'residual_{h}'] = residual
        report['local_pde'].append(dict(h=h, physical_rate_rms=rms(physical),
            discrete_rate_rms=rms(rate), residual_rms=rms(residual),
            viscous_rate_rms=rms(p.nu*lap),
            physical_fourth_difference_rms=rms(np.diff(physical,n=4,axis=0)),
            discrete_fourth_difference_rms=rms(np.diff(rate,n=4,axis=0)),
            residual_fourth_difference_rms=rms(np.diff(residual,n=4,axis=0))))
        print('INITIAL_PDE', json.dumps(report['local_pde'][-1]), flush=True)
        save()
    report['diagnostic_spacing_rate_change_rms'] = rms(
        arrays['physical_rate_0.002']-arrays['physical_rate_0.001'])
    report['rate_parts'] = dict(bspf_fourth_difference_rms=rms(np.diff(rb,n=4,axis=0)),
        rational_fourth_difference_rms=rms(np.diff(rr,n=4,axis=0)))
    load = p.explicit(a0)-p.linear@a0
    report['mass_solve_relative_residual'] = float(la.norm(p.mass@da-load)/la.norm(load))
    report['initial_linear_load_norm'] = float(la.norm(p.linear@a0+p.linear_lift))
    u,v,ux,uy,vx = [o@a0+l for o,l in zip(p.operators_fluid,p.lift_fields[1:])]
    bu,bv = p.operators_fluid[:2]; w = p.weights
    outu,outv = [o@a0+l for o,l in zip(p.out_ops,p.out_lift)]
    adv = bu.T@(w*(u*ux+v*uy))+bv.T@(w*(u*vx-v*ux))
    om = vx-uy
    rot = bu.T@(-w*v*om)+bv.T@(w*u*om)+p.out_ops[0].T@(
        p.out_weights*.5*(outu**2+outv**2))
    def dual(a):
        return float(np.sqrt(max(a@la.cho_solve(p.mass_factor,a),0)))
    report['advective_rotational_relative_dual_defect'] = dual(adv-rot)/dual(adv)
    report['minimum_initial_outlet_u'] = float(outu.min())
    # Smooth pressure with p=0 at the traction outlet; velocity test functions
    # vanish on all other boundaries, so its exact gradient load is zero.
    px,py = p.points.T; g = np.exp(-((px+.6)**2+py**2))
    gradx = -g-2*(px+.6)*(5-px)*g
    grady = -2*py*(5-px)*g
    gl = bu.T@(w*gradx)+bv.T@(w*grady)
    report['pressure_gradient_projected_velocity_relative_l2'] = dual(gl)/np.sqrt(
        np.sum(w*(gradx**2+grady**2)))
    ids = np.arange(0,len(p.points),max(1,len(p.points)//151))
    bb,rr = parts(p.transform@da,p.points[ids])
    resident = np.array([o[ids]@da for o in p.operators_fluid])
    report['rate_reconstruction_relative_difference'] = float(la.norm(
        (bb+rr)[1:]-resident)/la.norm(resident))
    hole,_ = p.arc.sample(256,offset=.371)
    bb,rr = parts(p.transform@da,hole)
    report['initial_acceleration_hole_speed'] = float(np.max(np.hypot(
        (bb+rr)[1],(bb+rr)[2])))
    print('ALGEBRA', json.dumps({k:v for k,v in report.items() if k not in
        ('parameters','local_pde')}), flush=True);save()
    report['first_steps'] = []
    hf = np.diff(arrays['residual_0.001'],n=4,axis=0)
    for dt in (.02,.01,.005,.001):
        a = p.stepper(dt).step(a0)
        bb,rr = parts(p.transform@(a-a0),points)
        increment_rate = ((bb[5]-bb[4])+(rr[5]-rr[4])).reshape(shape)/dt
        e = increment_rate-arrays['physical_rate_0.001']
        arrays[f'increment_rate_{dt}'] = increment_rate
        row = dict(dt=dt, incremental_rate_residual_rms=rms(e),
            gap_from_semidiscrete_rate_rms=rms(increment_rate-rate),
            residual_fourth_difference_rms=rms(np.diff(e,n=4,axis=0)),
            high_frequency_cosine_with_initial_spatial_residual=cosine(
                np.diff(e,n=4,axis=0),hf))
        report['first_steps'].append(row);print('FIRST_STEP',json.dumps(row),flush=True)
        save()
    # Same final physical time as baseline; compare fields, never reduced state
    # coordinates, which may change under an equivalent setup normalization.
    step = p.stepper(.01); a = a0.copy(); start = perf_counter()
    for k in range(100):
        a = step.step(a,k*.01)
        if (k+1)%25 == 0:
            print('HALF_DT', (k+1)*.01, flush=True)
    fine = p.grid(a,baseline['x'],baseline['y'])
    coarse = baseline['fields'][-1]
    mask = np.isfinite(fine['vorticity']) & np.isfinite(coarse[2])
    du,dv = (fine['u']-coarse[0])[mask],(fine['v']-coarse[1])[mask]
    dw = (fine['vorticity']-coarse[2])[mask]
    bx,by = baseline['x'],baseline['y']
    sy = (by[2:-2]>-.8)&(by[2:-2]<.8);sx=(bx>-.85)&(bx<-.45)
    def rough(a):return np.diff(a,n=4,axis=0)[sy][:,sx]
    rc,rf = rough(coarse[2]),rough(fine['vorticity'])
    report['same_time_step_control'] = dict(final_time=1,coarse_dt=.02,fine_dt=.01,
        velocity_relative_grid_change=float(np.sqrt(np.sum(du*du+dv*dv)/
            np.sum(fine['u'][mask]**2+fine['v'][mask]**2))),
        vorticity_relative_grid_change=float(la.norm(dw)/la.norm(fine['vorticity'][mask])),
        coarse_upstream_fourth_difference_rms=rms(rc),fine_upstream_fourth_difference_rms=rms(rf),
        upstream_fourth_difference_cosine=cosine(rc,rf),
        seconds=perf_counter()-start)
    np.savez_compressed(args.out/'half_dt_final.npz',x=bx,y=by,**fine)
    print('SAME_TIME',json.dumps(report['same_time_step_control']),flush=True);save()


if __name__ == '__main__':
    main()
