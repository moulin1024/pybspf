"""Unfiltered channel control for the experimental complete curl-residual form."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import sys

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jl
import numpy as np

from bspf_jax.immersed_flow import ImmersedFlowPlan
from bspf_jax.immersed_flow_gpu import _explicit, _step
from curl_residual import assemble, advance, explicit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from immersed_channel_flow import independent_checks


def rms(a): return float(np.sqrt(np.mean(np.asarray(a)**2)))
def metrics(a): return dict(rms=rms(a), d4y=rms(np.diff(a, n=4, axis=0)), maximum=float(np.max(np.abs(a))))


def json_safe(value):
    if isinstance(value, dict): return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list): return [json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strength',type=float,default=1.)
    parser.add_argument('--boundary-compatible',action='store_true')
    parser.add_argument('--time',type=float,default=2.)
    parser.add_argument('--dt',type=float,default=.01)
    parser.add_argument('--quadrature',type=float,default=2.5)
    parser.add_argument('--out',type=Path,default=Path('build/immersed_flow/re200_ripple_study/curl_residual/strength1'))
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    jax.config.update('jax_enable_x64',True);device=jax.devices('gpu')[0]
    p=ImmersedFlowPlan(assembly_device=device,basis_precision='float64',nx=73,ny=33,reynolds=200,
                      wall_method='rational',rational_preprocessing=False,quadrature_factor=args.quadrature)
    step=p.stepper(args.dt,device=device)
    d,setup=assemble(p,step,args.strength,boundary_compatible=args.boundary_compatible)
    print('SETUP',json.dumps(setup),flush=True)
    # Independent output operators, unchanged basis/lift and physical walls.
    x=np.linspace(-.85,-.35,61);y=np.linspace(-.8,.8,161)
    xx,yy=np.meshgrid(x,y);points=np.column_stack((xx.ravel(),yy.ravel()))
    ops,base=p.operators(points,with_base=True,device_output=True)
    transform=jax.device_put(p.transform,device)
    curl=(ops[5]-ops[4]) @ transform
    omega_base=base[5]-base[4]
    def omega(state):return np.asarray(curl @ state+omega_base).reshape(xx.shape)
    a0=step.initial_state
    new_stokes=jnp.linalg.solve(d['augmented_linear'],-d['augmented_lift'])
    initial_relative=float(jnp.sqrt((new_stokes-a0) @ step.data['mass'] @ (new_stokes-a0)))
    # Hold the original physical initial state fixed in this first control.
    rhs=explicit(d,a0)-d['augmented_linear'] @ a0
    derivative=jl.cho_solve(d['augmented_mass_factor'],rhs)
    rate=np.asarray(curl @ derivative).reshape(xx.shape)
    old_rhs=_explicit(step.data,a0)-step.data['linear'] @ a0
    old_derivative=jl.cho_solve((step.data['mass_factor'],True),old_rhs)
    old_rate=np.asarray(curl @ old_derivative).reshape(xx.shape)
    original=np.load('build/immersed_flow/re200_ripple_study/source_audit_fields.npz')
    exact_rate=original['pde_rhs_0.0_0.001']
    arrays=dict(x=x,y=y,initial_omega=omega(a0),initial_rate=rate,original_initial_rate=old_rate,local_pde_rate=exact_rate)
    report=dict(configuration=vars(args)|dict(out=str(args.out)),setup=setup,
                old_vs_new_stokes_velocity_l2=initial_relative,
                initial_rate=metrics(rate),original_initial_rate=metrics(old_rate),
                initial_local_pde_residual=metrics(rate-exact_rate),
                original_initial_local_pde_residual=metrics(old_rate-exact_rate),
                mass_relative_residual=float(jnp.linalg.norm(d['augmented_mass'] @ derivative-rhs)/jnp.linalg.norm(rhs)),
                history=[])
    if args.strength == 0.:
        reference=_step(step.data,a0,jax.device_put(0.,device),step.dt)
        candidate=advance(d,a0,step.dt,steps=1)
        report['zero_strength_one_step_max_error']=float(jnp.max(jnp.abs(reference-candidate)))
    print('INITIAL',json.dumps(report),flush=True)
    batch=10
    with jax.transfer_guard('disallow'):
        warm=advance(d,a0,step.dt,steps=batch);jax.block_until_ready(warm)
    state=a0;count=round(args.time/args.dt);done=0;start=perf_counter()
    while done<count:
        n=min(batch,count-done)
        state=advance(d,state,step.dt,steps=n);jax.block_until_ready(state)
        done+=n
        stats=jax.device_get(step.diagnostics(state))
        record=dict(time=done*args.dt,kinetic_energy=float(stats['kinetic_energy']),max_speed=float(stats['max_speed']),
                    flux_out=float(stats['flux_out']),vorticity=metrics(omega(state)))
        report['history'].append(record)
        print('STEP',json.dumps(record),flush=True)
        if not np.isfinite(record['kinetic_energy']) or record['max_speed']>100:
            report['failed']='Non-finite or runaway evolution';break
        if done*args.dt in (.1,.5,1.,2.,5.,10.,20.):arrays[f'omega_t{done*args.dt:g}']=omega(state)
    report['evolution_seconds']=perf_counter()-start
    report['warm_ms_per_step']=1000*report['evolution_seconds']/done
    if 'failed' not in report:
        report['independent_checks']=independent_checks(p,np.asarray(state))
    arrays['final_coefficients']=np.asarray(transform @ state+jax.device_put(p.lift_coefficients,device))
    arrays['final_omega']=omega(state)
    np.savez_compressed(args.out/'fields.npz',**arrays)
    (args.out/'report.json').write_text(json.dumps(json_safe(report),indent=2,allow_nan=False)+'\n')
    print('RESULT',json.dumps(report),flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,key in zip(axes,('initial_rate','original_initial_rate','final_omega')):
        field=arrays[key]
        if not np.all(np.isfinite(field)):
            ax.text(.5,.5,'Non-finite field: run failed',ha='center',transform=ax.transAxes)
            ax.set(title=key)
            continue
        limit=np.max(abs(field))
        im=ax.pcolormesh(x,y,field,cmap='RdBu_r',vmin=-limit,vmax=limit,shading='auto')
        fig.colorbar(im,ax=ax);ax.set(title=key,xlabel='x',ylabel='y')
    fig.savefig(args.out/'control.png',dpi=150)


if __name__=='__main__':main()
