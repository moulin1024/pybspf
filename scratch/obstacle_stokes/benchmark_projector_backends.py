"""Serial paired checkpoint replays: identical space, state, factors and audits."""
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from thin_airfoil_structural import boundary
from bspf_models.fluids.embedded_navier_stokes import plan_embedded_navier_stokes2d, geometry_prior_options

records=[]
for degree in (5,7):
    tick=perf_counter()
    p=plan_embedded_navier_stokes2d(boundary,dt=1e-4,viscosity=.002,cells=7,
        degree=degree,order=64,convection_order=48,center=(0.,0.),axes=(1.,.1),
        edges=[np.array([-60,-3,-1.2,-.4,.4,1.2,3,60]),np.array([-40,-2,-.2,-.05,.05,.2,2,40])],
        outflow=False,constraint_backend='implicit_qr',linear_backend='host_sparse',
        constraint_rank_tolerance=1e-9,wall_enforcement='constraint',projector_backend='array',
        **geometry_prior_options(degree))
    setup=perf_counter()-tick
    print('SETUP',degree,setup,flush=True)
    array=p.projector;implicit=p.spatial.projector
    try:
        with np.load(f'build/obstacle_stokes/ns/thin_airfoil_structural_p{degree}_noslip.npz') as z:
            initial=p._State(**{k:z[k].item() if z[k].ndim==0 else z[k] for k in p._State._fields})
        p.initialize(initial.coefficients,time=initial.time)
        trials={'implicit_qr':[],'array':[]};final={};iteration_counts={}
        for trial in range(3):
            order=['implicit_qr','array'] if trial%2==0 else ['array','implicit_qr']
            for name in order:
                p.projector=implicit if name=='implicit_qr' else array
                p.step(initial) # warm up identical operators before timing
                state=initial;times=[];iterations=[]
                for _ in range(20):
                    tick=perf_counter();state,d=p.step(state);times.append(perf_counter()-tick)
                    assert d.valid
                    iterations.append(p.last_iterations)
                trials[name].append(float(np.median(times)))
                final[name]=state;iteration_counts[name]=iterations
                print('TRIAL',degree,trial,name,trials[name][-1],flush=True)
        difference=final['array'].coefficients-final['implicit_qr'].coefficients
        n=2*p.nv;velocity_difference=float(np.sqrt(max(difference[:n]@(p.mass@difference[:n]),0)))
        relative_difference=velocity_difference/np.sqrt(final['implicit_qr'].coefficients[:n]@(p.mass@final['implicit_qr'].coefficients[:n]))
        assert relative_difference<1e-10
        r=dict(degree=degree,setup_seconds=setup,projector_export_seconds=p.info['array_projector_setup_seconds'],
            projector_bytes=array.bytes,trial_medians=trials,iterations=iteration_counts,
            speedup=float(np.median(trials['implicit_qr'])/np.median(trials['array'])),
            final_velocity_relative_L2_difference=float(relative_difference),
            final_wall_slip=p.wall_slip_error(final['array'].coefficients),
            steps_per_trial=20,trials_per_backend=3,info=p.info)
        records.append(r)
        Path('build/obstacle_stokes/ns/projector_backend_comparison.json').write_text(json.dumps(records,indent=2))
        print('RESULT',degree,r['speedup'],relative_difference,flush=True)
    finally:
        p.projector=array;p.close()
