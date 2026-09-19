"""Reference-free Stokes vorticity test: a harmonic field has the mean-value property.

All evaluation is on GPU. Circles stay inside a single patch, away from walls
and interfaces. Comparing 128 and 256 angular samples controls circle integration.
"""
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from fields import evaluate
from bspf_jax.mapped_poisson import _axis
from bspf_jax.mapped_navier_stokes import channel_streamfunction
from bspf_jax.immersed_poisson import EllipticHole


def rms(a):return float(np.sqrt(np.mean(np.asarray(a)**2)))


def main():
    jax.config.update('jax_enable_x64',True);device=jax.devices('gpu')[0]
    root=Path('build/mapped_ripple')
    hole=EllipticHole();bounds=(-1.,3.,-1.,1.)
    geometry=jax.device_put((np.array(((-1.,-1.),(3.,-1.),(3.,1.),(-1.,1.))),np.array(hole.center),np.array(hole.axes)),device)
    normal_lift=channel_streamfunction(bounds,hole)
    cx,cy=hole.center;a,b=hole.axes
    cutoff_radius=.9*min((cx+1)/a,(3-cx)/a,(cy+1)/b,(1-cy)/b);constant=cy-cy**3/3
    def c4_lift(z):
        q=jnp.sqrt(((z[0]-cx)/a)**2+((z[1]-cy)/b)**2)
        s=jnp.clip((q-1)/(cutoff_radius-1),0.,1.)
        cutoff=s**5*(126+s*(-420+s*(540+s*(-315+70*s))))
        return constant+cutoff*(z[1]-z[1]**3/3-constant)
    centers=np.stack(np.meshgrid(np.linspace(-.8,-.5,17),np.linspace(-.25,.2,17),indexing='ij'),axis=-1).reshape(-1,2)
    rows=[]
    for case in ('baseline','quadrature10','lift_c4','radial24','tangent24','degree4','degree4_c4'):
        folder=root/case
        report=json.loads((folder/'report.json').read_text());degree=report['degree']
        local=jax.device_put(np.load(folder/'stokes.npz')['local'],device)
        knots=jax.device_put(tuple(_axis(n,degree,degree+3)[0] for n in report['elements']),device)
        lift=c4_lift if report.get('lift')=='c4' else normal_lift
        def values(points):
            result=[]
            for start in range(0,len(points),4096):
                batch=points[start:start+4096]
                padded=np.pad(batch,((0,4096-len(batch)),(0,0)),mode='edge')
                data=evaluate(jax.device_put(padded,device),local,geometry,knots,degree=degree,lift=lift)
                result.append(np.asarray(data['vorticity'])[:len(batch)])
            return np.concatenate(result)
        source=np.load(folder/'fields.npz')
        reconstruction=values(source['points'].reshape(-1,2)).reshape(source['vorticity'].shape)
        parity=float(np.max(np.abs(reconstruction-source['vorticity'])))
        if parity > 1e-9:
            raise RuntimeError(f'{case}: saved-field reconstruction mismatch {parity}')
        center_values=values(centers)
        tests=[]
        for radius in (.025,.05,.075):
            angle=np.arange(256)*2*np.pi/256
            ring=centers[:,None,:]+radius*np.stack((np.cos(angle),np.sin(angle)),axis=-1)[None]
            ring_values=values(ring.reshape(-1,2)).reshape(len(centers),256)
            average=ring_values.mean(axis=1)
            defect=center_values-average
            angular_error=average-ring_values[:,::2].mean(axis=1)
            tests.append(dict(radius=radius,mean_value_defect_rms=rms(defect),max_defect=float(np.max(np.abs(defect))),
                              angular_quadrature_change_rms=rms(angular_error),vorticity_rms=rms(center_values)))
        row=dict(case=case,reconstruction_max_error=parity,elements=report['elements'],degree=degree,tests=tests)
        rows.append(row);print(json.dumps(row),flush=True)
        (root/'mean_value.json').write_text(json.dumps(dict(center_bounds=[-.8,-.5,-.25,.2],cases=rows),indent=2)+'\n')


if __name__=='__main__':main()
