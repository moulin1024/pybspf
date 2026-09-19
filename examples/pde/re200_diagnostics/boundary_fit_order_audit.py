"""Separate trace compression, fit truncation, and evaluation errors at Re200."""
import json
from pathlib import Path
import jax
import numpy as np
import bspf_jax.immersed_flow as flow
from bspf_jax.rational_stokes import RationalStokesExtension
jax.config.update('jax_enable_x64',True)
import bspf_jax._gpu_linalg as gl
original_svd=gl.gpu_svd
trace_svd=None
def capture_svd(matrix,**kw):
 global trace_svd
 result=original_svd(matrix,**kw)
 if matrix.shape==(1600,5795):trace_svd=result
 return result
gl.gpu_svd=capture_svd
class GeometryReady(Exception):pass
def stop_volume(*args,**kwargs):raise GeometryReady
# Stop only this diagnostic constructor before volume assembly. The complete
# geometry, original BSPF basis and rational trace compression are initialized.
flow.channel_quadrature=stop_volume
p=flow.ImmersedFlowPlan.__new__(flow.ImmersedFlowPlan)
try:
 flow.ImmersedFlowPlan.__init__(p,assembly_device=jax.devices('gpu')[0],basis_precision='float64',nx=97,ny=65,reynolds=200,wall_method='rational')
except GeometryReady:pass
coeff=np.load('build/immersed_flow/re200_gpu_97x65/fields.npz')['coefficients']
ext=p.rational
p.rational=None
# Evaluate original BSPF physical velocity rather than normalized trace modes.
def bare(points):
 ops=p.operators(points)
 base=flow.channel_lift(points,p.bounds[2],p.peak)
 return tuple(o@coeff+b for o,b in zip(ops,base))
hole,_=p.arc.sample(512,offset=.371)
x=np.linspace(p.bounds[0],p.bounds[1],197);y=np.linspace(-p.bounds[2],p.bounds[2],157)
outer=np.vstack((np.column_stack((x,np.full_like(x,-1))),np.column_stack((x,np.full_like(x,1))),np.column_stack((np.full_like(y,p.bounds[0]),y))))
probes=np.vstack((hole,outer));physical=bare(probes);reference=flow.channel_lift(probes,p.bounds[2],p.peak)
def measure(extension,rc):
 r=extension.evaluate(probes,rc,device=jax.devices('gpu')[0]);u=physical[1]+r[1];v=physical[2]+r[2]
 return dict(hole=float(np.max(np.hypot(u[:len(hole)],v[:len(hole)]))),outer=float(np.max(np.hypot(u[len(hole):]-reference[1][len(hole):],v[len(hole):]-reference[2][len(hole):]))),coefficient_norm=float(np.linalg.norm(rc)))
report=[]
rc=p.rational_modes@(p.rational_map@coeff)+p.rational_lift
report.append(dict(case='compressed baseline',**measure(ext,rc)))
print(report[-1],flush=True)
u,s,vh=trace_svd;keep=s>1e-13*s[0]
weighted_modes=ext.response(u[:,keep]*s[keep])
weighted_map=-vh[keep]/p.scale
rc=weighted_modes@(weighted_map@coeff)+p.rational_lift
report.append(dict(case='weighted compressed baseline',**measure(ext,rc)))
print(report[-1],flush=True)
for label,options in [('direct baseline',None),('higher degree 160',dict(degree=160,laurent=128,samples=1600))]:
 e=ext if options is None else RationalStokesExtension(p.bounds,p.hole,assembly_device=jax.devices('gpu')[0],**options)
 target=-np.concatenate(bare(e.hole_points)[1:3]);response=e.response(target)
 report.append(dict(case=label,**measure(e,response),extension=e.info))
 print(report[-1],flush=True)
Path('build/immersed_flow/re200_ripple_study/boundary_fit_audit.json').write_text(json.dumps(report,indent=2)+'\n')
