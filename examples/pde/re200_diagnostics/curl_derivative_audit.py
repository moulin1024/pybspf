"""Validate experimental analytic BSPF derivatives before evolving a new form."""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from bspf_jax.immersed_flow import ImmersedFlowPlan
from curl_residual import line_jets, assemble

jax.config.update('jax_enable_x64', True)
device = jax.devices('gpu')[0]
p = ImmersedFlowPlan(assembly_device=device, basis_precision='float64', nx=73, ny=33,
                     reynolds=200, wall_method='rational', rational_preprocessing=False)
report = dict(lines=[])
for axis in (0, 1):
    low, high = p.bounds[:2] if axis == 0 else (-p.bounds[2], p.bounds[2])
    points = np.linspace(low+.13, high-.17, 71)
    jets = [np.asarray(v) for v in line_jets(p, axis, points)]
    reference = p._line_values(axis, points)
    parity = [float(np.linalg.norm(a-b)/np.linalg.norm(b)) for a, b in zip(jets, reference)]
    checks = []
    for h in (.0002, .0001):
        second = [p._line_values(axis, points+k*h)[2] for k in (-2, -1, 0, 1, 2)]
        third = (second[0]-8*second[1]+8*second[3]-second[4])/(12*h)
        fourth = (-second[0]+16*second[1]-30*second[2]+16*second[3]-second[4])/(12*h*h)
        checks.append(dict(h=h, third_relative=float(np.linalg.norm(third-jets[3])/np.linalg.norm(jets[3])),
                           fourth_relative=float(np.linalg.norm(fourth-jets[4])/np.linalg.norm(jets[4]))))
    row = dict(axis=axis, production_parity=parity, finite_difference=checks)
    print('LINE', json.dumps(row), flush=True)
    report['lines'].append(row)
    assert max(parity) < 1e-7
    assert checks[0]['third_relative'] < 1e-5
    assert checks[0]['fourth_relative'] < 1e-4
step = p.stepper(.01, device=device)
d, timings = assemble(p, step)
report.update(timings)
# Low-order strong derivative parity with finite differences of production curl
# on a disjoint physical probe; rational gradient checked via separate evaluations.
from curl_residual import _rational_curl_rows
xy = np.column_stack((np.linspace(-.82, -.42, 31), np.linspace(-.6, .7, 31)))
z = xy[:, 0]+1j*xy[:, 1]-p.rational.center
z = jax.device_put(z, device)
rows = _rational_curl_rows(z, p.rational.basis.evaluate_gpu(z, device))
coeff = p.rational_lift
analytic = [np.asarray(row @ jax.device_put(coeff, device)) for row in rows]
checks=[]
for h in (.0002, .0001):
    errors=[]
    for axis in (0, 1):
        shift=np.eye(2)[axis]*h
        fields=[p.rational.evaluate(xy+k*shift, coeff, device=device) for k in (-2,-1,1,2)]
        omega=[v[5]-v[4] for v in fields]
        fd=(omega[0]-8*omega[1]+8*omega[2]-omega[3])/(12*h)
        errors.append(float(np.linalg.norm(fd-analytic[axis])/np.linalg.norm(analytic[axis])))
    checks.append(dict(h=h, relative_errors=errors))
assert max(checks[-1]['relative_errors']) < 1e-7
report['rational_gradient_fd']=checks
report['augmented_factor_finite']=bool(jnp.all(jnp.isfinite(d['augmented_factor'][0])))
report['mass_symmetry_relative']=float(jnp.linalg.norm(d['augmented_mass']-d['augmented_mass'].T)/jnp.linalg.norm(d['augmented_mass']))
print('RESULT',json.dumps(report),flush=True)
out=Path('build/immersed_flow/re200_ripple_study/curl_residual');out.mkdir(parents=True,exist_ok=True)
(out/'derivative_checks.json').write_text(json.dumps(report,indent=2)+'\n')
