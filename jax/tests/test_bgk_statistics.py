"""Do not mistake a flat energy trace with dying transport for saturation."""
from pathlib import Path
import importlib.util
import numpy as np
import pytest
pytest.importorskip('matplotlib')
spec=importlib.util.spec_from_file_location('bgk_saturation_example',Path(__file__).resolve().parents[2]/'examples/pde/bgk_itg_saturation.py')
example=importlib.util.module_from_spec(spec);spec.loader.exec_module(example)


@pytest.mark.parametrize('decay',[False,True])
def test_stationary_vs_decaying_transport(decay):
    t=np.arange(101.);d=np.tile([8,2,10,1,1],(len(t),1))
    data=dict(t=t,diagnostics=d,spectrum_ky=np.tile([5,5],(len(t),1)),
        spectrum_radial=np.tile([5,5],(len(t),1)),spectrum_velocity=np.full((len(t),2,2),2.))
    q=np.exp(-t/20) if decay else np.ones_like(t)
    work=80*(1-np.exp(-t/20)) if decay else 4*t
    data.update(work=np.stack([t*0,work,work],axis=1),rates=np.stack([t*0,4*q,4*q],axis=1),transport=np.stack([t*0,q],axis=1))
    report,_=example.analyze(data,25,4.)
    assert report['stationarity_screen_pass']==(not decay)
    assert report['budget_relative_max']==0
    blocks=report['blocks']
    assert blocks[0]['start']==25 and blocks[-1]['end']==100
    for i in range(3):assert blocks[i]['end']==blocks[i+1]['start']
    exact=(work[-1]-work[25])/4/(100-25)
    np.testing.assert_allclose(report['mean_heat_flux'],exact,atol=1e-15)
