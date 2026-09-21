import numpy as np
import pytest
from bspf_models.kinetic.itg_statistics import heat_batch_means
from bspf_models.kinetic.itg_statistics import heat_uncertainty
from bspf_models.kinetic.itg_statistics import compare_heat_means


def test_batch_means_use_integrated_work_and_exact_duration():
    t=np.arange(1001.);work=4*(2*t+.3*np.sin(t))
    b=heat_batch_means(t,work,4.,start=200,end=1000,width=100)
    edges=np.arange(200,1001,100)
    expected=2+.3*np.diff(np.sin(edges))/100
    np.testing.assert_allclose(b['values'],expected,atol=3e-15)
    assert b['count']==8 and b['start']==200 and b['end']==1000
    with pytest.raises(ValueError):heat_batch_means(t,work,4.,start=200,end=1000,width=-1)


def test_uncertainty_reports_block_size_sensitivity():
    rng=np.random.default_rng(5);t=np.arange(1001.);q=2+rng.normal(size=1000)*.1
    w=np.r_[0,np.cumsum(4*q)]
    u=heat_uncertainty(t,w,4.,start=200,end=1000)
    assert [b['count'] for b in u['blocks']]==[32,16,8,4]
    assert u['selected']['count']>=8
    eligible=[b['half_width'] for b in u['blocks'] if b['count']>=8]
    assert u['selected']['half_width']==max(eligible)


def test_overlapping_wide_intervals_do_not_prove_equivalence():
    def u(mean,se):return {'selected':{'mean':mean,'standard_error':se,'count':16}}
    assert not compare_heat_means(u(1,.1),u(1,.1))['equivalent_within_tolerance']
    assert compare_heat_means(u(1.01,.001),u(1,.001))['equivalent_within_tolerance']
    assert not compare_heat_means(u(1.1,.001),u(1,.001))['equivalent_within_tolerance']
