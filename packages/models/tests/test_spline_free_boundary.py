"""Independent vacuum checks for the axisymmetric free-space Green operator."""
import numpy as np
import pytest
from bspf_models.plasma.spline_free_boundary import ring_flux,FilamentCoils


def test_ring_green_reciprocity_superposition_and_blocks():
    x=np.array([[2.,.2],[3.,-.4],[.01,100.]])
    y=np.array([[4.,2.],[5.,-2.]])
    current=np.array([1.2,-.3])
    full=ring_flux(x,y,current,mu0=1.)
    np.testing.assert_allclose(full,ring_flux(x,y,current,mu0=1.,block_size=1,source_block=1),rtol=1e-14,atol=1e-15)
    reverse=np.array([ring_flux(y,p[None,:],[1.],mu0=1.) for p in x])
    np.testing.assert_allclose(full,reverse@current,rtol=1e-13,atol=1e-15)
    # On-axis asymptotic: psi/R^2 -> mu0 I a^2 / [4(a^2+z^2)^(3/2)].
    small=ring_flux(np.array([[1e-5,0.]]),np.array([[3.,2.]]),[1.],mu0=1.)[0]
    np.testing.assert_allclose(small/1e-10,9/(4*13**1.5),rtol=1e-9)


def test_filament_vacuum_gs_operator():
    coils=FilamentCoils(np.array([[5.,2.],[5.,-2.]]),np.array([-.2,-.2]),mu0=1.)
    x=np.array([[3.,0.],[2.3,.4]])
    h=1e-3
    center=coils.flux(x);lap=np.zeros(len(x));radial=None
    for i,e in enumerate(np.eye(2)):
        mm,m,p,pp=[coils.flux(x+s*h*e) for s in (-2,-1,1,2)]
        lap+=(-pp+16*p-30*center+16*m-mm)/(12*h*h)
        if i==0:radial=(mm-8*m+8*p-pp)/(12*h)
    np.testing.assert_allclose(lap-radial/x[:,0],0.,atol=2e-8)
    with pytest.raises(ValueError,match='singular'):
        coils.flux(coils.positions)
    with pytest.raises(ValueError,match='R>0'):
        coils.flux(np.array([[0.,0.]]))
