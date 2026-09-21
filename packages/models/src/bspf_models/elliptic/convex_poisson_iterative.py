"""Matrix-free convex BSPF least squares for resolution studies.

Same continuous H3/2-trace objective as ConvexPoissonPlan. Uses oversampled
masked tensor Gauss points, a separable H2 preconditioner, and LSMR rather
than dense SVD. Sampling and iterative convergence must be checked separately.
"""

import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import LinearOperator, lsmr
from scipy.special import roots_legendre

from bspf_models.elliptic.convex_poisson import ArcLengthBoundary
from bspf_models.elliptic.convex_poisson import trace_transform
from bspf_models.elliptic.convex_poisson import validate_convex
from bspf_models._numerics.trial_spaces import _stream_line
from bspf_models._numerics.trial_spaces import stream_evaluate_line


def trace_adjoint(values,length,count,order=1.5):
    modes=count//2+1
    frequency=2*np.pi*np.fft.rfftfreq(count,d=length/count)
    multiplicity=np.full(modes,2.);multiplicity[[0,-1]]=1
    scale=np.sqrt(length/count/multiplicity)*(1+frequency**2)**(order/2)
    return np.fft.irfft((values[:modes]+1j*values[modes:])*scale,n=count,norm="ortho")


class IterativeConvexPlan:
    def __init__(self,domain,*,nodes,half_width=1.2,sample_factor=3,boundary_count=None):
        validate_convex(domain)
        self.domain=domain;self.nodes=nodes
        self.line=_stream_line(np.linspace(-half_width,half_width,nodes),clamped=False,
                               dirichlet=False,endpoint_points=12,chebyshev_modes=12)
        q,w=roots_legendre(sample_factor*nodes+1)
        self.grid=half_width*q
        w=half_width*w
        self.b,_,self.h=stream_evaluate_line(self.line,self.grid)
        mask=np.zeros((len(q),len(q)),bool)
        for i,x in enumerate(self.grid):
            for lo,hi in domain.intersections(x):mask[i]|=(self.grid>lo)&(self.grid<hi)
        self.mask=mask
        xx,yy=np.meshgrid(self.grid,self.grid,indexing="ij")
        self.points=np.column_stack((xx[mask],yy[mask]))
        self.root_weights=np.sqrt((w[:,None]*w[None,:])[mask])
        self.arc=ArcLengthBoundary(domain)
        self.boundary_count=boundary_count or 2**int(np.ceil(np.log2(8*nodes)))
        self.boundary,_=self.arc.sample(self.boundary_count)
        self.bx=stream_evaluate_line(self.line,self.boundary[:,0])[0]
        self.by=stream_evaluate_line(self.line,self.boundary[:,1])[0]
        # J is the actual second-derivative Gram. A separable approximation to
        # the BOX H2 metric avoids an N^2 by N^2 dense Cholesky factor.
        k=np.diag(np.asarray(self.line.lam));j=np.asarray(self.line.bending)
        metric=np.eye(nodes)+2*k+j
        eigen,self.rotation=la.eigh((metric+metric.T)/2)
        if eigen.min()<=0:raise ValueError("Invalid one-dimensional H2 metric")
        root=np.sqrt(eigen)
        self.denominator=root[:,None]+root[None,:]
        self.operator=LinearOperator((len(self.points)+self.boundary_count+2,nodes**2),
                                     matvec=self.matvec,rmatvec=self.rmatvec,dtype=float)

    def decode(self,a):
        return self.rotation@(np.asarray(a).reshape(self.nodes,self.nodes)/self.denominator)@self.rotation.T

    def matvec(self,a):
        c=self.decode(a)
        lap=-(self.h@c@self.b.T+self.b@c@self.h.T)
        edge=np.sum((self.bx@c)*self.by,axis=1)
        return np.r_[self.root_weights*lap[self.mask],trace_transform(edge,self.arc.length)]

    def rmatvec(self,z):
        count=len(self.points)
        grid=np.zeros(self.mask.shape)
        grid[self.mask]=self.root_weights*z[:count]
        c=-(self.h.T@grid@self.b+self.b.T@grid@self.h)
        edge=trace_adjoint(z[count:],self.arc.length,self.boundary_count)
        c+=self.bx.T@(edge[:,None]*self.by)
        return ((self.rotation.T@c@self.rotation)/self.denominator).ravel()

    def solve(self,forcing,boundary,*,tolerance=1e-12,maxiter=6000,progress=False):
        rhs=np.r_[self.root_weights*forcing(self.points),
                  trace_transform(boundary(self.boundary),self.arc.length)]
        result=lsmr(self.operator,rhs,atol=tolerance,btol=tolerance,
                    conlim=1e14,maxiter=maxiter,show=progress)
        residual=self.operator@result[0]-rhs
        return self.decode(result[0]).ravel(),dict(stop_code=int(result[1]),iterations=int(result[2]),
                  relative_residual=float(la.norm(residual)/la.norm(rhs)),
                  normal_residual=float(result[4]),condition_estimate=float(result[6]),
                  scaled_coefficient_norm=float(result[7]),tolerance=tolerance,
                  volume_samples=len(self.points),boundary_count=self.boundary_count,
                  physical_data_only=True,backend="matrix_free_lsmr")
