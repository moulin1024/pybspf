"""Independent curved Taylor-Hood reference for frozen-force flow diagnostics.

Optional research dependencies: gmsh, meshio and scikit-fem. They are not core
package dependencies. No BSPF operators or spaces are used by this solver.
"""
from pathlib import Path
import gmsh
import meshio
import numpy as np
from scipy.sparse import bmat
from scipy.sparse.linalg import splu
from skfem import Basis, MeshTri, MeshTri2, ElementTriP3, ElementTriP2, ElementTriP4, BilinearForm, LinearForm, asm
from skfem.io import from_meshio
from skfem.models.poisson import mass, laplace


@BilinearForm
def divx(u,q,w): return u.grad[0]*q


@BilinearForm
def divy(u,q,w): return u.grad[1]*q


def make_mesh(path, scale):
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal',0)
        gmsh.model.add('frozen_force')
        box=gmsh.model.occ.addRectangle(-1,-1,0,6,2)
        hole=gmsh.model.occ.addDisk(.19,-.13,0,.31,.23)
        gmsh.model.occ.cut([(2,box)],[(2,hole)])
        gmsh.model.occ.synchronize()
        # Boundary layers in the Helmholtz response require finer wall spacing.
        curves=[tag for dim,tag in gmsh.model.getEntities(1)]
        dist=gmsh.model.mesh.field.add('Distance')
        gmsh.model.mesh.field.setNumbers(dist,'CurvesList',curves)
        gmsh.model.mesh.field.setNumber(dist,'Sampling',400)
        field=gmsh.model.mesh.field.add('Threshold')
        for key,val in [('InField',dist),('SizeMin',.025*scale),('SizeMax',.14*scale),
                        ('DistMin',.015),('DistMax',.25)]:
            gmsh.model.mesh.field.setNumber(field,key,val)
        gmsh.model.mesh.field.setAsBackgroundMesh(field)
        gmsh.option.setNumber('Mesh.MeshSizeFromPoints',0)
        gmsh.option.setNumber('Mesh.MeshSizeFromCurvature',0)
        gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary',0)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(2)
        gmsh.model.mesh.optimize('HighOrder')
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    mesh = from_meshio(meshio.read(path),force_meshio_type='triangle6')
    # P3 edge DOFs require globally ascending local vertex orientation. P2
    # geometry midpoints are indexed by globally sorted facets, unchanged here.
    return MeshTri2(mesh.p, np.sort(mesh.t, axis=0))


class FrozenForceFEM:
    def __init__(self, path, scale, force, intorder=8, degree=3):
        self.mesh=make_mesh(path,scale)
        velocity_element, pressure_element = {3: (ElementTriP3, ElementTriP2),
                                              4: (ElementTriP4, ElementTriP3)}[degree]
        self.basis=Basis(self.mesh,velocity_element(),intorder=intorder)
        pressure=Basis(self.mesh,pressure_element(),intorder=intorder)
        self.M=asm(mass,self.basis);self.K=asm(laplace,self.basis)
        self.Bx=asm(divx,self.basis,pressure);self.By=asm(divy,self.basis,pressure)
        self.n,self.np=self.basis.N,pressure.N
        boundary=self.mesh.boundary_facets()
        mids=self.mesh.p[:,self.mesh.facets[:,boundary]].mean(axis=1)
        fixed=boundary[abs(mids[0]-5)>1e-10]
        dofs=self.basis.get_dofs(facets=fixed).all()
        self.free=np.setdiff1d(np.arange(2*self.n+self.np),np.r_[dofs,self.n+dofs])
        coords=np.asarray(self.basis.global_coordinates())
        f=force(coords.reshape(2,-1).T).reshape(2,*coords.shape[1:])
        @LinearForm
        def load(v,w):return w.f*v
        self.rhs=np.r_[asm(load,self.basis,f=f[0]),asm(load,self.basis,f=f[1]),np.zeros(self.np)]
        # Linear triangles cover the exterior of the inscribed hole polygon;
        # final evaluation uses the actual curved inverse element map.
        self.linear=MeshTri(self.mesh.p[:,:self.mesh.nvertices],self.mesh.t)
        self.finder=self.linear.element_finder()
        self.info=dict(scale=scale,elements=int(self.mesh.nelements),velocity_dofs=int(2*self.n),
                       pressure_dofs=int(self.np),intorder=intorder,degree=degree)

    @classmethod
    def load_for_evaluation(cls, path, degree=4):
        """Read an existing reference mesh without remeshing or assembling a PDE.

        The matching saved mixed state must have the same polynomial degree.
        This reader is for evaluating frozen references at new quadrature nodes.
        """
        result = cls.__new__(cls)
        mesh = from_meshio(meshio.read(path), force_meshio_type='triangle6')
        result.mesh = MeshTri2(mesh.p, np.sort(mesh.t, axis=0))
        element = {3: ElementTriP3, 4: ElementTriP4}[degree]
        result.basis = Basis(result.mesh, element(), intorder=2*degree+2)
        result.n = result.basis.N
        result.linear = MeshTri(result.mesh.p[:, :result.mesh.nvertices], result.mesh.t)
        result.finder = result.linear.element_finder()
        return result

    def solve(self,alpha,viscosity):
        A=alpha*self.M+viscosity*self.K
        system=bmat([[A,None,-self.Bx.T],[None,A,-self.By.T],[-self.Bx,-self.By,None]],format='csc')
        interior=system[self.free][:,self.free]
        factor=splu(interior)
        state=np.zeros(system.shape[0]);state[self.free]=factor.solve(self.rhs[self.free])
        self.solve_residual=float(np.linalg.norm((system@state-self.rhs)[self.free])/np.linalg.norm(self.rhs[self.free]))
        return state

    def evaluate(self,state,points):
        """u,v,ux,uy,vx,vy, with analytic FE basis derivatives."""
        output=[]
        mapping=self.basis.mapping;elem=self.basis.elem
        for pts in np.array_split(points,max(1,int(np.ceil(len(points)/1500)))):
            cells=self.finder(*pts.T)
            local=mapping.invF(pts.T[:,:,None],tind=cells)
            # Inscribed polygon search can choose the adjacent element on a
            # curved interior edge; reject points outside actual mapped cell.
            tol=1e-7
            if np.min(local)<-tol or np.max(local.sum(axis=0))>1+tol:
                # Search neighboring cells using the curved map, including all
                # incident cells; only rare near-curved-edge probes enter here.
                bad=(local.min(axis=0).ravel()<-tol)|(local.sum(axis=0).ravel()>1+tol)
                for j in np.flatnonzero(bad):
                    vertices=self.mesh.t[:,cells[j]]
                    candidates=np.flatnonzero(np.any(np.isin(self.mesh.t,vertices),axis=0))
                    repeat=np.repeat(pts[j,:,None,None],len(candidates),axis=1)
                    q=mapping.invF(repeat,tind=candidates)
                    valid=(q.min(axis=0).ravel()>=-tol)&(q.sum(axis=0).ravel()<=1+tol)
                    if not valid.any():
                        raise ValueError(f'Probe outside curved reference mesh: {pts[j]}')
                    kk=np.flatnonzero(valid)[0];cells[j]=candidates[kk];local[:,j]=q[:,kk]
            values=np.zeros((6,len(pts)))
            for k in range(self.basis.Nbfun):
                field=elem.gbasis(mapping,local,k,tind=cells)[0]
                ids=self.basis.element_dofs[k,cells]
                cu,cv=state[ids],state[self.n+ids]
                values[0]+=cu*np.asarray(field).ravel();values[1]+=cv*np.asarray(field).ravel()
                values[2]+=cu*field.grad[0].ravel();values[3]+=cu*field.grad[1].ravel()
                values[4]+=cv*field.grad[0].ravel();values[5]+=cv*field.grad[1].ravel()
            output.append(values)
        return np.concatenate(output,axis=1)
