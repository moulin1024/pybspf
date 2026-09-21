"""Full spatial MMS operator validation through 129^3 with streamed velocity blocks.

All physical velocity quadrature nodes are retained. The direct path applies
production RHS to velocity blocks. The separable MMS path applies the same RHS
to spatial basis functions and contracts all velocity nodes by weighted QR.
This exact MMS-specific factorization is not a general-state performance test.
"""
from pathlib import Path
from dataclasses import replace
import argparse
import gc
import json
import math
import resource
import subprocess
import sys
import time
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from bspf_jax.gyrokinetic_slab import (plan_slab_gk,slab_gk_solve_charge,
    slab_gk_rhs_with_field_hat)
from bspf_jax.gyrokinetic_mms import plan_slab_mms


def fft(a): return jnp.fft.fftn(a,axes=(0,1,2))
def ifft(a): return jnp.fft.ifftn(a,axes=(0,1,2)).real


def prepare(p,*,t=.17,r=.3):
    m=plan_slab_mms(p,r=r)
    th1=m.theta1[:,0:1,:]-.9*t; th2=m.theta2[0:1,:,:]+.7*t
    q1=jnp.exp(1j*th1);q2=jnp.exp(1j*th2)
    A=.03*(1+.2*jnp.sin(.8*t));B=.025*(1+.15*jnp.cos(.6*t))
    g1=A*(q1/(1-r*q1)).real;g2=B*(q2/(1-r*q2)).imag
    z1=A*(1j*q1/(1-r*q1)**2).real
    z2=B*(1j*q2/(1-r*q2)**2).imag
    n=m.harmonics[:,None,None,None]
    co1=jnp.cos(n*th1);si2=jnp.sin(n*th2)
    phi=A*jnp.einsum('n,nxyz->xyz',m.coefficient1,co1)+B*jnp.einsum('n,nxyz->xyz',m.coefficient2,si2)
    d1=-A*(m.coefficient1*m.harmonics)[:,None,None,None]*jnp.sin(n*th1)
    d2=B*(m.coefficient2*m.harmonics)[:,None,None,None]*jnp.cos(n*th2)
    h1=m.h1[0,0,0];h2=m.h2[0,0,0]
    # Discrete quadrature, NOT continuum analytic moments, for numerical field.
    w1=jnp.sum(p.weights*h1,axis=0);w2=jnp.sum(p.weights*h2,axis=0)
    shape=(p.x.size,p.y.size,p.z.size)
    charge=fft(jnp.broadcast_to(g1,shape))*jnp.einsum('ijm,m->ij',p.j0,w1)[:,:,None]+fft(jnp.broadcast_to(g2,shape))*jnp.einsum('ijm,m->ij',p.j0,w2)[:,:,None]
    ph=slab_gk_solve_charge(p,charge)
    return m,(g1,g2,z1,z2),d1,d2,phi,ph


@jax.jit
def block_errors(p,spatial,h1,h2,psi1p,psi2p,phi_hat):
    g1,g2,z1,z2=spatial
    lift=lambda a:a[:,:,:,None,None]
    h1=h1[None,None,None,:,:];h2=h2[None,None,None,:,:]
    g=lift(g1)*h1+lift(g2)*h2
    gh=fft(g)*p.mask[:,:,:,None,None]
    gp=ifft(gh)
    numerical=slab_gk_rhs_with_field_hat(p,gh,phi_hat)
    # S-g_t is formed analytically before projection, by linearity of P.
    streaming=p.v[None,None,None,:,None]*(lift(z1)*h1-lift(z2)*h2+(psi1p-psi2p)[:,:,:,None,:])
    bracket=2*(psi1p[:,:,:,None,:]*lift(z2)*h2-psi2p[:,:,:,None,:]*lift(z1)*h1)
    defect=numerical+ifft(fft(streaming+bracket)*p.mask[:,:,:,None,None])
    average=lambda a:jnp.mean(jnp.sum(a*a*p.weights,axis=(-2,-1)))
    return jnp.stack((average(gp-g),average(defect)))


@jax.jit
def separable_block_errors(p,spatial,psi1p,psi2p,phi_hat,projection_factor,residual_factor):
    """Algebraically exact MMS factorization; production RHS evaluates bases.

    At fixed phi the nonlinear bracket is linear in g, and parallel streaming
    is linear in v. All physical velocity nodes enter the weighted QR factors.
    This optimization is valid for this separable MMS, not arbitrary states.
    """
    g1,g2,z1,z2=spatial
    shape=p.mask.shape
    a=jnp.broadcast_to(g1,shape);b=jnp.broadcast_to(g2,shape)
    bases=jnp.stack((a,b,jnp.zeros_like(a)),axis=-1)[:,:,:,:,None]
    gh=fft(bases)*p.mask[:,:,:,None,None]
    # First two columns isolate nonlinear advection, third isolates -psi_z.
    bp=replace(p,v=jnp.array([0.,0.,1.]),weights=jnp.zeros((3,1)))
    rhs=slab_gk_rhs_with_field_hat(bp,gh,phi_hat)[:,:,:,:,0]
    parallel=ifft(-1j*p.kz[None,None,:,None]*gh[:,:,:,:2,0])
    p1=psi1p[:,:,:,0];p2=psi2p[:,:,:,0]
    source=jnp.stack((jnp.broadcast_to(-2*p2*z1,shape),
        jnp.broadcast_to(2*p1*z2,shape),jnp.broadcast_to(z1,shape),
        jnp.broadcast_to(-z2,shape),jnp.broadcast_to(p1-p2,shape)),axis=-1)
    coefficients=jnp.concatenate((rhs[:,:,:,:2],parallel,rhs[:,:,:,2:3]),axis=-1)
    coefficients+=ifft(fft(source)*p.mask[:,:,:,None])
    difference=ifft(gh[:,:,:,:2,0])-bases[:,:,:,:2,0]
    # QR avoids cancellation or negative norms from explicitly formed Gram matrices.
    proj=jnp.einsum('xyzj,ij->xyzi',difference,projection_factor)
    defect=jnp.einsum('xyzj,ij->xyzi',coefficients,residual_factor)
    return jnp.array([jnp.mean(jnp.sum(proj*proj,axis=-1)),
                      jnp.mean(jnp.sum(defect*defect,axis=-1))])


def run(n,*,n_v=24,n_mu=40,mu_block=1,v_block=None,method='direct',verbose=True):
    started=time.perf_counter()
    v_block=n_v if v_block is None else v_block
    if v_block<1 or mu_block<1: raise ValueError("block sizes must be positive")
    p=plan_slab_gk((n,n,n),n_v=n_v,n_mu=n_mu,rho=.8)
    m,spatial,d1,d2,phi,ph=prepare(p)
    field_error=float(jnp.sqrt(jnp.mean((ifft(ph)-phi)**2)))
    total=np.zeros(2)
    if method not in ('direct','separable'): raise ValueError('unknown method')
    if method=='separable' and mu_block!=1: raise ValueError('separable MMS requires mu_block=1')
    wh=np.asarray(p.weights); h1all=np.asarray(m.h1[0,0,0]); h2all=np.asarray(m.h2[0,0,0]); va=np.asarray(p.v)
    for start in range(0,n_mu,mu_block):
        end=min(start+mu_block,n_mu)
        psi1p=jnp.einsum('nxyz,nm->xyzm',d1,m.j1[:,start:end])
        psi2p=jnp.einsum('nxyz,nm->xyzm',d2,m.j2[:,start:end])
        if method=='separable':
            h1=h1all[:,start];h2=h2all[:,start]
            basis=np.stack((h1,h2,va*h1,va*h2,va),axis=-1)
            weighted=np.sqrt(wh[:,start,None])*basis
            factor=np.linalg.qr(weighted,mode='reduced')[1]
            projfactor=np.linalg.qr(weighted[:,:2],mode='reduced')[1]
            bp=replace(p,mu=p.mu[start:end],weights=p.weights[:,start:end],j0=p.j0[:,:,start:end])
            values=separable_block_errors(bp,spatial,psi1p,psi2p,ph,jnp.asarray(projfactor),jnp.asarray(factor))
            total+=np.asarray(values)
        else:
            for vs in range(0,n_v,v_block):
                ve=min(vs+v_block,n_v)
                bp=replace(p,v=p.v[vs:ve],mu=p.mu[start:end],weights=p.weights[vs:ve,start:end],j0=p.j0[:,:,start:end])
                values=block_errors(bp,spatial,m.h1[0,0,0,vs:ve,start:end],m.h2[0,0,0,vs:ve,start:end],psi1p,psi2p,ph)
                total+=np.asarray(values)
        if verbose and (start==0 or end==n_mu or end%10==0):
            print(f'N={n}: mu {end}/{n_mu}, elapsed {time.perf_counter()-started:.1f}s',flush=True)
        del bp,psi1p,psi2p,values
    cutoff=math.ceil(n/3)-1
    r=.3;t=.17;A=.03*(1+.2*math.sin(.8*t));B=.025*(1+.15*math.cos(.6*t))
    K=lambda a,s,d:((1+4*a)**(-.5)+d*d*(1+4*a)**(-1.5))/(1+2*s)
    tail=math.sqrt((A*A*K(.4,.5,.2)*r**(2*cutoff)+B*B*K(.7,1.2,-.15)*r**(2*(cutoff//2)))/(2*(1-r*r)))
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform!='darwin': peak*=1024
    return dict(n=n,n_v=n_v,n_mu=n_mu,mu_block=mu_block,v_block=v_block,method=method,
        projection_error=float(np.sqrt(total[0])),forced_rhs_defect=float(np.sqrt(total[1])),
        phi_error=field_error,analytic_continuous_projection_tail=tail,
        elapsed_seconds=time.perf_counter()-started,peak_rss_bytes=peak,
        phase_space_dofs=n**3*n_v*n_mu,full_state_bytes=n**3*n_v*n_mu*8,
        scope='Complete velocity quadrature, streamed production nonlinear RHS; semi-discrete MMS at t=0.17')


def render(rows,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    ns=[a['n'] for a in rows]
    for key,label in [('projection_error','measured projection'),('analytic_continuous_projection_tail','analytic spectrum tail')]:
        ax[0].semilogy(ns,[a[key] for a in rows],'o-' if key=='projection_error' else '--',label=label)
    ax[0].legend();ax[0].set(title='Distribution projection',xlabel='N per spatial axis',ylabel='weighted L2 error')
    for key,label in [('forced_rhs_defect','forced PDE defect'),('phi_error','potential error')]:
        ax[1].semilogy(ns,[a[key] for a in rows],'o-',label=label)
    ax[1].legend();ax[1].set(title='Spatial semi-discrete consistency',xlabel='N per spatial axis')
    ax[2].semilogy(ns,[a['peak_rss_bytes']/1e9 for a in rows],'o-',label='MMS process peak RSS')
    ax[2].semilogy(ns,[a['full_state_bytes']/1e9 for a in rows],'--',label='one unblocked g array')
    ax[2].legend();ax[2].set(title='Memory of this MMS evaluation',xlabel='N per spatial axis',ylabel='GB')
    fig.savefig(out/'spatial_convergence.png',dpi=160);plt.close(fig)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--n',type=int)
    ap.add_argument('--sizes',type=int,nargs='+',default=[17,33,49,65,81,97,113,129])
    ap.add_argument('--n-mu',type=int,default=192);ap.add_argument('--n-v',type=int,default=40)
    ap.add_argument('--method',choices=['direct','separable'],default='separable')
    ap.add_argument('--mu-block',type=int,default=1);ap.add_argument('--v-block',type=int,default=4)
    ap.add_argument('--out',type=Path,default=Path('build/gyrokinetic_mms/spatial129'))
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if args.n:
        row=run(args.n,n_v=args.n_v,n_mu=args.n_mu,mu_block=args.mu_block,v_block=args.v_block,method=args.method)
        target=args.out/f'spatial_{args.n}_mu{args.n_mu}.json'
        target.write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row),flush=True)
        return
    rows=[]
    for n in args.sizes:
        subprocess.run([sys.executable,__file__,'--n',str(n),'--n-mu',str(args.n_mu),
                        '--n-v',str(args.n_v),'--method',args.method,'--v-block',str(args.v_block),'--mu-block',str(args.mu_block),'--out',str(args.out)],check=True)
        rows.append(json.loads((args.out/f'spatial_{n}_mu{args.n_mu}.json').read_text()))
        (args.out/'spatial_convergence.json').write_text(json.dumps(rows,indent=2)+'\n')
    render(rows,args.out)
    assert rows[-1]['n']>=65
    assert rows[-1]['projection_error']<1e-7
    if rows[-1]['n']==129:
        assert rows[-1]['projection_error']<1e-12
        assert rows[-1]['forced_rhs_defect']<1e-11
    if args.n_mu>=96 and args.n_v>=24 and rows[-1]['n']==65:
        assert rows[-1]['forced_rhs_defect']<1e-10
        assert rows[-1]['phi_error']<1e-10
    assert abs(rows[-1]['projection_error']/rows[-1]['analytic_continuous_projection_tail']-1)<.001

if __name__=='__main__':main()
