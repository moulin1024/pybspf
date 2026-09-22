"""Show how a paired streamfunction distributes thin and broad velocity changes."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from short_response_space import Jet,matched_primitive


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    ell=np.sqrt(.01*(2/3)*.46/20);outer=.2
    n=np.linspace(0,8*outer,10001);z=Jet(n,1)
    local=.5*z*z*((-1/ell)*z).exp()
    thin=matched_primitive(z,ell)*(1/(1/ell**2-1/outer**2))
    broad=-matched_primitive(z,outer)*(1/(1/ell**2-1/outer**2))
    pair=thin+broad
    np.testing.assert_allclose([local.a[3][0],pair.a[3][0]],1,rtol=1e-14)
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for ax in axes[:2]:
        ax.plot(n/ell,local.a[1]/ell,label='Local exponential',lw=1.4)
        ax.plot(n/ell,pair.a[1]/ell,label='Thin + broad pair',lw=1.4)
        ax.plot(n/ell,broad.a[1]/ell,':',label='Broad component',lw=1.2)
        ax.set(xlabel='Normal distance / thin scale',ylabel='Tangential correction / thin scale')
        ax.axhline(0,color='.5',lw=.6);ax.grid(alpha=.2)
    axes[0].set(xlim=(0,8),title='Near-wall profile; identical wall shear')
    axes[1].set(xlim=(0,80),ylim=(-.05,.015),title='Location of the compensating flow')
    for name,j in [('Local exponential',local),('Thin + broad pair',pair)]:
        axes[2].plot(n/ell,j.a[0]/ell**2,label=name)
    axes[2].set(xlim=(0,80),xlabel='Normal distance / thin scale',ylabel='Cumulative velocity integral / thin scale squared',title='Displacement carried into the outer region')
    axes[2].grid(alpha=.2)
    for ax in axes:ax.legend(fontsize=8)
    fig.suptitle('Flat-wall basis prototype, not a fitted flow solution; both total flux corrections vanish at infinity')
    fig.savefig(args.out/'paired_mode_structure.png',dpi=170);plt.close(fig)
    np.savez_compressed(args.out/'paired_mode_structure.npz',n=n,ell=ell,outer=outer,
                        local_psi=local.a[0],local_velocity=local.a[1],
                        paired_psi=pair.a[0],paired_velocity=pair.a[1])


if __name__=='__main__':main()
