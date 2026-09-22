"""Compare enriched NS runs on the same grid after quadrature refinement."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('coarse',type=Path)
    ap.add_argument('fine',type=Path)
    args=ap.parse_args()
    a,b=(np.load(p/'fields.npz') for p in (args.coarse,args.fine))
    for key in ['x','y','t']:
        np.testing.assert_allclose(a[key],b[key],atol=1e-14,rtol=0)
    final_a,final_b=a['fields'][-1],b['fields'][-1]
    def rel(u,v):
        mask=np.isfinite(u)&np.isfinite(v)
        return float(np.linalg.norm((u-v)[mask])/np.linalg.norm(v[mask]))
    x,y=a['x'],a['y'];sx=(x>-.85)&(x<-.45);sy=(y[2:-2]>-.8)&(y[2:-2]<.8)
    da,db=[np.diff(f[2],n=4,axis=0)[sy][:,sx] for f in [final_a,final_b]]
    ra,rb=[float(np.sqrt(np.mean(d*d))) for d in [da,db]]
    report=dict(coarse=str(args.coarse),fine=str(args.fine),
        definition='Unweighted identical-grid field differences; not exact continuum errors.',
        final_velocity_relative_grid_l2=rel(final_a[:2],final_b[:2]),
        final_vorticity_relative_grid_l2=rel(final_a[2],final_b[2]),
        upstream_roughness_coarse=ra,upstream_roughness_fine=rb,
        roughness_relative_change=abs(ra-rb)/rb,
        roughness_field_relative_change=float(np.linalg.norm(da-db)/np.linalg.norm(db)))
    (args.fine/'refinement.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
