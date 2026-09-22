"""Check complete response-mode energy Grams under scaled-rule order changes."""
import argparse
import json
from pathlib import Path
import numpy as np
from bspf_models.elliptic.immersed_poisson import EllipticHole
from scaled_response_quadrature import scaled_channel_quadrature
from short_response_space import ResponseModes


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--reynolds',type=float,nargs='+',default=[200,2000])
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    rows=[];hole=EllipticHole();bounds=(-1,5,1)
    for re in args.reynolds:
        ell=np.sqrt(.01*(2/3)*.46/re);thin=ell*np.array([.5,1,2,4])
        modes=ResponseModes(bounds,hole,np.r_[thin,.1,.2])
        grams=[];counts=[]
        for layer,bulk in [(48,32),(72,64)]:
            p,w=scaled_channel_quadrature(bounds,hole,73,33,layer_lengths=thin,
                                          layer_order=layer,bulk_order=bulk)
            G=np.zeros((648,648))
            for start in range(0,len(w),2048):
                stop=min(start+2048,len(w));ops=modes.operators(p[start:stop])[1:]
                for o,c in zip(ops,(1,1,2,1,1)):
                    G+=c*(o.T@(w[start:stop,None]*o))
            grams.append(G);counts.append(len(w))
            print('GRAM',re,layer,bulk,len(w),flush=True)
        scale=1/np.sqrt(np.diag(grams[-1]));ref=grams[-1]*scale[:,None]*scale
        error=(grams[0]-grams[1])*scale[:,None]*scale
        row=dict(reynolds=re,point_counts=counts,
                 maximum_diagonal_relative=float(np.max(abs(np.diag(error)))),
                 scaled_frobenius_relative=float(np.linalg.norm(error)/np.linalg.norm(ref)))
        rows.append(row);print(json.dumps(row),flush=True)
        np.savez_compressed(args.out/f'gram_re{re:g}.npz',production=grams[0],reference=grams[1])
        (args.out/'grams.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
