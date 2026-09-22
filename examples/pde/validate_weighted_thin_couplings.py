"""Validate complete_thin_gram against the saved higher-order reference."""
import argparse
import json
from pathlib import Path
import numpy as np
from weighted_response_assembly import complete_thin_gram


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--reference',type=Path,default=Path('build/immersed_flow/scaled_quadrature_20260922'))
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True);rows=[]
    for re in (200,2000):
        with np.load(args.reference/f'gram_re{re}.npz') as d:ref=d['reference'][:432,:432]
        s=1/np.sqrt(np.diag(ref));norm=np.linalg.norm(ref*s[:,None]*s)
        for order in (8,12):
            G,rec=complete_thin_gram(re,order)
            err=(G-ref)*s[:,None]*s
            rec.update(maximum_diagonal_relative=float(np.max(abs(np.diag(err)))),
                       scaled_frobenius_relative=float(np.linalg.norm(err)/norm),
                       symmetry_error=float(np.max(abs(G-G.T))))
            print(json.dumps(rec),flush=True);rows.append(rec)
            np.savez_compressed(args.out/f'complete_thin_re{re}_n{order}.npz',gram=G,reference=ref)
            (args.out/'complete_thin_grams.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
