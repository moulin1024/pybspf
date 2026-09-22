"""Validate weighted_wall_gram against the saved higher-order reference."""
import argparse
import json
from pathlib import Path
import numpy as np
from weighted_response_assembly import weighted_wall_gram


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--reference',type=Path,default=Path('build/immersed_flow/scaled_quadrature_20260922'))
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True);rows=[]
    for re in (200,2000):
        with np.load(args.reference/f'gram_re{re}.npz') as d:full=d['reference']
        for family,offset in [('top',33),('bottom',58),('inlet',83)]:
            ids=np.concatenate([np.arange(25)+offset+108*k for k in range(4)])
            ref=full[np.ix_(ids,ids)];scale=1/np.sqrt(np.diag(ref));normalized=ref*scale[:,None]*scale
            for order in (8,12):
                G,rec=weighted_wall_gram(re,family,order)
                err=(G-ref)*scale[:,None]*scale
                rec.update(maximum_diagonal_relative=float(np.max(abs(np.diag(err)))),
                           scaled_frobenius_relative=float(np.linalg.norm(err)/np.linalg.norm(normalized)),
                           previous_production_basis_value_visits=91136*100)
                rows.append(rec);print(json.dumps(rec),flush=True)
                np.savez_compressed(args.out/f'{family}_gram_re{re}_n{order}.npz',gram=G,reference=ref)
                (args.out/'weighted_wall_grams.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
