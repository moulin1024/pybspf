"""Validate weighted_hole_gram against the saved higher-order reference."""
import argparse
import json
from pathlib import Path
import numpy as np
from weighted_response_assembly import weighted_hole_gram


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--reference',type=Path,default=Path('build/immersed_flow/scaled_quadrature_20260922'))
    ap.add_argument('--reynolds',type=float,nargs='+',default=[200,2000])
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    rows=[];ids=np.concatenate([np.arange(33)+108*k for k in range(4)])
    for re in args.reynolds:
        with np.load(args.reference/f'gram_re{re:g}.npz') as data:ref=data['reference'][np.ix_(ids,ids)]
        scale=1/np.sqrt(np.diag(ref));normalized=ref*scale[:,None]*scale
        for order in (8,12,16):
            G,rec=weighted_hole_gram(re,order)
            error=(G-ref)*scale[:,None]*scale
            rec.update(reynolds=re,maximum_diagonal_relative=float(np.max(abs(np.diag(error)))),
                       scaled_frobenius_relative=float(np.linalg.norm(error)/np.linalg.norm(normalized)),
                       reference_basis_value_visits=148096*132,
                       previous_production_basis_value_visits=91136*132)
            rows.append(rec);print(json.dumps(rec),flush=True)
            np.savez_compressed(args.out/f'hole_gram_re{re:g}_n{order}.npz',gram=G,reference=ref)
            (args.out/'weighted_grams.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
