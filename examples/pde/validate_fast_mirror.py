"""Validate matrix-free mirror evolution against dense axes and characteristics."""
import json
import runpy
from pathlib import Path
import numpy as np


def main():
    module=runpy.run_path(str(Path(__file__).with_name('drift_kinetic_mirror.py')))
    run=module['run']
    fine,arrays=run(65)
    dense,reference=run(65,backend='dense')
    modulated,mod_arrays=run(65,profile='modulated')
    report=dict(matrix_free=fine,dense=dense,modulated=modulated,
        dense_distribution_difference=float(np.max(np.abs(arrays['history']-reference['history']))),
        dense_moment_difference=float(np.max(np.abs(arrays['moments']-reference['moments']))),
        dense_transfer_difference=float(np.max(np.abs(arrays['transfers']-reference['transfers']))))
    out=Path('build/fast_mirror');out.mkdir(parents=True,exist_ok=True)
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(out/'solution.npz',**arrays)
    np.savez_compressed(out/'modulated_solution.npz',**mod_arrays)
    module['render'](arrays,out)
    for result in (fine,modulated):
        assert result['max_distribution_error']<1e-7
        assert result['relative_particle_balance']<1e-8
        assert result['relative_energy_balance']<1e-8
        assert result['min_distribution']>=0 and result['min_quadrature_distribution']>=0
    assert report['dense_distribution_difference']<1e-8
    assert report['dense_moment_difference']<1e-9
    assert report['dense_transfer_difference']<1e-9
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
