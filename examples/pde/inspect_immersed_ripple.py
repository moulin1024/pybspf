"""Inspect saved channel snapshots using raw sections and fourth differences.

Fourth differences are a fixed-sampling roughness diagnostic, not an error norm
or a filter applied to the solution. The figures always include raw data.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True,
                    help='Existing immersed_channel_flow.py output directory')
    args = ap.parse_args()
    out = args.out
    d = np.load(out / 'fields.npz')
    summary = json.loads((out / 'summary.json').read_text())
    x, y, t, f = (d[k] for k in ('x', 'y', 't', 'fields'))
    xx, yy = np.meshgrid(x, y)
    fluid = ((xx-d['center'][0])/d['axes'][0])**2 + ((yy-d['center'][1])/d['axes'][1])**2 >= 1
    h = summary['bounds'][2]
    baseomega = 2*summary['peak_inlet']*yy/h**2
    indices = sorted(set([0, len(t)//2, len(t)-1]))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2), layout='constrained')
    for ax, k in zip(axes[0], indices):
        omega = np.where(fluid, f[k, 2]-baseomega, np.nan)
        im = ax.pcolormesh(x, y, omega, shading='nearest', cmap='RdBu_r',
                           vmin=-.25, vmax=.25)
        ax.add_patch(Ellipse(d['center'], *(2*d['axes']), facecolor='.7', edgecolor='k'))
        ax.set(xlim=(-1, 3), ylim=(-1, 1), xlabel='x', ylabel='y',
               title=f't={t[k]:g}: vorticity minus inlet shear')
        fig.colorbar(im, ax=ax, label='Clipped at +/-0.25')
    for ax in axes[0, len(indices):]:
        ax.set_visible(False)
    section_x = [-.7, -.4]
    for ax, xpos in zip(axes[1, :2], section_x):
        ix = np.argmin(abs(x-xpos))
        for k in indices:
            ax.plot(y, f[k, 2, :, ix], label=f't={t[k]:g}', lw=1.2)
        ax.set(xlabel='y', ylabel='Raw vorticity', xlim=(-.8, .8),
               title=f'Upstream section x={x[ix]:.3f}')
    iy = np.argmin(abs(y-.5))
    for k in indices:
        axes[1, 2].plot(x, f[k, 2, iy]-baseomega[iy], label=f't={t[k]:g}', lw=1.2)
    axes[1, 2].set(xlabel='x', ylabel='Vorticity minus inlet shear',
                   xlim=(1, 5), ylim=(-.25, .25), title=f'Downstream section y={y[iy]:g}')
    for ax in axes[1]:
        ax.grid(alpha=.2); ax.legend()
    fig.suptitle(f"{summary['wall_method']}, Re={summary['reynolds']:g}, "
                 f"{summary['nx']} x {summary['ny']}, dt={summary['dt']:g}, "
                 f"quadrature={summary['quadrature_factor']:g}, sponge={summary['buffer_strength']:g}\n"
                 'Raw samples; no smoothing; shared map color scale')
    fig.savefig(out/'ripple_inspection.png', dpi=170)
    plt.close(fig)

    # The center coordinate of a five-point forward fourth difference is y[2:-2].
    d4 = np.diff(f[:, 2], n=4, axis=1)
    yc = y[2:-2]
    sx = (x > -.85) & (x < -.45)
    sy = (yc > -.8) & (yc < .8)
    region = d4[:, sy][:, :, sx]
    fig, ax = plt.subplots(figsize=(9, 4.5), layout='constrained')
    ix = np.argmin(abs(x+.7))
    for k in indices:
        ax.plot(yc[sy], d4[k, sy, ix], label=f't={t[k]:g}', lw=1.1)
    ax.set(xlim=(-.8, .8), xlabel='y', ylabel='Fourth difference of vorticity',
           title=f'Fixed-grid roughness diagnostic at x={x[ix]:.3f}\nNot a physical error norm')
    ax.legend(); ax.grid(alpha=.2)
    fig.savefig(out/'ripple_fourth_difference.png', dpi=170)
    plt.close(fig)
    report = dict(
        definition='Unscaled fourth forward difference in y, centered at y[2:-2]; RMS over x in (-.85,-.45), y in (-.8,.8). No smoothing; not an exact error norm.',
        output_grid=[len(x),len(y)], dy=float(y[1]-y[0]),
        samples=[dict(t=float(tk), upstream_fourth_difference_rms=float(np.sqrt(np.mean(a*a))))
                 for tk,a in zip(t,region)],
        parameters={k:summary[k] for k in ['wall_method','nx','ny','dt','final_time',
            'reynolds','quadrature_factor','buffer_strength','basis_precision']},
        checks=summary['checks'])
    (out/'ripple_metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['samples'],indent=2))


if __name__ == '__main__':
    main()
