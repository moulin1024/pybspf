"""Compare identical-initial-state channel runs on the same output grid."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np


def roughness(data):
    x, y, fields = (data[k] for k in ('x', 'y', 'fields'))
    d4 = np.diff(fields[:, 2], n=4, axis=1)
    region = d4[:, (y[2:-2] > -.8) & (y[2:-2] < .8)]
    region = region[:, :, (x > -.85) & (x < -.45)]
    return np.sqrt(np.mean(region**2, axis=(1, 2)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--enriched', type=Path, required=True)
    args = parser.parse_args()
    data = [np.load(p/'fields.npz') for p in (args.baseline, args.enriched)]
    for key in ('x', 'y', 'center', 'axes'):
        np.testing.assert_array_equal(data[0][key], data[1][key])
    np.testing.assert_allclose(data[0]['fields'][0], data[1]['fields'][0], atol=1e-9, rtol=1e-9)
    common = [t for t in data[0]['t'] if np.any(np.isclose(data[1]['t'], t, atol=1e-12, rtol=0))]
    t = max(common)
    indices = [int(np.flatnonzero(np.isclose(d['t'], t, atol=1e-12, rtol=0))[0]) for d in data]
    x, y = data[0]['x'], data[0]['y']
    xx, yy = np.meshgrid(x, y)
    center, axes = data[0]['center'], data[0]['axes']
    fluid = ((xx-center[0])/axes[0])**2 + ((yy-center[1])/axes[1])**2 >= 1
    fig, panels = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
    curves = [roughness(d) for d in data]
    labels = ['Geometric space', 'Geometric + response enrichment']
    for d, k, label, ax in zip(data, indices, labels, panels[0]):
        im = ax.pcolormesh(x, y, np.where(fluid, d['fields'][k, 2]-2*yy, np.nan),
                          cmap='RdBu_r', vmin=-.25, vmax=.25, shading='nearest')
        ax.add_patch(Ellipse(center, *(2*axes), facecolor='.7', edgecolor='k'))
        ax.set(xlim=(-1, 3), ylim=(-1, 1), title=f'{label}, t={t:g}', xlabel='x', ylabel='y')
        fig.colorbar(im, ax=ax, label='Vorticity minus inlet shear (clipped)')
    ix = np.argmin(abs(x+.7))
    for d, k, label, curve in zip(data, indices, labels, curves):
        panels[1, 0].plot(y, d['fields'][k, 2, :, ix], label=label)
        panels[1, 1].semilogy(d['t'], curve, '.-', label=label)
    panels[1, 0].set(xlim=(-.8, .8), xlabel='y', ylabel='Raw vorticity', title=f'Section x={x[ix]:g}')
    panels[1, 1].set(xlabel='t', ylabel='Fourth-difference RMS', title='Fixed-grid upstream roughness; not an error norm')
    for ax in panels[1]:
        ax.grid(alpha=.2)
        ax.legend()
    fig.savefig(args.enriched/'enrichment_comparison.png', dpi=170)
    report = dict(comparison_time=float(t), baseline_rms=float(curves[0][indices[0]]),
                  enriched_rms=float(curves[1][indices[1]]))
    report['reduction_factor'] = report['baseline_rms']/report['enriched_rms']
    (args.enriched/'enrichment_comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
