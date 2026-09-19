"""Preserve ripple audit reports and compare unfiltered radial traces at t=2.

Run after source, projection, mean-value and transport audits. CPU work here is
limited to saved-output statistics, Fourier diagnostics and plotting.
"""
import json
from pathlib import Path
import shutil

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root = Path('build/mapped_ripple')
    out = Path('docs/data/mapped_ripple')
    out.mkdir(parents=True, exist_ok=True)
    cases = ('baseline', 'quadrature10', 'lift_c4', 'radial24', 'tangent24',
             'degree4', 'degree4_c4')
    mean_value = json.loads((root/'mean_value.json').read_text())
    harmonic = {row['case']: row for row in mean_value['cases']}
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    shown = {'quadrature10': '16×16, p=3 (q=10)', 'radial24': '24×16, p=3',
             'tangent24': '16×24, p=3', 'degree4': '16×16, p=4',
             'degree4_c4': '16×16, p=4, C4 lift'}
    for case in cases:
        folder = root/case
        report = json.loads((folder/'report.json').read_text())
        shutil.copyfile(folder/'report.json', out/(case+'.json'))
        fields = np.load(folder/'fields.npz')
        if 'omega_t2' in report['metrics']:
            evolved = report['metrics']['omega_t2']
        else:
            evolved = next(item['vorticity'] for item in report['history'] if item['time'] == 2.)
        row = dict(case=case, elements=report['elements'], degree=report['degree'],
                   quadrature=report['quadrature'], dofs=report['dofs'],
                   stokes_d4x=report['metrics']['vorticity']['d4x'],
                   t2_d4x=evolved['d4x'], t2_d4y=evolved['d4y'],
                   stokes_mean_value_rms_R005=harmonic[case]['tests'][1]['mean_value_defect_rms'])
        if 'line_omega_evolved' in fields:
            r, omega = fields['line_r'], fields['line_omega_evolved']
            detrended = omega-np.polynomial.polynomial.polyval(
                r, np.polynomial.polynomial.polyfit(r, omega, 3))
            frequency = np.fft.rfftfreq(len(r), r[1]-r[0])
            window = np.hanning(len(r))
            spectrum = 2*np.abs(np.fft.rfft(detrended*window))/window.sum()
            peak = 1+np.argmax(spectrum[1:])
            row.update(line_detrended_rms=float(np.sqrt(np.mean(detrended**2))),
                       line_peak_cycles_per_r=float(frequency[peak]))
            if case in shown:
                label = shown[case]
                axes[0, 0].plot(r, fields['line_omega_initial'], label=label)
                axes[0, 1].plot(r, omega, label=label)
                axes[1, 0].plot(r, detrended, label=label)
                axes[1, 1].plot(frequency, spectrum, label=label)
        rows.append(row)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    axes[0, 0].set(title='Initial Stokes: raw vorticity', xlabel='Reference radial coordinate r', ylabel='ω')
    axes[0, 1].set(title='t=2: raw vorticity', xlabel='Reference radial coordinate r', ylabel='ω')
    axes[1, 0].set(title='t=2: cubic trend removed for diagnosis only', xlabel='r', ylabel='Residual ω')
    axes[1, 1].set(title='Windowed spectrum of the diagnostic residual', xlabel='Cycles / unit r', ylabel='Amplitude', xlim=(0, 24))
    axes[1, 1].axvline(8, color='k', ls=':', lw=.7)
    axes[1, 1].axvline(12, color='k', ls=':', lw=.7)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(out/'radial_controls.png', dpi=160)
    plt.close(fig)

    # Independent closed-form quadratic periodic spline symbol.
    dispersion = np.load(root/'transport/dispersion.npz')
    theta = dispersion['theta']
    exact = ((5/6)*np.sin(theta)+(1/12)*np.sin(2*theta))/(11/20+(13/30)*np.cos(theta)+(1/60)*np.cos(2*theta))
    symbol_error = float(np.max(np.abs(exact-dispersion['frequency'])))
    assert symbol_error < 1e-12
    max_parity = max(row['reconstruction_max_error'] for row in mean_value['cases'])
    assert max_parity < 1e-9
    # Mean-value integration sanity checks on harmonic and non-harmonic fields.
    centers = np.array(((.1, .2), (-.6, -.1)))
    angle = np.arange(256)*2*np.pi/256
    radius = .05
    rings = centers[:, None]+radius*np.stack((np.cos(angle), np.sin(angle)), axis=-1)
    def h(z): return z[..., 0]**2-z[..., 1]**2+z[..., 1]
    harmonic_error = float(np.max(np.abs(h(centers)-h(rings).mean(axis=1))))
    nonharmonic_defect = np.sum(centers**2, axis=-1)-np.sum(rings**2, axis=-1).mean(axis=1)
    assert harmonic_error < 1e-14
    assert np.max(np.abs(nonharmonic_defect+radius**2)) < 1e-14
    checks = dict(periodic_symbol_max_error=symbol_error,
                  saved_stokes_reconstruction_max_error=max_parity,
                  analytic_harmonic_mean_value_error=harmonic_error)
    (out/'comparison.json').write_text(json.dumps(dict(cases=rows, checks=checks), indent=2)+'\n')
    shutil.copyfile(root/'mean_value.json', out/'mean_value.json')
    for case in ('projection_strong', 'projection_nitsche', 'transport'):
        shutil.copyfile(root/case/'report.json', out/(case+'.json'))
    for source, target in (('baseline/source.png', 'source.png'),
                           ('projection_strong/leakage.png', 'projection_leakage.png'),
                           ('transport/dispersion.png', 'dispersion.png')):
        shutil.copyfile(root/source, out/target)
    print(json.dumps(dict(cases=rows, checks=checks), indent=2))


if __name__ == '__main__':
    main()
