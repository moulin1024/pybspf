"""Animate the computed nonlinear open-domain Vlasov–Poisson notebook."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.animation import FFMpegWriter

root = Path(__file__).resolve().parents[1]
notebook = json.loads((root/'examples/pde/landau_open_1d.ipynb').read_text())
cells = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
ns = {}
# 0.04 between frames / 2 = 0.02: preserve the notebook's internal RK4 step.
setup = cells[0].replace('jnp.linspace(0., 12., 121)', 'jnp.linspace(0., 12., 301)')
setup = setup.replace('half_length=40., substeps=5', 'half_length=40., substeps=2')
exec(setup, ns)
exec(cells[1], ns)
exec(cells[2], ns)
plt = ns['plt']
z, v, times, delta, electric, field_energy, particle_energy, free_energy, boundary = (
    np.asarray(ns[k]) for k in ('z', 'v', 'times', 'delta', 'electric', 'field_energy',
                               'particle_free_energy', 'free_energy', 'boundary_transfer'))
assert len(times) == 301 and np.isfinite(delta).all()
print(f'{len(times)} computed frames; relative free-energy balance={ns["free_balance"]:.3e}', flush=True)

plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11,
    'axes.spines.top':False, 'axes.spines.right':False, 'axes.titleweight':'bold',
    'figure.facecolor':'#f6f8fc', 'axes.facecolor':'white', 'text.color':'#24344b'})
fig = plt.figure(figsize=(12.8, 8), dpi=100)
grid = fig.add_gridspec(2, 2, left=.085, right=.92, bottom=.12, top=.82,
                      height_ratios=(1.2, 1), hspace=.45, wspace=.29)
a = fig.add_subplot(grid[0, :]); e = fig.add_subplot(grid[1, 0]); w = fig.add_subplot(grid[1, 1])
fig.suptitle('Landau damping · open Vlasov–Poisson plasma', x=.085, y=.97,
             ha='left', fontsize=21, weight='bold')
fig.text(.085, .915, 'Self-consistent electric field · Maxwellian inflow · free outflow · grounded endpoints', fontsize=11)
clock = fig.text(.945, .966, '', ha='right', fontsize=13, weight='bold')
status = fig.text(.085, .037, '', fontsize=9.5)
limit = np.max(np.abs(delta))
heat = a.pcolormesh(z, v, 1e4*delta[0].T, shading='auto', cmap='RdBu_r', vmin=-1e4*limit, vmax=1e4*limit)
a.set(xlim=(-40, 40), ylim=(-6, 6), xlabel='z', ylabel=r'$v_\parallel$',
      title='Phase mixing: distribution perturbation f − Maxwellian')
pos = a.get_position(); cax = fig.add_axes([.934, pos.y0, .012, pos.height])
fig.colorbar(heat, cax=cax)
cax.set_title(r'$10^4\delta f$', fontsize=9, pad=8)
blue, orange, green = '#2463c5', '#d37521', '#008577'
field, = e.plot(z, electric[0], color=blue, lw=2.3)
elimit = 1.12*np.max(np.abs(electric))
e.set(xlim=(-40, 40), ylim=(-elimit, elimit), xlabel='z', ylabel='E',
      title='Electric field from BSPF Poisson integration')
scale = free_energy[0]
curves = (field_energy/scale, particle_energy/scale, free_energy/scale, 1+boundary/scale)
lines = []
for label, color, style in [('Electric field',blue,'-'), ('Particle free energy',orange,'-'),
                            ('Total',green,'-'), ('Initial + boundary transfer','#59647a','--')]:
    line, = w.plot([], [], color=color, ls=style, lw=1.8, label=label)
    lines.append(line)
w.set(xlim=(0, 12), ylim=(0, 1.08), xlabel='t', ylabel='Fraction of initial free energy',
      title='Field energy transfers into particle structure')
w.legend(loc='center right', frameon=False, fontsize=8.5)
for ax in (e, w): ax.grid(alpha=.15)
ffmpeg = Path('/opt/homebrew/bin/ffmpeg')
if ffmpeg.exists(): matplotlib.rcParams['animation.ffmpeg_path'] = str(ffmpeg)
else:
    import imageio_ffmpeg
    matplotlib.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
output = root/'examples/pde/results/landau_open_phase_space.mp4'
output.parent.mkdir(parents=True, exist_ok=True)
writer = FFMpegWriter(fps=25, codec='libx264', metadata={
    'title':'JAX BSPF open-domain Landau damping',
    'comment':'301 computed nonlinear Vlasov–Poisson states, t=0..12, 129x129 samples, dt=0.02; Poisson by BSPF integration.'},
    extra_args=['-crf','19','-pix_fmt','yuv420p','-movflags','+faststart'])
with writer.saving(fig, str(output), dpi=100):
    for i, t in enumerate(times):
        heat.set_array(1e4*delta[i].T.ravel()); field.set_ydata(electric[i])
        for line, curve in zip(lines, curves): line.set_data(times[:i+1], curve[:i+1])
        clock.set_text(f't = {t:.2f}')
        status.set_text(f'Electric energy / initial: {field_energy[i]/field_energy[0]:.3f}   |   Free energy lost at boundaries: {-100*boundary[i]/scale:.2f}%   |   129 × 129 samples')
        writer.grab_frame()
        if i == 250: fig.savefig(output.with_suffix('.png'), dpi=100)
plt.close(fig)
print(output, flush=True)
