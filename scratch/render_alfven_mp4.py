"""Render computed states and BSPF energy diagnostics from the Alfvén notebook."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.animation import FFMpegWriter

root = Path(__file__).resolve().parents[1]
notebook = json.loads((root/'examples/pde/alfven_1d.ipynb').read_text())
cells = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
ns = {}
for index in (0, 1, 4):
    exec(cells[index], ns)
plt = ns['plt']
z, times, xi, magnetic, energy, work = (np.asarray(ns[k]) for k in
    ('z', 'times', 'xi', 'magnetic', 'energy', 'work'))
amplitude, duration = ns['amplitude'], ns['drive_duration']
assert len(times) == 401 and np.isfinite(xi).all() and np.isfinite(magnetic).all()
print(f'{len(times)} computed frames, dt=0.0005; energy balance={ns["balance_error"]:.3e}', flush=True)

plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11,
    'axes.spines.top':False, 'axes.spines.right':False, 'axes.titleweight':'bold',
    'figure.facecolor':'#f6f8fc', 'axes.facecolor':'white', 'text.color':'#24344b'})
fig = plt.figure(figsize=(12.8, 8), dpi=100)
grid = fig.add_gridspec(2, 2, left=.085, right=.97, bottom=.11, top=.81,
                      hspace=.45, wspace=.28)
a = fig.add_subplot(grid[0, :]); m = fig.add_subplot(grid[1, 0]); e = fig.add_subplot(grid[1, 1])
fig.suptitle('Boundary-driven Alfvén waves', x=.085, y=.97,
             ha='left', fontsize=22, weight='bold')
fig.text(.085, .914, 'Nonperiodic cavity · density 1 → 4 · driven left footpoint · fixed right footpoint', fontsize=11)
clock = fig.text(.97, .968, '', ha='right', fontsize=14, weight='bold')
status = fig.text(.085, .035, '', fontsize=10)
blue, orange, green = '#2463c5', '#d37521', '#008577'
field, = a.plot(z, xi[0]/amplitude, color=blue, lw=2.5, label='Computed displacement')
left, = a.plot([0], [0], 'o', color=orange, ms=7, clip_on=False, label='Driven left endpoint')
a.plot([1], [0], 's', color='#253449', ms=7, clip_on=False, label='Fixed right endpoint')
a.axvspan(.42, .58, color=orange, alpha=.10, label='Density transition')
a.set(xlim=(0, 1), ylim=(-1.2, 1.2), xlabel='z / L', ylabel=r'$\xi/A$',
      title='Wave injection, internal scattering and wall reflections')
a.legend(loc='upper left', frameon=False, fontsize=9, ncol=4)
phase = a.text(.985, .06, '', transform=a.transAxes, ha='right', color=orange, fontsize=10)
mag, = m.plot(z, magnetic[0], color=green, lw=2.2)
m.axvspan(.42, .58, color=orange, alpha=.10)
magnetic_limit = 1.1*np.max(np.abs(magnetic))
m.set(xlim=(0, 1), ylim=(-magnetic_limit, magnetic_limit), xlabel='z / L', ylabel=r'$b_\perp/B_0$',
      title='Transverse magnetic perturbation')
e.plot(times, work*1e5, '--', color='#8996a8', lw=1.5, label='Integrated boundary work')
trace, = e.plot([], [], color=blue, lw=2.2, label='BSPF energy integral')
point, = e.plot([], [], 'o', color=blue, ms=5)
e.axvline(duration, color=orange, ls=':', lw=1.2, label='Driver stops')
e.set(xlim=(0, 4), ylim=(-.06, 1.85), xlabel='t', ylabel=r'$E\; (\times 10^{-5})$',
      title='Energy stays in the line-tied cavity')
e.legend(loc='lower right', frameon=False, fontsize=9)
for ax in (a, m, e): ax.grid(alpha=.15)
ffmpeg = Path('/opt/homebrew/bin/ffmpeg')
if ffmpeg.exists(): matplotlib.rcParams['animation.ffmpeg_path'] = str(ffmpeg)
else:
    import imageio_ffmpeg
    matplotlib.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
output = root/'examples/pde/results/alfven_boundary_driven.mp4'
output.parent.mkdir(parents=True, exist_ok=True)
writer = FFMpegWriter(fps=25, codec='libx264', metadata={
    'title':'JAX BSPF boundary-driven Alfvén waves',
    'comment':'401 computed states, t=0..4; 129 samples; dt=0.0005; variable density and nonperiodic line-tied boundaries.'},
    extra_args=['-crf','19','-pix_fmt','yuv420p','-movflags','+faststart'])
with writer.saving(fig, str(output), dpi=100):
    for i, t in enumerate(times):
        field.set_ydata(xi[i]/amplitude); left.set_ydata([xi[i, 0]/amplitude])
        mag.set_ydata(magnetic[i]); trace.set_data(times[:i+1], energy[:i+1]*1e5)
        point.set_data([t], [energy[i]*1e5]); clock.set_text(f't = {t:.3f}')
        phase.set_text('Left footpoint driving' if t < duration else 'Driver stopped · both ends fixed')
        status.set_text(f'JAX BSPF · 129 samples   |   Max relative energy–work residual: {ns["balance_error"]:.2e}   |   Post-drive energy drift: {ns["energy_drift"]:.2e}')
        writer.grab_frame()
        if i == 180: fig.savefig(output.with_suffix('.png'), dpi=100)
plt.close(fig)
print(output, flush=True)
