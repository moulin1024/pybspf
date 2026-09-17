"""Animate computed states from the sine–Gordon collision notebook."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.animation import FFMpegWriter

root = Path(__file__).resolve().parents[1]
notebook = json.loads((root/'examples/pde/sine_gordon_1d.ipynb').read_text())
cells = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
ns = {}
exec(cells[0], ns)
exec(cells[1].split('\nx, plan, weak, u, v = solve()')[0], ns)
jnp, plt = ns['jnp'], ns['plt']
ns['times'] = jnp.linspace(0., 10., 401)
x, plan, weak, u, v = ns['solve'](substeps=25)  # dt=0.001, computed frames
ns.update(x=x, plan=plan, weak=weak, u=u, v=v)
exec(cells[3], ns)  # Notebook's independent energy reference and balance checks.
reference = ns['exact'](x[None, :], ns['times'][:, None])
error = np.asarray(jnp.max(jnp.abs(u-reference), axis=1))
assert error.max() < 2e-8
ux = ns['ux']
density = np.asarray(.5*v*v+.5*ux*ux+1-jnp.cos(u))
rv = ns['exact_velocity'](x[None, :], ns['times'][:, None])
rx = ns['exact_gradient'](x[None, :], ns['times'][:, None])
reference_density = np.asarray(.5*rv*rv+.5*rx*rx+1-jnp.cos(reference))
x, times, u, reference, energy, exact_energy = map(np.asarray,
    (x, ns['times'], u, reference, ns['energy'], ns['exact_energy']))
assert np.max(np.abs(u[:, [0, -1]]-reference[:, [0, -1]])) < 1e-13
print(f'{len(times)} computed states; max field error={error.max():.4e}', flush=True)

plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11,
    'axes.spines.top':False, 'axes.spines.right':False, 'axes.titleweight':'bold',
    'figure.facecolor':'#f6f8fc', 'axes.facecolor':'white', 'text.color':'#24344b'})
fig = plt.figure(figsize=(12.8, 8), dpi=100)
grid = fig.add_gridspec(2, 2, left=.08, right=.97, bottom=.11, top=.83,
                      hspace=.4, wspace=.27)
a = fig.add_subplot(grid[0, :]); d = fig.add_subplot(grid[1, 0]); e = fig.add_subplot(grid[1, 1])
fig.suptitle('Sine–Gordon · kink–antikink collision', x=.08, y=.97,
             ha='left', fontsize=22, weight='bold')
fig.text(.08, .91, r'$u_{tt}=u_{xx}-\sin u$  |  Nonperiodic interval [−4, 4] · moving Dirichlet boundaries', fontsize=11)
clock = fig.text(.97, .967, '', ha='right', fontsize=14, weight='bold')
status = fig.text(.08, .035, '', fontsize=10)
blue, orange = '#2463c5', '#d37521'
field, = a.plot(x, u[0], color=blue, lw=2.5, label='JAX BSPF')
ref, = a.plot(x, reference[0], '--', color='#253449', lw=1.3, label='Exact two-soliton solution')
ends, = a.plot(x[[0, -1]], u[0, [0, -1]], 'o', color=orange, ms=6, clip_on=False, label='Prescribed endpoints')
a.set(xlim=(-4, 4), ylim=(-6.8, 6.8), xlabel='x', ylabel='u', title='Approach → collision at t = 5 → separation')
a.legend(loc='upper left', frameon=False, fontsize=9, ncol=3)
dens, = d.plot(x, density[0], color=orange, lw=2.3, label='Numerical')
dens_ref, = d.plot(x, reference_density[0], '--', color='#253449', lw=1.2, label='Exact')
d.set(xlim=(-4, 4), ylim=(0, 13.3), xlabel='x', ylabel='Energy density', title='Energy remains nonzero at collision')
d.legend(frameon=False, fontsize=9)
e.plot(times, exact_energy, '--', color='#8996a8', lw=1.5, label='Exact finite-interval energy')
trace, = e.plot([], [], color=blue, lw=2, label='Numerical energy')
point, = e.plot([], [], 'o', color=blue, ms=5)
e.axvline(5, ls=':', color='#8996a8', lw=1)
e.set(xlim=(0, 10), ylim=(14.8, 20.5), xlabel='t', ylabel='Energy', title='Energy enters and leaves through the boundaries')
e.legend(loc='lower center', frameon=False, fontsize=9)
for ax in (a, d, e): ax.grid(alpha=.15)
ffmpeg = Path('/opt/homebrew/bin/ffmpeg')
if ffmpeg.exists(): matplotlib.rcParams['animation.ffmpeg_path'] = str(ffmpeg)
else:
    import imageio_ffmpeg
    matplotlib.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
output = root/'examples/pde/results/sine_gordon_collision.mp4'
output.parent.mkdir(parents=True, exist_ok=True)
writer = FFMpegWriter(fps=25, codec='libx264', metadata={
    'title':'JAX BSPF sine–Gordon kink–antikink collision',
    'comment':'401 computed states, t=0..10, nonperiodic moving Dirichlet boundaries; dt=0.001.'},
    extra_args=['-crf','19','-pix_fmt','yuv420p','-movflags','+faststart'])
with writer.saving(fig, str(output), dpi=100):
    for i, t in enumerate(times):
        field.set_ydata(u[i]); ref.set_ydata(reference[i]); ends.set_ydata(u[i, [0, -1]])
        dens.set_ydata(density[i]); dens_ref.set_ydata(reference_density[i])
        trace.set_data(times[:i+1], energy[:i+1]); point.set_data([t], [energy[i]])
        clock.set_text(f't = {t:.3f}')
        status.set_text(f'Max field error: {error[i]:.2e}   |   Energy error: {abs(energy[i]-exact_energy[i]):.2e}   |   129 spatial samples')
        writer.grab_frame()
        if i == 200: fig.savefig(output.with_name('sine_gordon_collision_frame.png'), dpi=100)
plt.close(fig)
print(output, flush=True)
