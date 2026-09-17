"""Animate computed open-boundary 1z1v dynamics from the JAX notebook."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.animation import FFMpegWriter

root = Path(__file__).resolve().parents[1]
notebook = json.loads((root/'examples/pde/parallel_kinetic_1d.ipynb').read_text())
cells = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
ns = {}
exec(cells[0], ns)
ns['times'] = ns['jnp'].linspace(0., 1.2, 301)
# 0.004 between frames / 8 = 0.0005: same internal step as the notebook.
exec(cells[1].replace('z, v, zp, vp, f = solve()',
                     'z, v, zp, vp, f = solve(substeps=8)'), ns)
exec(cells[3], ns)  # Actual BSPF particle/energy integrals and assertions.
plt = ns['plt']
z, v, times, f, density, number, transferred, error = (np.asarray(ns[k]) for k in
    ('z', 'v', 'times', 'f', 'density', 'number', 'number_from_flux', 'error'))
assert len(times) == 301 and np.isfinite(f).all()
print(f'{len(times)} computed states; maximum distribution error={error.max():.4e}', flush=True)

plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11,
    'axes.spines.top':False, 'axes.spines.right':False, 'axes.titleweight':'bold',
    'figure.facecolor':'#f6f8fc', 'axes.facecolor':'white', 'text.color':'#24344b'})
fig = plt.figure(figsize=(12.8, 8), dpi=100)
grid = fig.add_gridspec(2, 2, left=.08, right=.945, bottom=.12, top=.83,
                      height_ratios=(1.25, 1), hspace=.43, wspace=.28)
a = fig.add_subplot(grid[0, :]); d = fig.add_subplot(grid[1, 0]); n = fig.add_subplot(grid[1, 1])
fig.suptitle('Parallel kinetic dynamics · open phase space', x=.08, y=.97,
             ha='left', fontsize=21, weight='bold')
fig.text(.08, .916, r'$f_t+v_\parallel f_z+0.3 f_{v_\parallel}=0$  |  Prescribed electric field · unequal incoming beams · free outflow', fontsize=11)
clock = fig.text(.945, .964, '', ha='right', fontsize=13, weight='bold')
status = fig.text(.08, .038, '', fontsize=9.5)
heat = a.pcolormesh(z, v, f[0].T, shading='auto', cmap='magma', vmin=0, vmax=1)
a.set(xlim=(0, 1), ylim=(-2.5, 2.5), xlabel='z', ylabel=r'$v_\parallel$',
      title='Phase-space distribution: injection, streaming, acceleration and escape')
# A dedicated colorbar keeps the physical panel aligned with the lower plots.
position = a.get_position()
cax = fig.add_axes([.957, position.y0, .013, position.height])
fig.colorbar(heat, cax=cax, label='f')
for start, end, height, label in [(.025,.15,1.95,'Inflow'),(.85,.975,1.95,'Outflow'),
                                 (.15,.025,-1.95,'Outflow'),(.975,.85,-1.95,'Inflow')]:
    a.annotate('', xy=(end,height), xytext=(start,height),
               arrowprops=dict(arrowstyle='->',color='white',lw=1.4))
    a.text((start+end)/2, height+.2, label, color='white', ha='center', fontsize=9)
blue, orange = '#2463c5', '#d37521'
profile, = d.plot(z, density[0], color=blue, lw=2.4)
d.set(xlim=(0, 1), ylim=(0, 1.12*density.max()), xlabel='z', ylabel=r'$\int f\,dv_\parallel$',
      title='Number density from BSPF integration')
n.plot(times, transferred, '--', color='#8996a8', lw=1.6, label='Initial + net boundary transfer')
trace, = n.plot([], [], color=orange, lw=2.3, label='BSPF particle integral')
point, = n.plot([], [], 'o', color=orange, ms=5)
n.set(xlim=(0, 1.2), ylim=(0, 1.13*number.max()), xlabel='t', ylabel='Particles in the finite domain',
      title='Particles enter and leave through the boundaries')
n.legend(loc='lower center', frameon=False, fontsize=9)
for ax in (d, n): ax.grid(alpha=.15)
ffmpeg = Path('/opt/homebrew/bin/ffmpeg')
if ffmpeg.exists(): matplotlib.rcParams['animation.ffmpeg_path'] = str(ffmpeg)
else:
    import imageio_ffmpeg
    matplotlib.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
output = root/'examples/pde/results/parallel_kinetic_phase_space.mp4'
output.parent.mkdir(parents=True, exist_ok=True)
writer = FFMpegWriter(fps=25, codec='libx264', metadata={
    'title':'JAX BSPF open parallel kinetic phase space',
    'comment':'301 computed states, t=0..1.2, 65x129 grid, dt=0.0005; incoming-only reservoirs, prescribed constant acceleration.'},
    extra_args=['-crf','19','-pix_fmt','yuv420p','-movflags','+faststart'])
with writer.saving(fig, str(output), dpi=100):
    for i, t in enumerate(times):
        heat.set_array(f[i].T.ravel()); profile.set_ydata(density[i])
        trace.set_data(times[:i+1], number[:i+1]); point.set_data([t], [number[i]])
        clock.set_text(f't = {t:.3f}')
        status.set_text(f'Max distribution error: {error[i]:.2e}   |   Minimum f: {f[i].min():.2e} (no clipping)   |   65 × 129 samples')
        writer.grab_frame()
        if i == 175: fig.savefig(output.with_suffix('.png'), dpi=100)
plt.close(fig)
print(output, flush=True)
