"""Render the validated notebook with denser exact output times, not frame interpolation."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.animation import FFMpegWriter

root = Path(__file__).resolve().parents[1]
notebook = json.loads((root/'examples/pde/schroedinger_1d.ipynb').read_text())
namespace = {}
# Execute the notebook's numerical cells, including all validation assertions.
cells = [c for c in notebook['cells'] if c['cell_type'] == 'code']
exec('\n\n'.join(''.join(c['source']) for c in cells[:-1]), namespace)
jax, jnp, bspf, plt = (namespace[k] for k in ('jax','jnp','bspf','plt'))
weak, x = namespace['weak'], namespace['x']
namespace['times'] = jnp.linspace(0., 2.5, 301)
times = namespace['times']
solution = jax.jit(bspf.integrate_schrodinger)(weak.mass, weak.stiffness, namespace['solution'][0], times)
exact = namespace['reference'](x)
norm = jnp.real(jnp.einsum('ti,ij,tj->t',solution.conj(),weak.mass,solution))
x, times, solution, exact, norm = map(np.asarray, (x,times,solution,exact,norm))
error = np.abs(solution-exact)
assert np.max(error) < 2e-8
print(f'Animation: 301 computed states; max field error={error.max():.4e}', flush=True)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                     'axes.spines.right':False,'axes.titleweight':'bold',
                     'figure.facecolor':'#f6f8fc','axes.facecolor':'white',
                     'axes.labelcolor':'#24344b','text.color':'#24344b','xtick.color':'#52627a','ytick.color':'#52627a'})
fig = plt.figure(figsize=(12.8,8),dpi=100)
grid=fig.add_gridspec(2,2,left=.075,right=.97,bottom=.09,top=.85,hspace=.38,wspace=.25)
a=fig.add_subplot(grid[0,:]);b=fig.add_subplot(grid[1,0]);c=fig.add_subplot(grid[1,1])
fig.suptitle('Schrödinger wave packet · reflecting walls',x=.075,y=.968,ha='left',fontsize=21,weight='bold')
fig.text(.075,.917,'JAX BSPF  |  513 samples · degree 7 · resolved quadrature · exact linear phases',fontsize=11)
clock=fig.text(.97,.966,'',ha='right',fontsize=14,weight='bold')
status=fig.text(.075,.025,'',fontsize=10)
blue, orange, green = '#2463c5','#d37521','#008577'
density,=a.plot(x,np.abs(solution[0])**2,color=blue,lw=2,label='JAX density')
ref,=a.plot(x,np.abs(exact[0])**2,color=orange,lw=1.6,ls='--',label='Continuum reference')
a.set(xlim=(0,20),ylim=(0,1.05*np.max(np.abs(exact)**2)),ylabel=r'$|\psi|^2$',xlabel='x',title='Probability density')
a.legend(loc='upper left',frameon=False,ncol=2)
real,=b.plot(x,solution[0].real,color=blue,lw=1.5,label='Real')
imag,=b.plot(x,solution[0].imag,color=green,lw=1.5,label='Imaginary')
amplitude=np.max(np.abs(solution))*1.05
b.set(xlim=(0,20),ylim=(-amplitude,amplitude),xlabel='x',ylabel=r'$\psi$',title='Complex wave field')
b.legend(loc='upper left',frameon=False,ncol=2)
err,=c.plot(x,error[0],color=orange,lw=1.5)
c.set(xlim=(0,20),ylim=(0,error.max()*1.12),xlabel='x',ylabel=r'$|\psi_{JAX}-\psi_{ref}|$',title='Absolute complex field error')
c.ticklabel_format(axis='y',style='sci',scilimits=(0,0),useMathText=True)
for ax in (a,b,c): ax.grid(alpha=.15)
matplotlib.rcParams['animation.ffmpeg_path']='/opt/homebrew/bin/ffmpeg'
output=root/'examples/pde/results/schroedinger_1d.mp4'
writer=FFMpegWriter(fps=25,codec='libx264',metadata={'title':'JAX BSPF Schrödinger wave packet','comment':'301 computed times over [0,2.5]; continuum cosine reference.'},extra_args=['-crf','19','-pix_fmt','yuv420p','-movflags','+faststart'])
with writer.saving(fig,str(output),dpi=100):
 for i,t in enumerate(times):
  density.set_ydata(np.abs(solution[i])**2); ref.set_ydata(np.abs(exact[i])**2)
  real.set_ydata(solution[i].real);imag.set_ydata(solution[i].imag);err.set_ydata(error[i])
  clock.set_text(f't = {t:.3f}')
  status.set_text(f'Max field error: {error[i].max():.2e}     |     Norm drift: {norm[i]-norm[0]:+.2e}     |     Zero-flux walls at x = 0, 20')
  writer.grab_frame()
  if i==120:fig.savefig(output.with_suffix('.png'),dpi=100)
plt.close(fig)
print(output,flush=True)
