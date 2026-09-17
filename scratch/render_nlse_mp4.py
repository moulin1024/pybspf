"""Animate the notebook's NLSE model at computed output times (no frame interpolation)."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.animation import FFMpegWriter

root = Path(__file__).resolve().parents[1]
notebook = json.loads((root/'examples/pde/nlse_1d.ipynb').read_text())
cells = [c for c in notebook['cells'] if c['cell_type'] == 'code']
namespace = {}
# Reuse notebook imports, parameters, reference, and spatial-plan factory.
exec(''.join(cells[0]['source']), namespace)
exec(''.join(cells[1]['source']).split('\nx, weak = spatial_plan')[0], namespace)
jax, jnp, bspf, plt = (namespace[k] for k in ('jax','jnp','bspf','plt'))
x, weak = namespace['spatial_plan'](513)
reference = namespace['reference']
times = jnp.linspace(0., 4., 321)
initial = reference(x, 0.)
solution = jax.jit(lambda: bspf.integrate_nlse(
    weak, initial, times, coupling=2., substeps=5))()  # Same dt=0.0025 as notebook.
linear = jax.jit(bspf.integrate_schrodinger)(weak.mass, weak.stiffness, initial, times)
exact = reference(x[None, :], times[:, None])
norm = jnp.real(jnp.einsum('ti,ij,tj->t',solution.conj(),weak.mass,solution))
x, times, solution, linear, exact, norm = map(np.asarray,(x,times,solution,linear,exact,norm))
error = np.max(np.abs(solution-exact),axis=1)
center = namespace['x0']+2*namespace['wave_number']*times
width = np.sqrt(np.trapezoid((x[None,:]-center[:,None])**2*np.abs(solution)**2,x,axis=1)
                /np.trapezoid(np.abs(solution)**2,x,axis=1))
assert np.isfinite(solution).all()
assert error.max() < 5e-9
assert np.max(np.abs(norm/norm[0]-1)) < 1e-10
assert np.max(np.abs(width-np.pi/np.sqrt(12))) < 1e-8
print(f'321 computed states; max complex field error={error.max():.4e}; norm drift={np.max(np.abs(norm/norm[0]-1)):.4e}',flush=True)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
 'axes.spines.right':False,'axes.titleweight':'bold','figure.facecolor':'#f6f8fc',
 'axes.facecolor':'white','axes.labelcolor':'#24344b','text.color':'#24344b',
 'xtick.color':'#52627a','ytick.color':'#52627a'})
fig=plt.figure(figsize=(12.8,8),dpi=100)
grid=fig.add_gridspec(2,2,left=.075,right=.97,bottom=.10,top=.83,hspace=.42,wspace=.25)
a=fig.add_subplot(grid[0,:]);b=fig.add_subplot(grid[1,0]);c=fig.add_subplot(grid[1,1])
fig.suptitle('Focusing NLSE · a traveling bright soliton',x=.075,y=.965,ha='left',fontsize=21,weight='bold')
fig.text(.075,.913,r'$i\psi_t=-\psi_{xx}-2|\psi|^2\psi$  |  JAX BSPF · 513 samples · $\Delta t=0.0025$',fontsize=12)
clock=fig.text(.97,.963,'',ha='right',fontsize=14,weight='bold')
status=fig.text(.075,.031,'',fontsize=10)
blue,orange,green='#2463c5','#d37521','#008577'
soliton,=a.plot(x,np.abs(solution[0])**2,color=blue,lw=2.4,label='NLSE soliton')
analytic,=a.plot(x,np.abs(exact[0])**2,color='#253449',lw=1.3,ls='--',label='Exact soliton')
dispersive,=a.plot(x,np.abs(linear[0])**2,color=orange,lw=1.8,alpha=.85,label='Linear evolution · same initial pulse')
a.set(xlim=(-12,12),ylim=(0,1.12),xlabel='x',ylabel=r'$|\psi|^2$',title='Nonlinearity balances dispersion')
a.legend(loc='upper right',frameon=False,fontsize=10)
xi=np.linspace(-6,6,401)
b.plot(xi,1/np.cosh(xi)**2,'--',color='#253449',lw=1.3,label='Initial sech² shape')
moving,=b.plot(x-center[0],np.abs(solution[0])**2,color=blue,lw=2,label='NLSE in moving frame')
b.set(xlim=(-6,6),ylim=(0,1.12),xlabel=r'$\xi=x-x_c(t)$',ylabel='Density',title='The shape stays unchanged')
b.legend(loc='upper right',frameon=False,fontsize=9)
real,=c.plot(x,solution[0].real,color=blue,lw=1.7,label='Real')
imag,=c.plot(x,solution[0].imag,color=green,lw=1.7,label='Imaginary')
upper,=c.plot(x,np.abs(exact[0]),color='#8996a8',ls=':',lw=1.2)
lower,=c.plot(x,-np.abs(exact[0]),color='#8996a8',ls=':',lw=1.2)
c.set(xlim=(-9,7),ylim=(-1.12,1.12),xlabel='x',ylabel=r'$\psi$',title='Carrier phase within the sech envelope')
c.legend(loc='upper right',frameon=False,ncol=2,fontsize=10)
for ax in (a,b,c):ax.grid(alpha=.15)
ffmpeg=Path('/opt/homebrew/bin/ffmpeg')
if ffmpeg.exists():matplotlib.rcParams['animation.ffmpeg_path']=str(ffmpeg)
else:
 import imageio_ffmpeg
 matplotlib.rcParams['animation.ffmpeg_path']=imageio_ffmpeg.get_ffmpeg_exe()
output=root/'examples/pde/results/nlse_soliton.mp4'
output.parent.mkdir(parents=True,exist_ok=True)
writer=FFMpegWriter(fps=25,codec='libx264',metadata={'title':'JAX BSPF focusing NLSE bright soliton','comment':'321 computed output states, t=0..4; dt=0.0025; linear comparison and analytic soliton.'},extra_args=['-crf','19','-pix_fmt','yuv420p','-movflags','+faststart'])
with writer.saving(fig,str(output),dpi=100):
 for i,t in enumerate(times):
  soliton.set_ydata(np.abs(solution[i])**2);analytic.set_ydata(np.abs(exact[i])**2)
  dispersive.set_ydata(np.abs(linear[i])**2)
  moving.set_data(x-center[i],np.abs(solution[i])**2)
  real.set_ydata(solution[i].real);imag.set_ydata(solution[i].imag)
  upper.set_ydata(np.abs(exact[i]));lower.set_ydata(-np.abs(exact[i]))
  clock.set_text(f't = {t:.3f}')
  status.set_text(f'Center: {center[i]:+.2f}   |   Width: {width[i]:.6f}   |   Max field error: {error[i]:.2e}   |   Relative norm drift: {norm[i]/norm[0]-1:+.2e}')
  writer.grab_frame()
  if i==240:fig.savefig(output.with_suffix('.png'),dpi=100)
plt.close(fig)
print(output,flush=True)
