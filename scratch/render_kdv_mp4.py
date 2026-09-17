"""Animate the nonperiodic KdV notebook with computed intermediate states."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.animation import FFMpegWriter

root=Path(__file__).resolve().parents[1]
nb=json.loads((root/'examples/pde/kdv_1d.ipynb').read_text())
cells=[c for c in nb['cells'] if c['cell_type']=='code']
ns={}
exec(''.join(cells[0]['source']),ns)
exec(''.join(cells[1]['source']).split('\nx, spatial, plan = setup')[0],ns)
jax,jnp,bspf,plt=(ns[k] for k in ('jax','jnp','bspf','plt'))
reference,boundary=ns['reference'],ns['boundary']
x,spatial,plan=ns['setup'](129)
times=jnp.linspace(0.,2.,321)
solution=jax.jit(lambda:bspf.integrate_kdv(plan,reference(x,0.),times,boundary=boundary,substeps=40))()
# 0.00625 output interval / 40 = 0.00015625, identical to the notebook.
exact=reference(x[None,:],times[:,None]);prescribed=jax.vmap(boundary)(times)
slope=bspf.differentiate(spatial,solution.T)[-1]
qfield=solution[:,plan.free]@plan.values.T+prescribed[:,:2]@plan.boundary_values.T
mass=qfield@plan.quadrature_weights
z=jnp.sqrt(ns['speed'])/2*(ns['ends'][None,:]-ns['x0']-ns['speed']*times[:,None])
exact_mass=jnp.sqrt(ns['speed'])*(jnp.tanh(z[:,1])-jnp.tanh(z[:,0]))
x,times,solution,exact,prescribed,slope,mass,exact_mass=map(np.asarray,(x,times,solution,exact,prescribed,slope,mass,exact_mass))
error=np.max(np.abs(solution-exact),axis=1)
slope_error=np.abs(slope-prescribed[:,2])
assert error.max()<2e-9
assert slope_error.max()<2e-8
assert np.max(np.abs(mass-exact_mass))<5e-9
assert np.max(np.abs(solution[:,[0,-1]]-prescribed[:,:2]))<1e-14
print(f'321 computed states; max field error={error.max():.4e}; slope residual={slope_error.max():.4e}',flush=True)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
 'axes.spines.right':False,'axes.titleweight':'bold','figure.facecolor':'#f6f8fc',
 'axes.facecolor':'white','axes.labelcolor':'#24344b','text.color':'#24344b',
 'xtick.color':'#52627a','ytick.color':'#52627a'})
fig=plt.figure(figsize=(12.8,8),dpi=100)
grid=fig.add_gridspec(2,2,left=.085,right=.97,bottom=.11,top=.84,hspace=.42,wspace=.27)
a=fig.add_subplot(grid[0,:]);b=fig.add_subplot(grid[1,0]);c=fig.add_subplot(grid[1,1])
fig.suptitle('KdV soliton · nonperiodic boundaries',x=.085,y=.968,ha='left',fontsize=21,weight='bold')
fig.text(.085,.912,r'$u_t+6uu_x+u_{xxx}=0$  |  JAX BSPF · 129 samples · prescribed values and right slope',fontsize=11)
clock=fig.text(.97,.966,'',ha='right',fontsize=14,weight='bold')
status=fig.text(.085,.027,'',fontsize=10)
blue,orange,green='#2463c5','#d37521','#008577'
field,=a.plot(x,solution[0],color=blue,lw=2.4,label='JAX BSPF')
ref,=a.plot(x,exact[0],color='#253449',lw=1.3,ls='--',label='Analytical soliton')
endpoints,=a.plot(x[[0,-1]],solution[0,[0,-1]],'o',color=orange,ms=6,clip_on=False,label='Prescribed endpoint values')
a.set(xlim=(-6,6),ylim=(-.025,1.12),xlabel='x',ylabel='u',title='Traveling wave on the finite interval [−6, 6]')
a.legend(loc='upper right',frameon=False,fontsize=10)
boundary_lines=[];boundary_dots=[]
for i,color,label in [(0,blue,'u(left)'),(1,orange,'u(right)'),(2,green,'u_x(right)')]:
 b.plot(times,prescribed[:,i],color=color,alpha=.16,lw=1.3)
 line,=b.plot([],[],color=color,lw=2,label=label);dot,=b.plot([],[],'o',color=color,ms=5)
 boundary_lines.append(line);boundary_dots.append(dot)
b.set(xlim=(0,2),ylim=(-.022,.017),xlabel='t',ylabel='Boundary data',title='Unequal, time-dependent endpoints')
b.legend(loc='lower left',frameon=False,fontsize=9)
c.plot(times,exact_mass,color='#8996a8',ls='--',lw=1.5,label='Exact finite-interval mass')
trace,=c.plot([],[],color=blue,lw=2,label='JAX mass')
point,=c.plot([],[],'o',color=blue,ms=5)
c.set(xlim=(0,2),ylim=(exact_mass.min()-.0005,exact_mass.max()+.0005),xlabel='t',ylabel=r'$\int_{-6}^{6} u\,dx$',title='Mass changes through the boundaries')
c.ticklabel_format(axis='y',useOffset=False)
c.legend(loc='lower center',frameon=False,fontsize=9)
for ax in (a,b,c):ax.grid(alpha=.15)
ffmpeg=Path('/opt/homebrew/bin/ffmpeg')
if ffmpeg.exists():matplotlib.rcParams['animation.ffmpeg_path']=str(ffmpeg)
else:
 import imageio_ffmpeg
 matplotlib.rcParams['animation.ffmpeg_path']=imageio_ffmpeg.get_ffmpeg_exe()
output=root/'examples/pde/results/kdv_nonperiodic.mp4';output.parent.mkdir(parents=True,exist_ok=True)
writer=FFMpegWriter(fps=25,codec='libx264',metadata={'title':'JAX BSPF nonperiodic KdV soliton','comment':'Computed states on [-6,6], t=0..2; time-dependent Dirichlet data and right-slope boundary load.'},extra_args=['-crf','19','-pix_fmt','yuv420p','-movflags','+faststart'])
with writer.saving(fig,str(output),dpi=100):
 for i,t in enumerate(times):
  field.set_ydata(solution[i]);ref.set_ydata(exact[i]);endpoints.set_ydata(solution[i,[0,-1]])
  for k in range(3):
   boundary_lines[k].set_data(times[:i+1],prescribed[:i+1,k]);boundary_dots[k].set_data([t],[prescribed[i,k]])
  trace.set_data(times[:i+1],mass[:i+1]);point.set_data([t],[mass[i]])
  clock.set_text(f't = {t:.3f}')
  status.set_text(f'Max field error: {error[i]:.2e}   |   Right-slope residual: {slope_error[i]:.2e}   |   Mass error: {abs(mass[i]-exact_mass[i]):.2e}')
  writer.grab_frame()
  if i==240:fig.savefig(output.with_suffix('.png'),dpi=100)
plt.close(fig)
print(output,flush=True)
