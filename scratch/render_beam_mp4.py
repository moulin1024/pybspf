"""Render the corrected, validated Euler--Bernoulli notebook without a duplicate solver."""
from pathlib import Path
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Rectangle

root=Path(__file__).resolve().parents[1]
nb=json.loads((root/'examples/pde/euler_bernoulli_1d.ipynb').read_text())
cells=[c for c in nb['cells'] if c['cell_type']=='code']
ns={}
# Execute the actual notebook, including its numerical assertions, before rendering.
exec('\n\n'.join(''.join(c['source']) for c in cells[:-1]),ns)
jax,jnp,bspf,plt=(ns[k] for k in ('jax','jnp','bspf','plt'))
x,times,displacement,exact,energy,static=map(np.asarray,
    (ns['x'],ns['times'],ns['displacement'],ns['exact'],ns['energy'],ns['static_exact']))
error=displacement-exact;drift=energy/energy[0]-1
assert np.isfinite(displacement).all()
assert np.max(np.abs(error))<5e-9
assert np.max(np.abs(drift))<1e-9
print(f'{len(times)} computed states; max displacement error={np.max(np.abs(error)):.4e}; energy drift={np.max(np.abs(drift)):.4e}',flush=True)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
 'axes.spines.right':False,'axes.titleweight':'bold','figure.facecolor':'#f6f8fc',
 'axes.facecolor':'white','axes.labelcolor':'#24344b','text.color':'#24344b',
 'xtick.color':'#52627a','ytick.color':'#52627a'})
fig=plt.figure(figsize=(12.8,8),dpi=100)
grid=fig.add_gridspec(2,2,left=.085,right=.97,bottom=.105,top=.83,hspace=.43,wspace=.27)
a=fig.add_subplot(grid[0,:]);b=fig.add_subplot(grid[1,0]);c=fig.add_subplot(grid[1,1])
fig.suptitle('Euler–Bernoulli beam · loaded cantilever',x=.085,y=.968,ha='left',fontsize=21,weight='bold')
fig.text(.085,.91,'JAX BSPF  |  All four boundary conditions · resolved quadrature · exact modal evolution',fontsize=11)
clock=fig.text(.97,.966,'',ha='right',fontsize=14,weight='bold')
status=fig.text(.085,.028,'',fontsize=10)
blue,orange='#2463c5','#d37521'
a.axhline(0,color='#b1bac7',lw=.8)
a.add_patch(Rectangle((-.035,-.04),.027,.13,facecolor='#dae0e9',edgecolor='#52627a',hatch='////',lw=.8))
a.plot([0,0],[-.035,.085],color='#52627a',lw=3)
for pos in (.16,.34,.52,.70,.88):
 a.annotate('',xy=(pos,-.003),xytext=(pos,-.037),arrowprops=dict(arrowstyle='->',color='#8996a8',lw=1.2))
a.text(.5,-.047,'Uniform load applied at t = 0',ha='center',va='bottom',fontsize=10,color='#52627a')
beam,=a.plot(x,displacement[0],color=blue,lw=3,label='JAX beam')
ref,=a.plot(x,exact[0],color=orange,lw=1.5,ls='--',label='256-mode analytical reference')
a.plot(x,static,':',color='#8996a8',lw=1.4,label='Static deflection')
tip,=a.plot([1],[displacement[0,-1]],'o',color=blue,ms=5)
a.set(xlim=(-.045,1.03),ylim=(max(displacement.max(),exact.max())*1.12,-.065),xlabel='x',ylabel='Downward deflection w',title='Cantilever motion under a constant load')
a.legend(loc='lower left',frameon=False,fontsize=9)
b.plot(times,exact[:,-1],color=orange,ls='--',lw=1.5,label='Reference')
trace,=b.plot([],[],color=blue,lw=2,label='JAX')
point,=b.plot([],[],'o',color=blue,ms=5)
b.set(xlim=(0,3),ylim=(-.01,exact[:,-1].max()*1.1),xlabel='t',ylabel='Tip displacement',title='Tip response')
b.legend(loc='upper right',frameon=False,fontsize=10)
err,=c.plot(x,error[0],color=orange,lw=1.8)
c.axhline(0,color='#b1bac7',lw=.8)
limit=np.max(np.abs(error))*1.15
c.set(xlim=(0,1),ylim=(-limit,limit),xlabel='x',ylabel='w − reference',title='Signed displacement error')
c.ticklabel_format(axis='y',style='sci',scilimits=(0,0),useMathText=True)
for ax in (a,b,c):ax.grid(alpha=.15)
ffmpeg=Path('/opt/homebrew/bin/ffmpeg')
if ffmpeg.exists():matplotlib.rcParams['animation.ffmpeg_path']=str(ffmpeg)
else:
 import imageio_ffmpeg
 matplotlib.rcParams['animation.ffmpeg_path']=imageio_ffmpeg.get_ffmpeg_exe()
output=root/'examples/pde/results/euler_bernoulli_beam_corrected.mp4';output.parent.mkdir(parents=True,exist_ok=True)
writer=FFMpegWriter(fps=25,codec='libx264',metadata={'title':'JAX BSPF Euler-Bernoulli cantilever','comment':'Corrected BSPF cantilever: resolved quadrature, SVD modes, exact time evolution, 256-mode reference.'},extra_args=['-crf','19','-pix_fmt','yuv420p','-movflags','+faststart'])
with writer.saving(fig,str(output),dpi=100):
 for i,t in enumerate(times):
  beam.set_ydata(displacement[i]);ref.set_ydata(exact[i]);tip.set_ydata([displacement[i,-1]])
  trace.set_data(times[:i+1],displacement[:i+1,-1]);point.set_data([t],[displacement[i,-1]])
  err.set_ydata(error[i]);clock.set_text(f't = {t:.3f}')
  status.set_text(f'Max displacement error: {np.max(np.abs(error[i])):.2e}   |   Relative energy drift: {drift[i]:+.2e}   |   Reference: 256 cantilever modes')
  writer.grab_frame()
  if i==90:fig.savefig(output.with_suffix('.png'),dpi=100)
plt.close(fig)
# Refresh the original link as well as the explicitly corrected artifact.
for suffix in ('.mp4', '.png'):
 shutil.copyfile(output.with_suffix(suffix), output.parent/('euler_bernoulli_beam'+suffix))
print(output,flush=True)
