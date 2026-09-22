"""Render saved response-enriched NS fields with fixed scales and no filtering."""
import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Ellipse
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--fps',type=int,default=10)
    parser.add_argument('--ffmpeg',default='ffmpeg',help='FFmpeg executable name or absolute path')
    args=parser.parse_args()
    if args.fps<1:
        parser.error('fps must be positive')
    ffmpeg=shutil.which(args.ffmpeg)
    if ffmpeg is None:
        raise RuntimeError('ffmpeg is required')
    matplotlib.rcParams['animation.ffmpeg_path']=ffmpeg
    with np.load(args.out/'fields.npz') as data:
        x,y,t,fields,center,axes=[data[k] for k in ('x','y','t','fields','center','axes')]
    report=json.loads((args.out/'report.json').read_text())
    reynolds=report.get('reynolds',20.)
    speed=np.hypot(fields[:,0],fields[:,1])
    omega=fields[:,2]
    perturbation=omega-2*y[None,:,None]
    sx=(x>-.85)&(x<-.45)
    sy=(y[2:-2]>-.8)&(y[2:-2]<.8)
    rough=np.diff(omega,n=4,axis=1)[:,sy][:,:,sx]
    if not np.all(np.isfinite(rough)):
        raise RuntimeError('Non-finite values in upstream diagnostic region')
    rms=np.sqrt(np.mean(rough**2,axis=(1,2)))
    vmax=float(np.nanmax(speed))
    wmax=max(.25,float(np.nanpercentile(abs(perturbation),99)))
    rmax=max(1e-12,float(np.max(abs(rough))))
    fig,axs=plt.subplots(2,2,figsize=(14,8),layout='constrained')
    top=axs[0]
    for ax in top:
        ax.set(xlim=(x[0],x[-1]),ylim=(y[0],y[-1]),xlabel='x',ylabel='y',aspect='equal')
        ax.add_patch(Ellipse(center,*(2*axes),facecolor='.7',edgecolor='k',zorder=3))
    im0=top[0].pcolormesh(x,y,speed[0],shading='nearest',cmap='viridis',vmin=0,vmax=vmax,rasterized=True)
    im1=top[1].pcolormesh(x,y,perturbation[0],shading='nearest',cmap='RdBu_r',vmin=-wmax,vmax=wmax,rasterized=True)
    top[0].set_title('Speed')
    top[1].set_title('Vorticity minus inlet shear (clipped fixed scale)')
    fig.colorbar(im0,ax=top[0],orientation='horizontal',shrink=.85,pad=.08)
    fig.colorbar(im1,ax=top[1],orientation='horizontal',shrink=.85,pad=.08,extend='both')
    im2=axs[1,0].pcolormesh(x[sx],y[2:-2][sy],rough[0],shading='nearest',cmap='RdBu_r',vmin=-rmax,vmax=rmax)
    axs[1,0].set(xlabel='x',ylabel='y',title='Upstream fourth y-difference of vorticity')
    fig.colorbar(im2,ax=axs[1,0],label='Unscaled difference')
    axs[1,1].semilogy(t,rms,color='.75',lw=1)
    trajectory,=axs[1,1].semilogy([],[],color='tab:blue',lw=2)
    dot,=axs[1,1].semilogy([],[],'o',color='tab:blue')
    axs[1,1].set(xlabel='Time',ylabel='Fourth-difference RMS',xlim=(0,float(t[-1])),title='Upstream roughness (not an NS error norm)')
    axs[1,1].grid(alpha=.25)
    title=fig.suptitle('')
    writer=FFMpegWriter(fps=args.fps,codec='libx264',bitrate=2500,
                        extra_args=['-pix_fmt','yuv420p','-movflags','+faststart'])
    output=args.out/f'{report["family"]}_response_re{reynolds:g}_t{t[-1]:g}_{len(t)}frames.mp4'
    def update(k):
        im0.set_array(speed[k].ravel())
        im1.set_array(perturbation[k].ravel())
        im2.set_array(rough[k].ravel())
        trajectory.set_data(t[:k+1],rms[:k+1])
        dot.set_data([t[k]],[rms[k]])
        title.set_text(f'{report["family"].capitalize()} enrichment | Re={reynolds:g} | t={t[k]:.2f} | dt={report["dt"]:g}\n'
                       f'Frame {k+1}/{len(t)} | Fixed scales, no filtering or sponge')
    with writer.saving(fig,str(output),dpi=120):
        for k in range(len(t)):
            update(k)
            writer.grab_frame()
            if k==0:
                fig.savefig(args.out/'movie_first_frame.png',dpi=120)
            if (k+1)%10==0:
                print(f'FRAME {k+1}/{len(t)}',flush=True)
        fig.savefig(args.out/'movie_last_frame.png',dpi=120)
    plt.close(fig)
    metadata=dict(file=output.name,frames=len(t),fps=args.fps,duration_seconds=len(t)/args.fps,
                  first_time=float(t[0]),last_time=float(t[-1]),speed_max=vmax,
                  perturbation_color_limit=wmax,roughness_color_limit=rmax,
                  final_roughness=float(rms[-1]),maximum_roughness=float(np.max(rms)))
    (args.out/'movie.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))


if __name__=='__main__':
    main()
