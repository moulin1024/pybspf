"""Animate saved log-BSPF mirror states, without temporal interpolation."""
import argparse
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.animation import FFMpegWriter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('build/drift_kinetic_mirror/solution.npz'))
    parser.add_argument('--output', type=Path, default=Path('build/drift_kinetic_mirror/mirror_dynamics.mp4'))
    args = parser.parse_args()
    with np.load(args.input) as f:
        data = dict(f)
    t,z,v,mu,f,mom = (data[k] for k in ('times','z','v','mu','history','moments'))
    assert np.isfinite(f).all() and f.min() >= 0
    assert np.allclose(np.diff(t), np.diff(t)[0])
    ffmpeg = str(Path('/opt/homebrew/bin/ffmpeg')) if Path('/opt/homebrew/bin/ffmpeg').exists() else shutil.which('ffmpeg')
    if ffmpeg is None:
        raise RuntimeError('ffmpeg is required')
    matplotlib.rcParams['animation.ffmpeg_path'] = ffmpeg
    if 'Arial Unicode MS' in {font.name for font in font_manager.fontManager.ttflist}:
        plt.rcParams['font.family'] = 'Arial Unicode MS'
    plt.rcParams.update({'font.size':11,'axes.unicode_minus':False,
        'axes.spines.top':False,'axes.spines.right':False,
        'figure.facecolor':'#f5f7fb','axes.facecolor':'white','text.color':'#223247',
        'axes.labelcolor':'#223247','xtick.color':'#223247','ytick.color':'#223247'})
    blue, orange, green = '#1873b5','#d8952a','#358556'
    k = int(np.argmin(abs(mu-1)))
    fig = plt.figure(figsize=(12,8),dpi=120)
    grid=fig.add_gridspec(2,2,left=.08,right=.94,bottom=.13,top=.82,hspace=.47,wspace=.32)
    phase,orbit,energy,balance=(fig.add_subplot(grid[i,j]) for i,j in ((0,0),(0,1),(1,0),(1,1)))
    fig.text(.08,.955,'BSPF 漂移动理学 · 磁镜反射',fontsize=22,weight='bold')
    fig.text(.08,.906,f'正性表示 f = exp(g)   |   B(z) = 1 + z²/2   |   μ = {mu[k]:.4f}   |   无裁剪',fontsize=11)
    clock=fig.text(.94,.953,'',ha='right',fontsize=16)
    stage=fig.text(.94,.87,'',ha='right',fontsize=12,color=blue)
    zm,vm=abs(z)<=2,abs(v)<=2
    heat=phase.pcolormesh(z[zm],v[vm],f[0,:,:,k][np.ix_(zm,vm)].T,
        shading='auto',cmap='viridis',vmin=0,vmax=f[...,k].max())
    phase.set(xlim=(-2,2),ylim=(-2,2),xlabel='位置 z',ylabel='平行速度 v',title='分布函数 f(z,v)：局部视图')
    phase.plot(data['orbit_z'],data['orbit_v'],'--',color='white',alpha=.35,lw=1)
    center,=phase.plot([],[],'+',color='white',ms=13,mew=2)
    pos=phase.get_position()
    cax=fig.add_axes([pos.x1+.009,pos.y0,.01,pos.height])
    fig.colorbar(heat,cax=cax,label='f')
    orbit.plot(t,data['orbit_z'],'--',color=blue,alpha=.3,label='解析位置')
    orbit.plot(t,data['orbit_v'],'--',color=green,alpha=.3,label='解析速度')
    zline,=orbit.plot([],[],color=blue,lw=2,label='数值位置')
    vline,=orbit.plot([],[],color=green,lw=2,label='数值速度')
    orbit.axhline(0,color='#999',lw=.7)
    orbit.set(xlim=(0,4),ylim=(-1,1),xlabel='模拟时间 t',ylabel='质心位置 / 速度',title='质心运动与解析轨道')
    orbit.legend(fontsize=8.5,ncol=2,loc='lower left',framealpha=.9)
    elines=[]
    for j,label,color in ((1,'平行能量',blue),(2,'垂直能量',orange),(3,'总能量',green)):
        line,=energy.plot([],[],color=color,lw=2,label=label);elines.append((j,line))
    energy.set(xlim=(0,4),ylim=(0,float(mom[:,3].max()*1.12)),xlabel='模拟时间 t',ylabel='归一化能量',title='全部 μ 切片的能量交换')
    energy.legend(fontsize=9,loc='center right')
    rel=data['balance']/mom[0,[0,3]]
    nline,=balance.plot([],[],color=blue,lw=2,label='粒子收支')
    hline,=balance.plot([],[],'--',color=orange,lw=2,label='总能量收支')
    extent=max(abs(rel).max()*1.18,1e-14)
    balance.set(xlim=(0,4),ylim=(-.08*extent,extent),xlabel='模拟时间 t',ylabel='相对残差',title='计入全部边界通量的收支检查')
    balance.legend(fontsize=9,loc='upper left')
    cursors=[ax.axvline(0,color='#a0aab7',ls=':',lw=1) for ax in (orbit,energy,balance)]
    for ax in (orbit,energy,balance): ax.grid(alpha=.15)
    status=fig.text(.08,.061,'',fontsize=10)
    fig.text(.08,.028,'81 个实际计算状态 · 无时间插值 · 原网格 65 × 65 × 12 · 白色十字为质心',fontsize=9,color='#637086')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    # 8 computed states/second. fps converts to 24 fps by repeating frames;
    # tpad holds the first/last state for .75 s without synthesizing data.
    writer=FFMpegWriter(fps=8,codec='libx264',metadata={
        'title':'BSPF 1z2v magnetic mirror dynamics',
        'comment':'81 saved computed states, t=0..4; no temporal interpolation; log representation; 65x65x12 grid.'},
        extra_args=['-vf','fps=24,tpad=start_duration=0.75:stop_duration=0.75:start_mode=clone:stop_mode=clone',
                    '-crf','18','-pix_fmt','yuv420p','-movflags','+faststart'])
    with writer.saving(fig,str(args.output),dpi=120):
        for i,time in enumerate(t):
            heat.set_array(f[i,:,:,k][np.ix_(zm,vm)].T.ravel())
            center.set_data([data['mean_z'][i]],[data['mean_v'][i]])
            zline.set_data(t[:i+1],data['mean_z'][:i+1]);vline.set_data(t[:i+1],data['mean_v'][:i+1])
            for j,line in elines: line.set_data(t[:i+1],mom[:i+1,j])
            nline.set_data(t[:i+1],rel[:i+1,0]);hline.set_data(t[:i+1],rel[:i+1,1])
            for line in cursors: line.set_xdata([time,time])
            clock.set_text(f't = {time:.2f}')
            turning=np.pi/(2*np.sqrt(mu[k]))
            stage.set_text('向右运动 · 平行能量转为垂直能量' if time<turning-.15 else
                           '磁镜转向 · 平行速度接近零' if time<turning+.15 else '反射后返回 · 平行速度为负')
            status.set_text(f'最小 f = {f[i].min():.2e}（远尾部允许下溢为零）   |   粒子残差 {rel[i,0]:.2e}   |   能量残差 {rel[i,1]:.2e}')
            writer.grab_frame()
            if i==int(np.argmin(abs(t-turning))): fig.savefig(args.output.with_suffix('.png'),dpi=120)
    plt.close(fig)
    print(args.output.resolve(),flush=True)


if __name__=='__main__':
    main()
