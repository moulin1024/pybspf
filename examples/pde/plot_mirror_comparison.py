"""Plot saved linear-f/log-f mirror results without rerunning the solver."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, default=Path('build/mirror_negativity/linear_f_solution.npz'))
    parser.add_argument('--after', type=Path, default=Path('build/drift_kinetic_mirror/solution.npz'))
    parser.add_argument('--out', type=Path, default=Path('build/drift_kinetic_mirror'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with np.load(args.before) as f:
        old = dict(f)
    with np.load(args.after) as f:
        new = dict(f)
    for key in ('times', 'z', 'v', 'mu'):
        np.testing.assert_array_equal(old[key], new[key])
    families = {f.name for f in font_manager.fontManager.ttflist}
    if 'Arial Unicode MS' in families:
        plt.rcParams['font.family'] = 'Arial Unicode MS'
    plt.rcParams.update({'font.size': 11, 'axes.unicode_minus': False,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': '#fafbfc', 'axes.facecolor': 'white'})
    red, blue, green = '#c84b4b', '#176ca4', '#39834d'
    t, z, v, mu = (new[k] for k in ('times', 'z', 'v', 'mu'))
    # Analytic characteristic reference, evaluated independently with NumPy.
    reference = np.empty_like(new['history'])
    omega = np.sqrt(mu)[None, None, :]
    zz, vv = z[:, None, None], v[None, :, None]
    for i, time in enumerate(t):
        c, s = np.cos(omega*time), np.sin(omega*time)
        z0, v0 = zz*c-vv*s/omega, vv*c+omega*zz*s
        reference[i] = np.exp(-4*z0*z0-6*(v0-.8)**2-12*(mu[None,None,:]-1)**2)
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 8.5), layout='constrained')
    fig.set_constrained_layout_pads(h_pad=.12, w_pad=.08, hspace=.09, wspace=.1)
    fig.suptitle('磁镜模拟：修复前后对照', fontsize=19, weight='bold')
    for data, label, color in ((old, '原线性 f', red), (new, '对数表示', blue)):
        ax[0,0].plot(t, data['history'].min(axis=(1,2,3)), label=label, color=color, lw=2)
        error = np.max(np.abs(data['history']-reference), axis=(1,2,3))
        # Exclude initial zeros on a logarithmic axis; no fabricated error floor.
        ax[0,1].semilogy(t[1:], error[1:], label=label, color=color, lw=2)
    ax[0,0].axhline(0, color='#444', lw=.7, zorder=0)
    ax[0,0].set(title='各输出时刻的最小节点值', xlabel='时间 t', ylabel='min f')
    ax[0,0].text(.04,.12,'原最小值：−1.066×10⁻⁷\n修复后：无负值；远尾部下溢为零',transform=ax[0,0].transAxes,fontsize=10)
    ax[0,1].set(title='分布函数对解析解的最大绝对误差', xlabel='时间 t', ylabel='max |f − f_exact|')
    it, iz, iv, im = np.unravel_index(old['history'].argmin(), old['history'].shape)
    mask = (z >= z[iz]-.5)&(z <= z[iz]+.125)
    ax[1,0].plot(z[mask], old['history'][it,mask,iv,im], 'o-', ms=4, color=red, label='原线性 f')
    ax[1,0].plot(z[mask], new['history'][it,mask,iv,im], 's-', ms=3, color=blue, label='对数表示')
    ax[1,0].plot(z[mask], reference[it,mask,iv,im], '--', color=green, label='解析解', lw=2)
    ax[1,0].axhline(0, color='#444', lw=.7)
    ax[1,0].set(title=f'原最严重负值附近：t={t[it]:.2f}, v={v[iv]:.3f}, μ={mu[im]:.3f}', xlabel='z', ylabel='f（原始数值，无裁剪）')
    for data, label, color in ((old, '原线性 f', red), (new, '对数表示', blue)):
        rel = data['balance']/data['moments'][0,[0,3]]
        ax[1,1].plot(t, rel[:,0], color=color, label=label+'：粒子')
        ax[1,1].plot(t, rel[:,1], '--', color=color, label=label+'：能量')
    ax[1,1].set(title='计入全部边界通量的相对收支残差', xlabel='时间 t', ylabel='相对残差')
    for panel in ax.flat:
        panel.grid(alpha=.18)
        panel.legend(fontsize=9, framealpha=.9)
    fig.savefig(args.out/'negativity_comparison.png', dpi=180)
    fig.savefig(args.out/'negativity_comparison.pdf')
    plt.close(fig)

    fig = plt.figure(figsize=(12.5, 11), layout='constrained')
    fig.set_constrained_layout_pads(h_pad=.1, w_pad=.07, hspace=.06, wspace=.08)
    fig.suptitle('修复后的磁镜运动与能量交换', fontsize=19, weight='bold')
    gs = fig.add_gridspec(3, 2)
    k = int(np.argmin(abs(mu-1)))
    turning = np.pi/(2*np.sqrt(mu[k]))
    snapshots = [int(np.argmin(abs(t-t0))) for t0 in (0,turning,2*turning,4)]
    # Shared scale across all snapshots; crop display around the trapped packet.
    zm, vm = abs(z)<=2, abs(v)<=2
    vmax = new['history'][...,k].max()
    panels = []
    for slot, idx in enumerate(snapshots):
        panel = fig.add_subplot(gs[slot//2,slot%2]); panels.append(panel)
        values = new['history'][idx,:,:,k][np.ix_(zm,vm)]
        mesh = panel.pcolormesh(z[zm],v[vm],values.T,shading='auto',cmap='viridis',vmin=0,vmax=vmax,rasterized=True)
        panel.plot(new['mean_z'][idx],new['mean_v'][idx],'+',color='white',ms=12,mew=1.8)
        panel.set(title=f't={t[idx]:.2f}，μ={mu[k]:.3f}',xlabel='位置 z',ylabel='平行速度 v')
    fig.colorbar(mesh, ax=panels, label='f（四幅图使用同一色标）', fraction=.025, pad=.02)
    a = fig.add_subplot(gs[2,0]); a.plot(t,new['mean_z'],color=blue,label='数值平均位置')
    a.plot(t,new['orbit_z'],'--',color='#efad50',label='解析位置')
    a.plot(t,new['mean_v'],color=green,label='数值平均速度')
    a.plot(t,new['orbit_v'],'--',color=red,label='解析速度')
    a.axvline(turning,color='#888',ls=':',lw=1)
    a.set(title='质心运动：速度反号表明磁镜反射',xlabel='时间 t',ylabel='位置 / 速度')
    a.legend(fontsize=9,ncol=2);a.grid(alpha=.18)
    a=fig.add_subplot(gs[2,1])
    for j,label,color in ((1,'平行能量',blue),(2,'垂直能量','#dc9232'),(3,'总能量',green)):
        a.plot(t,new['moments'][:,j],label=label,color=color,lw=2)
    a.set(title='平行、垂直能量交换',xlabel='时间 t',ylabel='归一化能量')
    a.legend(fontsize=9);a.grid(alpha=.18)
    fig.savefig(args.out/'mirror_evolution.png',dpi=180)
    fig.savefig(args.out/'mirror_evolution.pdf')
    plt.close(fig)
    print(args.out.resolve())


if __name__ == '__main__':
    main()
