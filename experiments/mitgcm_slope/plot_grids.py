"""Plot actual BSPF construction coordinates versus MITgcm output geometry."""
from pathlib import Path
import argparse
import json
import os
import sys
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mitgcm-grid-matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'examples/pde/isw_slope/source'))
from common import geometry, depth


def plot(run, bspf_run, out):
    out.mkdir(parents=True, exist_ok=True)
    font = Path('/System/Library/Fonts/PingFang.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    cfg = json.loads((bspf_run/'config.json').read_text())
    nx, nz = cfg['nx'], cfg['nz']
    # These are the construction points from build_generic_bspf / ClosedLine;
    # do not confuse them with the Gauss points used for nonlinear quadrature.
    geo = geometry(np.linspace(0, 1, nx), np.linspace(0, 1, nz))
    xb, zb = geo['x']*100, geo['z']*100
    case = json.loads((run/'case.json').read_text())
    ncol, nlev = case['allocated_x_columns'], case['z_levels']
    start=case.get('dry_left_columns',1)
    end=start+case['wet_x_columns']
    hfac = np.fromfile(run/'hFacC.001.001.data', dtype='>f8').reshape(nlev,ncol)[:,start:end]
    xf = np.fromfile(run/'XG.001.001.data', dtype='>f8')[start:end+1]
    zf = np.fromfile(run/'RF.data', dtype='>f8')
    dz = -np.diff(zf)
    bed = -np.sum(hfac*dz[:,None], axis=0)
    assert xf.size == case['wet_x_columns']+1 and zf.size == nlev+1
    assert hfac.shape == (nlev, case['wet_x_columns'])
    fine_x = np.linspace(0, 1800, 5001)
    exact_bed = -100*depth(fine_x/100)[0]
    zoom = (830, 870, -82, -62)
    colors = dict(bspf='#13887f', mit='#3466a8', ground='#e5ddd0', partial='#e8a64d')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.1))
    fig.subplots_adjust(left=.075, right=.985, top=.84, bottom=.145, hspace=.40, wspace=.17)
    fig.suptitle('BSPF 与 MITgcm：实际计算网格对比', x=.075, y=.975, ha='left', fontsize=21, weight='bold')
    fig.text(.075, .923, '同一斜坡与水平加密映射  ·  二维 x–z（MITgcm Ny = 1）  ·  坐标单位：m', color='#475569', fontsize=12)
    fig.text(.075, .884, 'BSPF：321 × 161 构造点；642 × 322 Gauss 积分点', fontsize=11, color=colors['bspf'])
    fig.text(.56, .884, f'MITgcm：321 水柱 × 161 层；{np.count_nonzero(hfac):,} 个湿网格', fontsize=11, color=colors['mit'])

    def indices(n, stride):
        return np.unique(np.r_[np.arange(0,n,stride), n-1])

    def bspf(ax, stride, bounds):
        seg = []
        for i in indices(nx, stride):
            if bounds[0] <= xb[i] <= bounds[1]:
                seg.append(np.column_stack((np.full(nz, xb[i]), zb[i])))
        for j in indices(nz, stride):
            seg.append(np.column_stack((xb, zb[:,j])))
        ax.add_collection(LineCollection(seg, colors=colors['bspf'], linewidths=.55, alpha=.75))
        ax.fill_between(fine_x, bounds[2], exact_bed, color=colors['ground'], zorder=0)
        ax.plot(fine_x, exact_bed, color='#524638', lw=1.5, zorder=5)
        if stride == 1:
            mask=(xb>=bounds[0]) & (xb<=bounds[1])
            xx=np.broadcast_to(xb[mask,None],zb[mask].shape)
            ax.scatter(xx,zb[mask],s=2.5,color=colors['bspf'],alpha=.65,zorder=3)

    def mit(ax, stride, bounds):
        seg=[]
        # Vertical face only extends through its neighboring wet geometry.
        for i in indices(xf.size, stride):
            if not bounds[0] <= xf[i] <= bounds[1]:
                continue
            bottom = min(bed[max(i-1,0)], bed[min(i,len(bed)-1)])
            seg.append([(xf[i], bottom), (xf[i], 0)])
        for k in indices(zf.size, stride):
            if not bounds[2] <= zf[k] <= bounds[3]:
                continue
            for i in range(len(bed)):
                if bed[i] < zf[k] and xf[i+1]>=bounds[0] and xf[i]<=bounds[1]:
                    seg.append([(xf[i], zf[k]),(xf[i+1],zf[k])])
        polygons=[]
        for k,i in zip(*np.where((hfac>0)&(hfac<1-1e-10))):
            if xf[i+1]>=bounds[0] and xf[i]<=bounds[1]:
                polygons.append([(xf[i],zf[k]),(xf[i+1],zf[k]),(xf[i+1],bed[i]),(xf[i],bed[i])])
        ax.add_collection(PolyCollection(polygons, facecolors=colors['partial'], edgecolors='none', alpha=.72,zorder=2))
        ax.stairs(bed,xf,baseline=bounds[2],fill=True,color=colors['ground'],zorder=0)
        ax.add_collection(LineCollection(seg,colors=colors['mit'],linewidths=.55,alpha=.80,zorder=3))
        ax.stairs(bed,xf,color='#524638',linewidth=1.25,zorder=5)
        ax.plot(fine_x,exact_bed,color='#bb5b38',lw=1.2,ls='--',zorder=6)

    for col,draw in enumerate((bspf,mit)):
        for row,bounds in enumerate(((0,1800,-105,3),zoom)):
            ax=axes[row,col]
            draw(ax,8 if row==0 else 1,bounds)
            ax.set(xlim=bounds[:2], ylim=bounds[2:], xlabel='x / m', ylabel='z / m')
            ax.set_facecolor('#f5f9fc')
            ax.tick_params(labelsize=9)
            if row==0:
                ax.add_patch(Rectangle((zoom[0],zoom[2]),zoom[1]-zoom[0],zoom[3]-zoom[2],
                    fill=False,ec='#c53945',lw=1.25,zorder=10))
                ax.text(1060,-92,'红框：下方局部放大',color='#9e3440',fontsize=9)
            ax.set_title(('全域：每 8 条网格线显示 1 条' if row==0 else '斜坡底部：显示全部网格线'), loc='left',fontsize=11,pad=12)
    axes[1,0].text(.035,.07,'层面随地形弯曲；底部无截断网格',transform=axes[1,0].transAxes,
                  fontsize=10,bbox=dict(facecolor='white',alpha=.90,edgecolor='none',pad=5))
    axes[1,1].legend(handles=[Patch(facecolor=colors['partial'],alpha=.72,label='底部部分网格（hFac < 1）'),
        Line2D([0],[0],color='#bb5b38',ls='--',label='连续地形'),
        Line2D([0],[0],color='#524638',label='MITgcm 有效阶梯底界')],loc='lower left',fontsize=9,framealpha=.95)
    fig.text(.075,.075,'注意：BSPF 画的是构造点连线，不是有限体积单元；积分点未画。MITgcm 画的是实际网格边界。',fontsize=10,color='#475569')
    fig.text(.075,.045,'两列使用相同坐标范围与纵横比例；为便于查看，全域图统一夸大垂向比例。相同“321 × 161”不代表相同自由度。',fontsize=10,color='#475569')
    fig.savefig(out/'grid_comparison.png',dpi=190,facecolor='white')
    fig.savefig(out/'grid_comparison.svg',facecolor='white')
    meta=dict(bspf_construction_points=[nx,nz],bspf_quadrature=[int(np.ceil(nx*cfg['quad'])),int(np.ceil(nz*cfg['quad']))],
        mitgcm_wet_columns=case['wet_x_columns'],mitgcm_vertical_levels=nlev,
        mitgcm_wet_cells=int(np.count_nonzero(hfac)),mitgcm_partial_cells=int(np.count_nonzero((hfac>0)&(hfac<1-1e-10))),
        zoom_bounds_m=zoom,overview_stride=8,mitgcm_geometry_source=str(run),bspf_config_source=str(bspf_run/'config.json'))
    (out/'grid_comparison.json').write_text(json.dumps(meta,indent=2))
    print(out/'grid_comparison.png')
    print(json.dumps(meta,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--mitgcm-run',type=Path,default=ROOT/'build/mitgcm-slope-321x161/run-50s-converged')
    p.add_argument('--bspf-run',type=Path,default=ROOT/'build/isw-slope-jax-50s-8psjp_57')
    p.add_argument('--out',type=Path,default=ROOT/'build/mitgcm-slope-321x161/figures')
    args=p.parse_args()
    plot(args.mitgcm_run,args.bspf_run,args.out)
