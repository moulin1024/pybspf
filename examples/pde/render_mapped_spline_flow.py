"""Render a recorded mapped Re=200 setup as a velocity/vorticity MP4.

Re-evolves the saved channel configuration on GPU to capture intermediate
states. Only visualization fields and coefficient snapshots are downloaded.
"""
import argparse
import json
from pathlib import Path
import shutil
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from bspf_jax.mapped_navier_stokes import (
    MappedNavierStokesPlan, channel_streamfunction, _stream_fields,
)
from bspf_jax.immersed_poisson import EllipticHole


@jax.jit
def frame_fields(velocity_op, vorticity_op, lift_velocity, lift_vorticity, state):
    velocity = velocity_op @ state+lift_velocity
    vorticity = vorticity_op @ state+lift_vorticity
    return jnp.linalg.norm(velocity, axis=-1), vorticity


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', type=Path, default=Path('build/mapped_spline_flow_re200_strong'))
    ap.add_argument('--out', type=Path)
    ap.add_argument('--fps', type=int, default=20)
    ap.add_argument('--frame-dt', type=float, default=.05)
    ap.add_argument('--vorticity-limit', type=float, default=30.)
    args = ap.parse_args()
    if args.fps < 1 or args.frame_dt <= 0 or args.vorticity_limit <= 0:
        ap.error('fps, frame-dt and vorticity-limit must be positive')
    report = json.loads((args.run_dir/'report.json').read_text())
    if report['case'] != 'channel':
        ap.error('run-dir must contain a channel report')
    case = report['cases'][-1]
    target = args.out or args.run_dir/'flow.mp4'
    target.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    matplotlib.rcParams['animation.ffmpeg_path'] = ffmpeg
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from matplotlib.patches import Ellipse

    jax.config.update('jax_enable_x64', True)
    device = jax.devices('gpu')[0]
    bounds, hole = (-1., 3., -1., 1.), EllipticHole()
    p = MappedNavierStokesPlan(elements=(case['elements'],)*2, degree=case['degree'],
        viscosity=case['viscosity'], dt=case['dt'], boundary=case.get('boundary', 'nitsche'),
        lift=channel_streamfunction(bounds, hole), device=device)
    print('GPU_SETUP', p.setup_seconds, 'dofs', p.dofs, flush=True)
    nr, nt = 65, 97
    r, t = jax.device_put((np.linspace(0, 1, nr), np.linspace(0, 1, nt)), device)
    op = p._operators(r, t)
    lift_v, lift_g = _stream_fields(p.lift, op['points'])
    velocity_op = op['velocity']*p.scale
    vorticity_op = (op['gradient'][..., 1, 0, :]-op['gradient'][..., 0, 1, :])*p.scale
    lift_w = lift_g[..., 1, 0]-lift_g[..., 0, 1]
    points = np.asarray(op['points']).reshape(4, nr, nt, 2)
    del op, lift_g
    every = max(1, round(args.frame_dt/p.dt))
    steps = case['steps']
    frame_steps = np.unique(np.r_[np.arange(0, steps+1, every), steps]).astype(int)
    times = frame_steps*p.dt
    fields = np.empty((len(times), 2, 4, nr, nt))
    states = np.empty((len(times), p.dofs))
    state, done = p.stokes_state, 0
    start = perf_counter()
    for i, step in enumerate(frame_steps):
        if step > done:
            state = p.advance(state, time=done*p.dt, steps=int(step-done))
        speed, vort = jax.device_get(frame_fields(velocity_op, vorticity_op, lift_v, lift_w, state))
        fields[i, 0], fields[i, 1] = speed.reshape(4, nr, nt), vort.reshape(4, nr, nt)
        states[i] = np.asarray(state)
        if not np.all(np.isfinite(fields[i])):
            raise RuntimeError(f'Non-finite fields at t={times[i]}')
        done = step
        if i % 50 == 0 or i == len(times)-1:
            print('CAPTURE', i+1, '/', len(times), 't', times[i], flush=True)
    capture_seconds = perf_counter()-start
    saved = np.load(args.run_dir/'solution.npz')['coefficients']
    state_difference = float(np.max(np.abs(states[-1]-saved)))
    np.savez_compressed(target.with_suffix('.states.npz'), times=times, coefficients=states)
    speed_limit = float(np.ceil(fields[:, 0].max()*10)/10)
    print('CAPTURE_DONE', capture_seconds, 'final_state_max_difference', state_difference, flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.6), layout='constrained')
    title = fig.suptitle('', fontsize=16)
    meshes = []
    for panel, (ax, name, cmap, lo, hi) in enumerate(zip(axes,
        ('Velocity magnitude', 'Vorticity'), ('viridis', 'RdBu_r'),
        (0., -args.vorticity_limit), (speed_limit, args.vorticity_limit))):
        patch_meshes = []
        for k in range(4):
            xy = points[k]
            mesh = ax.pcolormesh(xy[..., 0], xy[..., 1], fields[0, panel, k],
                                shading='gouraud', cmap=cmap, vmin=lo, vmax=hi, rasterized=True)
            patch_meshes.append(mesh)
        ax.add_patch(Ellipse(hole.center, 2*hole.axes[0], 2*hole.axes[1],
                            facecolor='white', edgecolor='#333333', linewidth=1.))
        ax.set(title=name, xlabel='x', ylabel='y', aspect='equal',
               xlim=bounds[:2], ylim=bounds[2:])
        fig.colorbar(mesh, ax=ax, shrink=.75, extend='both' if panel else 'neither')
        meshes.append(patch_meshes)
    fig.supxlabel('Exact no-slip boundaries · GPU simulation · fixed colour scales'
                  if p.boundary == 'strong' else 'Nitsche boundaries · GPU simulation · fixed colour scales', fontsize=11)
    writer = FFMpegWriter(fps=args.fps, codec='libx264',
        metadata={'title': 'Mapped spline Navier-Stokes, Re=200'},
        extra_args=['-crf', '18', '-preset', 'fast', '-pix_fmt', 'yuv420p', '-movflags', '+faststart'])
    start = perf_counter()
    with writer.saving(fig, str(target), dpi=100):
        for i, time in enumerate(times):
            title.set_text(f'Mapped spline flow around an ellipse  |  Re = {report["reynolds"]:g}  |  t = {time:05.2f}')
            for panel in range(2):
                for k, mesh in enumerate(meshes[panel]):
                    mesh.set_array(fields[i, panel, k])
            writer.grab_frame()
            if i == len(times)//2:
                fig.savefig(target.with_suffix('.preview.png'), dpi=100)
            if i % 50 == 0 or i == len(times)-1:
                print('ENCODE', i+1, '/', len(times), flush=True)
    plt.close(fig)
    info = dict(source_run=str(args.run_dir), video=str(target), frames=len(times), fps=args.fps,
        duration_seconds=len(times)/args.fps, simulation_start=float(times[0]), simulation_end=float(times[-1]),
        resolution=[1600, 560], device=str(device), boundary=p.boundary, dt=p.dt, dofs=p.dofs,
        capture_seconds=capture_seconds, encode_seconds=perf_counter()-start,
        final_state_max_difference=state_difference, speed_colour_limits=[0., speed_limit],
        vorticity_colour_limits=[-args.vorticity_limit, args.vorticity_limit],
        note='Vorticity colours saturate outside the stated range; numerical fields are not filtered.')
    target.with_suffix('.video.json').write_text(json.dumps(info, indent=2)+'\n')
    print('DONE', json.dumps(info), flush=True)


if __name__ == '__main__':
    main()
