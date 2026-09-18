"""Plot saved independent MMS samples without rerunning the Poisson solve."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import PathPatch
from matplotlib.path import Path as PlotPath

from bspf_jax.embedded_poisson import benchmark_domains
from bspf_jax.random_wave_mms import RandomWaveMMS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("build/convex_poisson_n65"))
    args = parser.parse_args()
    rows = json.loads((args.input / "results.json").read_text())
    domain = benchmark_domains()[0]
    curve = domain.curve(np.linspace(0, domain.period, 2001))
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 8.4), layout="constrained")
    metrics = []
    for i, band in enumerate((4, 12)):
        row = next(r for r in rows if r["name"] == "h32_h2" and r["band"] == band)
        data = np.load(args.input / f"h32_h2_{band}pi.npz")
        points, exact, value = data["points"], data["exact"], data["value"]
        # Check that the saved reference matches the documented manufactured field.
        reference = RandomWaveMMS.create(kmax=band * np.pi).evaluate(points)[0]
        np.testing.assert_allclose(exact, reference, rtol=1e-14, atol=1e-14)
        error = abs(value - exact)
        relative = float(np.linalg.norm(error) / np.linalg.norm(exact))
        np.testing.assert_allclose(relative, row["value"], rtol=1e-10)
        metrics.append(
            dict(
                band=band,
                nodes=row["nodes"],
                relative_sample_l2=relative,
                max_sample_error=float(error.max()),
                samples=len(points),
            )
        )
        x, xi = np.unique(points[:, 0], return_inverse=True)
        y, yi = np.unique(points[:, 1], return_inverse=True)
        limit = max(abs(exact).max(), abs(value).max())
        field_norm = Normalize(-limit, limit)
        error_top = 10.0 ** np.ceil(np.log10(error.max()))
        error_bottom = error_top * 1e-4
        for j, (field, title) in enumerate(
            (
                (exact, "MMS exact solution"),
                (value, "BSPF numerical solution"),
                (error, "Absolute error (log scale)"),
            )
        ):
            ax = axes[i, j]
            grid = np.ma.masked_all((len(y), len(x)))
            grid[yi, xi] = np.maximum(field, error_bottom) if j == 2 else field
            im = ax.pcolormesh(
                x,
                y,
                grid,
                shading="nearest",
                rasterized=True,
                cmap="magma" if j == 2 else "RdBu_r",
                norm=LogNorm(error_bottom, error_top) if j == 2 else field_norm,
            )
            im.set_clip_path(PathPatch(PlotPath(curve), transform=ax.transData))
            ax.plot(*curve.T, color="#242c38", lw=0.9)
            ax.set(
                aspect="equal",
                xlim=(-0.92, 0.92),
                ylim=(-0.79, 0.79),
                xlabel="$x$",
                ylabel="$y$",
                title=title,
            )
            ax.set_facecolor("#f4f5f7")
            fig.colorbar(
                im,
                ax=ax,
                shrink=0.82,
                pad=0.025,
                label="$|u_N-u_*|$" if j == 2 else "$u$",
            )
        axes[i, 0].set_title(rf"MMS exact solution | $k_{{\max}}={band}\pi$")
        axes[i, 2].set_title(
            "Absolute error (log scale)\n"
            rf"relative $\ell^2={relative:.2e}$; max $={error.max():.2e}$",
            fontsize=10,
        )
    nodes = {m["nodes"] for m in metrics}
    if len(nodes) != 1:
        raise ValueError("Both panels must use the same background resolution")
    n = nodes.pop()
    fig.suptitle(
        rf"Random-wave MMS on a convex B-spline domain | $N={n}$, "
        rf"$h={2.4 / (n - 1):g}$"
        "\n"
        "64 waves; independent evaluation samples; exact/numerical colors matched per row",
        fontsize=14,
    )
    fig.supxlabel(
        "Error color scales differ by row; values below each scale minimum are clipped. "
        "Grey denotes unsampled/outside region.",
        fontsize=9,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(args.input / f"mms_solution_error.{suffix}", dpi=200)
    (args.input / "mms_plot_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
