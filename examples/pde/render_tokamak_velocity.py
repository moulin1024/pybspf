"""Plot actual velocity-induction trajectories and independent reference errors."""

from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    paths = [Path("build/tokamak_velocity"), Path("build/tokamak_velocity_far")]
    fig, axs = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    colors = ["#1565c0", "#d65f00"]
    labels = ["Close shaped wall", "Distant rectangular wall"]
    for i, path in enumerate(paths):
        data = np.load(path / "evolution.npz")
        t = data["time"]
        q = data["displacement_coefficients"]
        ref = data["reference_displacement"]
        # The documented initial condition is proportional to the centroid functional.
        h = q[0] * data["centroid_z"][0] / (q[0] @ q[0])
        ax = axs[0, i]
        ax.plot(
            t, data["centroid_z"], color=colors[i], label="Velocity + induction RK4"
        )
        ax.plot(
            t,
            ref @ h,
            "--",
            color="#555555",
            label="Independent displacement reference",
        )
        ax.set(xlabel="Time", ylabel="Mean vertical displacement", title=labels[i])
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        error = np.linalg.norm(q - ref, axis=1) / np.maximum(
            np.linalg.norm(ref, axis=1), 1e-30
        )
        axs[1, 0].semilogy(
            t, np.maximum(error, 1e-18), color=colors[i], label=labels[i]
        )
        energy = data["energy"]
        drift = abs(energy - energy[0]) / abs(energy[0])
        axs[1, 1].semilogy(
            t, np.maximum(drift, 1e-18), color=colors[i], label=labels[i]
        )
    axs[1, 0].set(
        xlabel="Time",
        ylabel="Relative displacement error",
        title="Comparison with the old discrete equations",
    )
    axs[1, 1].set(
        xlabel="Time",
        ylabel="Relative energy drift",
        title="Explicit RK4 temporal energy error",
    )
    for ax in axs[1]:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Shared BSPF NS kernels: velocity–induction MHD\nLinear n=0, incompressible; pressure eliminated by the stream basis"
    )
    fig.savefig(paths[0] / "velocity_framework_checks.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
