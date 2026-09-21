"""Plot existing acceptance differences without modifying experiment evidence."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def create_report(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    if out == root or root in out.parents:
        raise ValueError("Choose a separate report directory")
    out.mkdir(parents=True, exist_ok=False)
    lines = [
        "# Resolution acceptance evidence",
        "",
        "These are finite refinement comparisons. A passing screen is not an exact error bound. Roundoff candidates are marked in the source JSON; no temporal order is fitted here.",
        "",
    ]
    count = 0
    for path in sorted(root.rglob("validation.json")):
        data = json.loads(path.read_text())
        if "differences" not in data:
            continue
        count += 1
        label = str(path.parent.relative_to(root)).replace("/", "_") or data["suite"]
        lines += [f"## {label}", "", f"Accepted: **{data['passed']}**", ""]
        for hour, detail in data["differences"].items():
            m = detail["metrics"]
            fields = list(detail["passed"])
            space = m["space_65_81"]
            fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
            x = np.arange(len(fields))
            for key in ("relative_rms", "relative_linf"):
                axes[0].semilogy(
                    x, [max(space[f][key], 1e-18) for f in fields], "o-", label=key
                )
            limits = data["config"]["validation"]
            axes[0].axhline(
                limits.get("space_rms_limit", 0.01),
                color="k",
                ls="--",
                label="RMS threshold",
            )
            axes[0].axhline(
                limits.get("space_peak_limit", 0.05),
                color="gray",
                ls=":",
                label="peak threshold",
            )
            axes[0].set(
                title=f"{hour} h: finest spatial difference",
                ylabel="Relative difference",
            )
            for key in m:
                if key.startswith("time_n81") or key.startswith("quadrature_"):
                    ratio = [
                        m[key][f]["rms"] / max(space[f]["rms"], 1e-30) for f in fields
                    ]
                    axes[1].semilogy(x, np.maximum(ratio, 1e-18), "o-", label=key)
            axes[1].axhline(
                limits.get("error_separation_ratio", 0.1),
                color="k",
                ls="--",
                label="separation threshold",
            )
            axes[1].set(
                title=f"{hour} h: time/quadrature vs space", ylabel="RMS error ratio"
            )
            for a in axes:
                a.set_xticks(x, fields, rotation=55, ha="right")
                a.legend(fontsize=8)
                a.grid(alpha=0.2)
            image = f"{label}_{hour}h.png"
            fig.savefig(out / image, dpi=150)
            plt.close(fig)
            failed = [k for k, v in detail["passed"].items() if not v]
            noise = sorted(
                {
                    f
                    for key, metric in m.items()
                    if key.startswith(("time_", "quadrature_"))
                    for f, value in metric.items()
                    if value.get("roundoff_candidate")
                }
            )
            lines += [
                f"### {hour} hours",
                "",
                f"Failed quantities: {failed or 'none'}. Roundoff candidates: {noise or 'none'}.",
                "",
                f"![{hour} h]({image})",
                "",
            ]
    if count == 0:
        lines += [
            "No completed resolution/quadrature comparison found; this report does not claim acceptance.",
            "",
        ]
    (out / "report.md").write_text("\n".join(lines))
    return out / "report.md"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    print(create_report(a.root, a.out))
