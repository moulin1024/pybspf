"""Insert reproducible benchmark tables into the comparison report."""

import json
from pathlib import Path

v = json.loads(Path("build/pressure3d_pyamg/results.json").read_text())
cg = json.loads(Path("build/pressure3d_pyamg_cg/results.json").read_text())
bspf = json.loads(Path("build/pressure3d_benchmark/results.json").read_text())
parts = [
    "## 结果\n",
    "### 光滑离散 RHS：重复求解时间\n",
    "| 网格 | BSPF dense | BSPF compressed | AMG V | V 周期 | AMG+CG | CG 步数 | AMG+CG / BSPF dense |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for a, c in zip(v, cg):
    n = a["n"]
    d, k = [r for r in bspf if r["n"] == n]
    dt, kt = d["timing"]["median_seconds"], k["timing"]["median_seconds"]
    at, ct = [r["cases"]["smooth_discrete"]["median_seconds"] for r in (a, c)]
    it, jt = [r["cases"]["smooth_discrete"]["cycles"][-1] for r in (a, c)]
    parts.append(
        f"| {n}³ | {dt:.4f} s | {kt:.4f} s | {at:.4f} s | {it} | {ct:.4f} s | {jt} | {ct / dt:.2f}× |"
    )
parts += [
    "",
    "### 准备开销、稀疏矩阵与内存\n",
    "| 网格 | 内部未知量 | nnz(A) | CSR A | 层次 A/P/R 合计¹ | FD 装配 | AMG 建层次² | BSPF dense 冷 setup |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for a in v:
    d = next(r for r in bspf if r["n"] == a["n"] and r["backend"] == "dense")
    parts.append(
        f"| {a['n']}³ | {a['unknowns']:,} | {a['nnz']:,} | {a['matrix_bytes'] / 2**20:.2f} MiB | {a['hierarchy_csr_bytes'] / 2**20:.2f} MiB | {a['matrix_seconds']:.3f} s | {a['hierarchy_seconds']:.3f} s | {d['dense_setup_seconds']:.3f} s |"
    )
parts += [
    "",
    "¹ 对所有层的 CSR A/P/R 数据和索引按地址去重计数；包含 finest A，不包含全部临时工作区，不能和进程 RSS 混为一谈。",
    "² 此表列 V 进程的建层次时间，CG 独立进程另有一次相同配置的构造；原始 JSON 保留两次值。BSPF setup 包含 JAX 冷编译，首次核心调用的编译另计，见上一份报告。",
    "",
    "| 网格 | BSPF dense 峰值 RSS | BSPF compressed 峰值 RSS | AMG V 峰值 RSS | AMG+CG 峰值 RSS | AMG operator complexity | 层数 |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for a, c in zip(v, cg):
    d, k = [r for r in bspf if r["n"] == a["n"]]
    parts.append(
        f"| {a['n']}³ | {d['peak_rss_bytes'] / 2**30:.3f} GiB | {k['peak_rss_bytes'] / 2**30:.3f} GiB | {a['peak_rss_bytes'] / 2**30:.3f} GiB | {c['peak_rss_bytes'] / 2**30:.3f} GiB | {a['operator_complexity']:.3f} | {len(a['levels'])} |"
    )
parts += [
    "",
    "RSS 均是各独立工作进程的累计高水位，包含运行时、构造临时量、真解/RHS 和验证。两种实现的内存分配器及诊断工作流不同，不能把它解释为单次核心调用的独占内存。",
    "",
    "### 离散制造解：压力恢复误差\n",
    "| 网格 | 方法 | 光滑解最大误差 | 随机解最大误差 | 光滑相对残差 | 随机相对残差 |",
    "|---|---|---:|---:|---:|---:|",
]
for a, c in zip(v, cg):
    for r in [r for r in bspf if r["n"] == a["n"]]:
        s, t = r["accuracy"]["smooth"], r["accuracy"]["random"]
        parts.append(
            f"| {r['n']}³ | BSPF {r['backend']} | {s['error_linf']:.3e} | {t['error_linf']:.3e} | {s['residual_relative_l2']:.3e} | {t['residual_relative_l2']:.3e} |"
        )
    for label, r in [("AMG V", a), ("AMG+CG", c)]:
        s, t = r["cases"]["smooth_discrete"], r["cases"]["random_discrete"]
        parts.append(
            f"| {r['n']}³ | {label} | {s['error_linf']:.3e} | {t['error_linf']:.3e} | {s['residual_relative_l2']:.3e} | {t['residual_relative_l2']:.3e} |"
        )
parts += [
    "",
    "随机 RHS 的收敛周期和计时另见 JSON；主计时表严格使用同一种光滑制造解，不混用更容易收敛的随机数据。",
    "",
    "### 连续制造解：FD 二阶离散误差\n",
    "| 网格 | AMG V 最大压力误差 | AMG+CG 最大压力误差 | V/CG 真残差均通过？ |",
    "|---|---:|---:|---|",
]
for a, c in zip(v, cg):
    s, t = [r["cases"]["smooth_continuous"] for r in (a, c)]
    parts.append(
        f"| {a['n']}³ | {s['error_linf']:.6e} | {t['error_linf']:.6e} | {s['converged'] and t['converged']} |"
    )
failed = [
    (r["n"], key)
    for r in v + cg
    for key, case in r["cases"].items()
    if not case["converged"]
]
parts += [
    "",
    f"FD/PyAMG 的 18 项独立真实残差检查：{'全部通过' if not failed else str(failed)}。停止迭代后的 `A@u-b` 重新计算，未只相信求解器内部残差。",
    "",
    "连续误差随网格加密约按 h² 下降，两种迭代策略得到相同的离散精度。减小 AMG 容差无法消除二阶离散误差。BSPF 本轮尚无相同 Dirichlet 连续问题的对照，因此这里不声称测得了两者的连续 PDE 精度比。",
    "",
    "## 解读\n",
    "本机这一组结果中，复用 setup 后的 BSPF 稠密张量直接核心比两个 PyAMG 基线快。256³ 的默认 V-cycle 收敛明显恶化，CG 加速对照可区分“默认迭代配置的瓶颈”和“稀疏层次应用的成本”。这不是所有 AMG 配置或硬件的性能上限。",
    "",
    "对重复压力求解，setup 可以摊销；单次冷启动应把装配、建层次及 JIT 都计入，不能只比较 warm solve。尤其小规模下 BSPF 的 JAX 冷 setup 明显更昂贵。",
    "",
    "本问题是规则长方体和常系数，张量可分离性本来就有利于 BSPF 的直接核心。相同 FD Dirichlet 矩阵也可用 DST 做直接求解；因此本表不能被概括成“BSPF 必然比二阶有限差分快”。PyAMG 的适用范围比这种可分离算子更广。",
    "",
    "![Comparison](../build/pressure3d_pyamg/comparison.png)",
    "",
]
p = Path("docs/jax_pressure3d_pyamg.md")
text = p.read_text()
start = (
    text.index("<!-- RESULTS -->")
    if "<!-- RESULTS -->" in text
    else text.index("## 结果")
)
end = text.index("## 检查", start)
p.write_text(text[:start] + "\n".join(parts) + "\n" + text[end:])
