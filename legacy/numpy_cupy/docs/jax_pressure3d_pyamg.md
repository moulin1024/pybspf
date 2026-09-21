# 三维 BSPF 直接核心与二阶有限差分＋PyAMG 对照

将已有的 64³、128³、256³ BSPF 稠密／压缩直接法，与实际构造 CSR 稀疏矩阵的七点有限差分＋PyAMG 对照。BSPF 始终为零次精修；PyAMG 是此次新增的迭代基线，不替换 BSPF 实现。

## 比较的含义

- BSPF 仍采用先前的 `div(Q grad)` 张量直接核心及零空间 lift，包含面消元与恢复，尚不包含物理壁面梯度补全。
- 有限差分采用 `[0,1]³` 上的 `-Δp=f`，六个面给定精确 Dirichlet 数据，将边界消去后构造 SPD 的七点矩阵。
- `N` 都表示包含两个端点的每轴节点数。BSPF 返回 N³ 个值；FD 的矩阵有 `(N-2)³` 个内部未知量。
- 因而这是**两种离散求解流程的实现成本对照**，不是同一个矩阵更换求解器的对照，也不是相同连续 PDE 精度的成本对照。不能用本表给完整三维压力投影下结论。

采用两个分开的精度实验：

1. **离散制造解**：使用相同光滑采样场，分别令 `b_BSPF=(S+L)p` 与 `b_FD=A p_interior`，测各自离散系统的压力恢复误差。随机制造解也使用同一 seed 71、全网格样本后取内部节点。
2. **FD 连续制造解**：给定解析 `f=-Δp` 和解析边界数据，测 FD 的离散误差与迭代误差之和。本实验的误差不能与 BSPF 的离散制造解误差当作同一指标比较。

光滑真解仍为

```
p = exp(x + 0.5y - 0.3z) + sin(3πx) cos(2πy) cos(πz)
f = -1.34 exp(x + 0.5y - 0.3z) + 14π² sin(3πx) cos(2πy) cos(πz)
```

## PyAMG 配置与计时

本机 PyAMG 5.3.0；使用已安装版本的默认经典 AMG：

```python
A = pyamg.gallery.poisson((N-2, N-2, N-2), format="csr", dtype=np.float64)
A.data *= (N-1)**2
ml = pyamg.ruge_stuben_solver(A)
u = ml.solve(b, x0=None, tol=(1e-9 + 1e-10*np.linalg.norm(b))/np.linalg.norm(b),
             maxiter=100, cycle="V", accel=None, residuals=history)
```

另测 `accel="cg"`，复用同样的 AMG 配置作为 CG 的 V-cycle 预条件器；其层次在独立进程中重新构建以测量内存，实际使用时不必为切换加速器重新建层次。

默认参数为 classical strength θ=0.25、RS coarsening（second_pass=False）、classical interpolation、前后各 symmetric Gauss–Seidel，max_coarse=10。V-cycle 组未使用 Krylov 加速，CG 组采用 PyAMG 提供的 CG；两组均未做参数调优或使用上次解作为初值。

每个规模在独立进程中运行；CPU 双精度，`OPENBLAS_NUM_THREADS=1`、`OMP_NUM_THREADS=1`。PyAMG 光滑离散 RHS 先求解一次预热，再独立从零初值求解三次取中位数；BSPF 沿用上一轮七次预热后计时结果，不重跑或选择更有利的样本。线程环境变量一致，但不代表 JAX/XLA 的内部并行度与 PyAMG 相同。

两个流程都采用残差通过阈值 `||r||₂ ≤ 1e-9 + 1e-10 ||b||₂`。BSPF 是固定一次直接应用，实际残差往往远低于阈值；相同停止阈值不保证相同压力误差。BSPF 的独立残差诊断不计入核心耗时，PyAMG 自身的停止检查属于迭代求解耗时。

脚本：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 scratch/benchmark_pressure3d_pyamg.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 scratch/benchmark_pressure3d_pyamg.py --accel cg
MPLCONFIGDIR=/tmp/pybspf-mpl python3 scratch/report_pressure3d_pyamg.py
```

原始结果为 `build/pressure3d_pyamg/results.json` 与 `build/pressure3d_pyamg_cg/results.json`，保留每次求解的逐周期相对残差、周期数、时间、矩阵/层次结构及独立验证的真实残差。

## 结果

### 光滑离散 RHS：重复求解时间

| 网格 | BSPF dense | BSPF compressed | AMG V | V 周期 | AMG+CG | CG 步数 | AMG+CG / BSPF dense |
|---|---:|---:|---:|---:|---:|---:|---:|
| 64³ | 0.0111 s | 0.0446 s | 0.2902 s | 11 | 0.2423 s | 7 | 21.82× |
| 128³ | 0.1186 s | 0.3673 s | 3.0767 s | 14 | 2.5007 s | 9 | 21.09× |
| 256³ | 1.6682 s | 5.0347 s | 80.6379 s | 43 | 31.6247 s | 14 | 18.96× |

### 准备开销、稀疏矩阵与内存

| 网格 | 内部未知量 | nnz(A) | CSR A | 层次 A/P/R 合计¹ | FD 装配 | AMG 建层次² | BSPF dense 冷 setup |
|---|---:|---:|---:|---:|---:|---:|---:|
| 64³ | 238,328 | 1,645,232 | 19.74 MiB | 85.61 MiB | 0.007 s | 0.539 s | 11.875 s |
| 128³ | 2,000,376 | 13,907,376 | 166.79 MiB | 736.39 MiB | 0.046 s | 4.720 s | 12.125 s |
| 256³ | 16,387,064 | 114,322,352 | 1370.83 MiB | 6095.73 MiB | 0.391 s | 52.804 s | 12.446 s |

¹ 对所有层的 CSR A/P/R 数据和索引按地址去重计数；包含 finest A，不包含全部临时工作区，不能和进程 RSS 混为一谈。
² 此表列 V 进程的建层次时间，CG 独立进程另有一次相同配置的构造；原始 JSON 保留两次值。BSPF setup 包含 JAX 冷编译，首次核心调用的编译另计，见上一份报告。

| 网格 | BSPF dense 峰值 RSS | BSPF compressed 峰值 RSS | AMG V 峰值 RSS | AMG+CG 峰值 RSS | AMG operator complexity | 层数 |
|---|---:|---:|---:|---:|---:|---:|
| 64³ | 1.159 GiB | 1.162 GiB | 0.355 GiB | 0.379 GiB | 2.824 | 8 |
| 128³ | 2.090 GiB | 2.371 GiB | 2.394 GiB | 2.460 GiB | 2.892 | 9 |
| 256³ | 3.382 GiB | 4.280 GiB | 7.600 GiB | 7.986 GiB | 2.920 | 10 |

RSS 均是各独立工作进程的累计高水位，包含运行时、构造临时量、真解/RHS 和验证。两种实现的内存分配器及诊断工作流不同，不能把它解释为单次核心调用的独占内存。

### 离散制造解：压力恢复误差

| 网格 | 方法 | 光滑解最大误差 | 随机解最大误差 | 光滑相对残差 | 随机相对残差 |
|---|---|---:|---:|---:|---:|
| 64³ | BSPF dense | 3.125e-10 | 6.058e-10 | 6.531e-13 | 1.619e-15 |
| 64³ | BSPF compressed | 3.100e-10 | 6.073e-10 | 4.468e-13 | 1.594e-15 |
| 64³ | AMG V | 3.730e-09 | 9.256e-09 | 1.677e-11 | 2.074e-11 |
| 64³ | AMG+CG | 1.836e-10 | 1.711e-09 | 2.145e-11 | 2.014e-11 |
| 128³ | BSPF dense | 1.185e-09 | 4.886e-09 | 1.241e-11 | 3.544e-15 |
| 128³ | BSPF compressed | 1.096e-09 | 5.035e-09 | 4.045e-12 | 2.521e-13 |
| 128³ | AMG V | 5.702e-08 | 3.761e-07 | 8.683e-11 | 4.806e-11 |
| 128³ | AMG+CG | 1.402e-10 | 2.999e-08 | 6.984e-12 | 9.399e-11 |
| 256³ | BSPF dense | 1.034e-08 | 2.813e-08 | 4.870e-12 | 6.890e-15 |
| 256³ | BSPF compressed | 1.034e-08 | 2.810e-08 | 4.758e-12 | 1.154e-13 |
| 256³ | AMG V | 1.758e-07 | 2.455e-06 | 9.321e-11 | 6.961e-11 |
| 256³ | AMG+CG | 2.494e-09 | 1.252e-08 | 7.347e-11 | 2.358e-11 |

随机 RHS 的收敛周期和计时另见 JSON；主计时表严格使用同一种光滑制造解，不混用更容易收敛的随机数据。

### 连续制造解：FD 二阶离散误差

| 网格 | AMG V 最大压力误差 | AMG+CG 最大压力误差 | V/CG 真残差均通过？ |
|---|---:|---:|---|
| 64³ | 1.041032e-03 | 1.041034e-03 | True |
| 128³ | 2.569901e-04 | 2.570000e-04 | True |
| 256³ | 6.373236e-05 | 6.378310e-05 | True |

FD/PyAMG 的 18 项独立真实残差检查：全部通过。停止迭代后的 `A@u-b` 重新计算，未只相信求解器内部残差。

连续误差随网格加密约按 h² 下降，两种迭代策略得到相同的离散精度。减小 AMG 容差无法消除二阶离散误差。BSPF 本轮尚无相同 Dirichlet 连续问题的对照，因此这里不声称测得了两者的连续 PDE 精度比。

## 解读

本机这一组结果中，复用 setup 后的 BSPF 稠密张量直接核心比两个 PyAMG 基线快。256³ 的默认 V-cycle 收敛明显恶化，CG 加速对照可区分“默认迭代配置的瓶颈”和“稀疏层次应用的成本”。这不是所有 AMG 配置或硬件的性能上限。

对重复压力求解，setup 可以摊销；单次冷启动应把装配、建层次及 JIT 都计入，不能只比较 warm solve。尤其小规模下 BSPF 的 JAX 冷 setup 明显更昂贵。

本问题是规则长方体和常系数，张量可分离性本来就有利于 BSPF 的直接核心。相同 FD Dirichlet 矩阵也可用 DST 做直接求解；因此本表不能被概括成“BSPF 必然比二阶有限差分快”。PyAMG 的适用范围比这种可分离算子更广。

![Comparison](../build/pressure3d_pyamg/comparison.png)

## 检查

- 10³ smoke test：光滑／随机离散制造解和连续制造解全部通过真实残差检查。
- 采用独立的三维 DST 对角化求解同一 FD Dirichlet 系统，16³、32³、64³ 的连续制造解最大误差分别为 0.0169974、0.00427721、0.00104103，确认边界 RHS 的实现及二阶误差趋势。
- 新增 benchmark 与绘图脚本通过 Ruff；未修改 BSPF 算法或为它加入迭代。

## 误差收敛专图

![Error convergence](../build/pressure3d_pyamg/error_convergence.png)

生成脚本：`scratch/plot_pressure3d_errors.py`；矢量版：`build/pressure3d_pyamg/error_convergence.pdf`。FD 的连续误差实测收敛阶（AMG+CG）为 1.9954、1.9992。BSPF 稠密/压缩的离散恢复误差几乎重合，但该制造 RHS 实验不能证明连续 PDE 的指数收敛。迭代残差曲线也不等于逐步压力误差；当前数据未记录每个迭代解的压力误差。

后续新增了[相同连续 PDE 和 Dirichlet 边界下的 BSPF 误差测试](bspf3d_continuous_error.md)。该测试采用独立的 BSPF Dirichlet 闭合；本文原有 masked 核心的计时与误差数据保持其原始含义。
