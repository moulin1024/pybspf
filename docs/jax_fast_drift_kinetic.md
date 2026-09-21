# 轴级 FFT / 低秩 BSPF 漂移动理学

`plan_drift_kinetic` 现在默认 `backend="matrix_free"`，原实现通过
`backend="dense"` 显式选择，用于小规模对照。没有自动退回稠密轴算子的分支。
默认磁镜示例也使用快速轴，并继续演化 log(f)。

## 实际替换的计算

| 运算 | 原路径 | 默认快速路径 |
|---|---|---|
| 样本到求积点 | 完整 Q 矩阵乘法 | 每个 Gauss 偏移一次 FFT，加局部 degree+1 项样条评价 |
| 弱导数 | 完整轴输运矩阵 | FFT 导数加固定秩修正 |
| 固定 v、磁镜力乘法投影 | 完整轴乘法矩阵 | Fourier 系数的 Toeplitz FFT 卷积，加样条修正 |
| 质量矩阵逆 | 完整轴 Cholesky / 预计算逆作用 | hI 外加低秩空间的小核，只在初始化做小核分解 |
| 边界提升与投影 | 完整提升/投影矩阵 | FFT 转置、局部样条转置及相同的低秩质量逆 |
| log(f) 积分诊断 | Qz、Qv 稠密乘法 | 同一快速评价后取指数，积分物理 f |

运行中不做轴质量矩阵分解或三角求解。质量逆的正交补用 1/h，修正空间用
预计算的白化因子；没有形成 N×N 的质量逆、导数、投影或乘法矩阵。
仍保存 N×r、N×m 的低秩因子及 r×r 的小核，不能称为“完全不使用矩阵”。
固定样条基数 m、Gauss 阶数 q 时，r≤2m+q+1，轴应用复杂度为 O(N log N)
加 O(Nr)，存储随 N 线性增长。把 m 随 N 大幅增加会失去这一性质。

偶数网格的实 Nyquist 模态专门拆成正负频率的一半，并纳入质量修正；
因此支持 2 的幂次 FFT 网格。边界仍是开放边界，并未因使用 FFT 改为周期物理边界。

## 求积、节点与条件数

快速求积在每个均匀样本单元内放相同的 Gauss 节点。使用
`sample_aligned_knots` 或 `boundary_clustered_knots` 后，每个样条结点也都在
单元边界上，因此与原来的“样本点 + 样条结点”分割求积一致。
如果调用者传入不对齐的结点，需要自行提高求积阶数并验证；默认示例使用对齐结点。

大 N 时，固定数量但在整个区间均匀分布的结点，会让高阶端点导数的 h^-k
与跨越整个区间的样条支撑组合而恶化条件数。`boundary_clustered_knots`
将端点样条支撑保持在固定数量的网格单元内，内部光滑残差由 FFT 表示。
7 次样条使用固定 22 个基函数；初始化还会缩放 KKT 约束行。
这是可选的大轴配置，不能在小网格上不经验证就替换原试验空间。
65 点非高斯调制算例使用端点集中结点时误差曾达到 6.05e-5；因此默认物理算例
保留覆盖整个区间的对齐结点（65 点时 21 个基函数）。大轴性能表使用的是明确标注的
22 基端点集中配置，其性能结果不能替代对具体物理解的收敛验证。
该对照保存在 `build/fast_mirror/boundary_clustered_validation.json`。

构造器要求 `2*m+q+(N为偶数) < N`，保证小核严格小于整条轴。
过小网格会报错，需要减少样条核、加密网格，或显式选择稠密参考。

## 使用

```python

import bspf_models.kinetic.drift_kinetic as bspf_drift_kinetic
import pybspf.fast_axis as bspf_fast_axis
import pybspf.plans as bspf_plans
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pybspf as b

z = jnp.linspace(-4., 4., 65)
v = jnp.linspace(-3., 3., 65)
zp = bspf_plans.plan_1d(z, degree=7,
    knots=bspf_fast_axis.sample_aligned_knots(z, degree=7, n_basis=21), boundary_points=9)
vp = bspf_plans.plan_1d(v, degree=7,
    knots=bspf_fast_axis.sample_aligned_knots(v, degree=7, n_basis=21), boundary_points=9)
model = bspf_drift_kinetic.plan_drift_kinetic(
    zp, vp, magnetic_field=lambda z: 1+z*z/2,
    magnetic_gradient=lambda z: z, mu_max=2., n_mu=12,
    quadrature_order=12,
)
# integrate_log_drift_kinetic 和 log_drift_kinetic_diagnostics 的接口不变。
```

实现位于 `src/pybspf/fast_axis.py` 和 `fast_drift_kinetic.py`。
`test_fast_axis.py` 用独立直接 Fourier 求和与稠密求解验证奇偶网格、伴随关系、
质量逆、弱分部积分和非多项式乘法。测试和基准可以组装稠密参考；快速求解器本身不组装。

## 完整磁镜验证（2026-09-21）

使用区间内分布且与采样点对齐的 21 个样条基，65×65×12 网格、12 阶 Gauss
积分、RK4 步长 0.00125，模拟 t=0 到 4，共保存 81 个状态。默认 log 表示下：

| 初值 | 最大分布误差 | 相对粒子收支误差 | 相对能量收支误差 |
|---|---:|---:|---:|
| 高斯 | 2.061e-11 | 1.224e-12 | 1.372e-12 |
| 非高斯调制 | 3.286e-9 | 1.341e-12 | 1.325e-12 |

两种初值在节点和积分点上均无负值；极小尾部会下溢为零，完整 log 状态仍保留。
高斯磁镜轨道的质心位置和速度误差分别为 3.754e-13、2.679e-13。
同一结点和积分设置下，与显式稠密参考的最大分布差异为 1.080e-10，
矩差异为 2.108e-12，边界累计通量差异为 7.065e-18。

相关测试共 30 项通过：快速轴、默认工厂禁止稠密组装、磁镜模型、平行输运和
Vlasov 回归测试。物理验证脚本的全部断言通过；结果保存在
`build/fast_mirror/validation.json`、`solution.npz`、`modulated_solution.npz`
和 `validation.png`。

## 性能基准的范围

`examples/pde/benchmark_fast_axes.py` 测量“弱输运后做乘法投影”的完整轴应用，
包含质量逆，固定 32 条批处理数据。两条路径预热 JIT 并等待设备完成后计时。
稠密参考只在基准程序中用分块探测生成，避免参考构造本身需要巨大 FFT 临时空间。
内存比较采用完整快速轴计划，对照仅计稠密运行需要的两张 N×N 矩阵。

小网格上稠密 BLAS 的常数开销低，不能把渐近复杂度改善等同于所有网格都提速。
本机测试应设置 `OMP_NUM_THREADS=1`，或使用仓库中的修正 OpenBLAS 启动器；
系统原有 OpenBLAS 并发问题的说明见 `corrected_blas_build.md`。

### 本机轴基准（2026-09-21）

degree=7，固定 22 个边界集中样条基，Gauss 阶数 12，偶数网格，修正核维数固定 57。
时间为预热后多次同步调用的中位数；单位 ms。速度比为“稠密耗时 / 快速耗时”。

| N | 快速轴 / ms | 稠密轴 / ms | 速度比 | 快速计划 / MB | 稠密两算子 / MB |
|---:|---:|---:|---:|---:|---:|
| 64 | 0.305 | 0.0338 | 0.111 | 0.321 | 0.0655 |
| 256 | 1.053 | 0.285 | 0.271 | 1.191 | 1.049 |
| 1024 | 3.583 | 0.963 | 0.269 | 4.668 | 16.777 |
| 2048 | 7.274 | 2.954 | 0.406 | 9.305 | 67.109 |
| 4096 | 19.374 | 11.592 | 0.598 | 18.578 | 268.435 |
| 8192 | 42.576 | 49.329 | 1.159 | 37.125 | 1073.742 |

8192 点轴上约快 16%，计划存储约少 29 倍；当前 CPU 实现的小轴性能仍落后于
稠密 BLAS。这里没有声称 65×65 的完整磁镜模拟会加速，也没有测量 GPU 性能。
原始结果见 `build/fast_mirror/axis_benchmark.json` 和 `axis_benchmark_large.json`。
基准的同一算子两种应用方式相对差异低于 7e-15；数学正确性另由独立直接求和测试验证。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python examples/pde/benchmark_fast_axes.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python examples/pde/benchmark_fast_axes.py --sizes 4096 8192 \
  --out build/fast_mirror/axis_benchmark_large.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python examples/pde/validate_fast_mirror.py
```
