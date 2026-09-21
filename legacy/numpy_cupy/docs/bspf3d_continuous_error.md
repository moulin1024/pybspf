# BSPF 三维连续 Poisson 误差：独立 Dirichlet 闭合测试

前面的 masked pressure 测试使用 `b=(S+L)p_exact`，只测离散系统恢复误差。本次新增 **连续 PDE 制造解**，求解与 FD/PyAMG 相同的 `[0,1]³` Dirichlet 问题：

```
p_exact = exp(x+0.5y-0.3z) + sin(3πx)cos(2πy)cos(πz)
Δp = 1.34 exp(x+0.5y-0.3z) - 14π² sin(3πx)cos(2πy)cos(πz)
p|boundary = p_exact|boundary
```

**RHS 由解析公式产生，没有用任何离散算子作用于真解来生成。** 求解函数只接收解析 Laplacian 和六面边界值；内部真解只在求解完成后计算误差时使用。FD 求解 `-Δp=f`，与这里符号相反但完全等价。

## 算法和范围

这是新增的 **BSPF Dirichlet 闭合**，不是原来的 masked NS pressure 核心；未修改其实现或既有性能数据，也不能把此前 masked 核心的计时直接当作此闭合的计时。

- 复用相同的 BSPF 一阶微分 `D`，取二阶离散算子 `D2=D@D`，不施加 Q 掩码。
- 将六面 Dirichlet 数据消元，内部矩阵为三个 `D2_II` 的 Kronecker 和。
- 对 `D2_II` 做一维特征分解，沿三个轴变换后除以特征值之和，再变换回来。
- 稠密版用完整 V/V⁻¹；压缩版复用 DCT＋按层分块压缩。Dirichlet 矩阵没有原 masked 算子的零空间 lift。
- 全部零迭代、零精修，每批 2048 条网格线。测试端点包含的均匀网格，q=9、degree=13、n_basis=32、Chebyshev 12 模式/16 点端点窗口、正则化 1e-12。
- 压缩容差 1e-12、leaf_size=16、protected_modes=8。采用 JAX 双精度；本轮只记录准确度，没有进行新的性能基准。

## 连续真解误差

| 网格 | BSPF dense 最大误差 | BSPF compressed 最大误差 | FD+AMG-CG 最大误差 |
|---|---:|---:|---:|
| 40³ | 1.605719e-08 | 1.605719e-08 | — |
| 48³ | 1.775407e-09 | 1.775402e-09 | — |
| 64³ | 4.915357e-11 | 4.915268e-11 | 1.041034e-03 |
| 96³ | 1.812772e-12 | 1.815437e-12 | — |
| 128³ | 4.536371e-12 | 4.538592e-12 | 2.570000e-04 |
| 192³ | 9.882761e-12 | 9.871215e-12 | — |
| 256³ | 4.571898e-12 | 4.572343e-12 | 6.378310e-05 |

40→48 和 48→64 的实测有效阶（以 h=1/(N-1) 计）分别为 11.80、12.24。随后误差在约 1e-12–1e-11 范围内波动，不再单调下降。可以确认本例具有很高的精度、初始快速收敛及后续数值误差平台；**不能据这几个采样点声称已验证严格的指数收敛**。本实验没有单独拆分端点拟合、浮点舍入和条件数对平台的贡献。

稠密与压缩结果几乎重合，说明本例中当前压缩容差没有显著影响连续解误差。这是指定光滑解析解和 Dirichlet 边界下的结论，不自动推广到原 NS masked pressure 投影的物理壁面精度。

![Continuous error](../build/bspf3d_continuous/continuous_error.png)

## 验证与复现

7 个规模 × 2 个后端的边界误差均为 0，重算原始 `sum(D2_axis p)-analytic_laplacian` 的内部相对 L2 残差均小于 1e-10。独立七点差分模板、随机非零六面边界的小网格消元测试通过，最大恢复误差 1.44e-15。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=jax/src \
  python3 scratch/benchmark_bspf3d_continuous.py
MPLCONFIGDIR=/tmp/pybspf-mpl python3 scratch/plot_bspf3d_continuous.py
```

原始数据：`build/bspf3d_continuous/results.json`；矢量图：`build/bspf3d_continuous/continuous_error.pdf`。数值设置与全部误差指标保存在脚本/JSON 中。
