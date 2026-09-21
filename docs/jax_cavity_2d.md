# BSPF 二维不可压 NS 顶盖驱动方腔

入口：[`examples/pde/cavity_2d.py`](../examples/pde/cavity_2d.py)。
求解器：[`packages/models/src/bspf_models/fluids/cavity.py`](../packages/models/src/bspf_models/fluids/cavity.py)。

## 物理问题

在单位正方形上求解

\[
\partial_t\boldsymbol u+(\boldsymbol u\cdot\nabla)\boldsymbol u
=-\nabla p+Re^{-1}\Delta\boldsymbol u,\qquad
\nabla\cdot\boldsymbol u=0.
\]

默认 `Re=100`，初始速度为零，没有体力、海绵、滤波或人工黏性。
左、右、底壁无滑移，顶壁为

\[
u(x,1,t)=s(t)16x^2(1-x)^2,\qquad v(x,1,t)=0.
\]

其中 `s=10z³−15z⁴+6z⁵`，`z=clip(t/ramp_time,0,1)`，默认启动时间为 1。
长度尺度和充分启动后的顶盖峰值速度均为 1，因此 `nu=1/Re`。
这是**正则化顶盖方腔**；顶盖速度在角点平滑降为零，不能直接用恒速顶盖的
Ghia 数据评价误差。四边均为非周期物理壁面。

## BSPF 离散及时间推进

复用现有 `stream_navier_stokes.py` 中的 BSPF 流函数 Galerkin 空间：
degree-13 B-spline、q9 修正、Cheb12/window16 端点处理，float64 运行，
高精度组装只在主机初始化时使用。没有额外边界层富集。

写成 `psi=psi_h+s(t)*psi_L`，其中
`psi_L=16*x²*(1−x)²*y²*(y−1)`，齐次部分 `psi_h` 的值和法向导数在四壁为零。
速度采用 `(psi_y,-psi_x)`，因此流函数的混合导数逐点相消。
无散度检验函数消去压力；本例输出流函数、速度和涡量，不恢复压力。

模态方程为

\[
M\dot a=C(u)-\nu Da+\nu sL(\Delta u_L)-\dot sL(u_L).
\]

`L` 是速度检验函数的积分负载。最后一项是启动边界的惯性贡献，不能省略。
对流用旋转形式及过积分；黏性弱算子是流函数 Hessian 内积。

时间格式为二阶 IMEX midpoint：半步后向 Euler 黏性预测，完整步
Crank–Nicolson 黏性校正，对流和边界提升项在预测中点计算。
固定步长时只分解一次 `M+dt*nu*D/2`。对流仍有显式稳定性限制，
提高 Re、网格数或步长后需要重新验证稳定性。

这里为了优先完成可验证的方腔，采用稠密二维 Cholesky；存储约为
`O((n−4)^4)`，适合中小型演示，不是大网格可扩展求解方案。

## 运行

依赖当前 Python 环境中的 JAX、NumPy、SciPy、gmpy2、Matplotlib；也可安装
项目依赖组：`pip install -e '.[host,notebook,test]' -e './packages/models[precision,test]' -e './packages/sim[air-sea,test]'`。仓库根目录运行：

```bash
MPLCONFIGDIR=/tmp/bspf-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python examples/pde/cavity_2d.py --n 41 --re 100 --dt 0.01 --T 30
```

默认每 0.5 时间单位检查一次。启动结束后，节点上的最大速度时间导数
小于 `1e-7` 时提前停止。`--steady-tol 0` 禁用提前停止并运行到 T；此时
输出 `steady=false` 表示没有满足设置为零的阈值，应查看实际残差。
`T/dt` 非整数时向上取步数并略减 dt，以准确到达终止时间。

输出目录默认 `build/cavity_2d`：

- `cavity.png`：速度及流函数等值线、涡量、中心线速度、动能和稳态残差。
- `state.npz`：最终模态、节点、流函数、速度、涡量与物理参数。
- `summary.json`：配置及诊断时间序列。

数组空间顺序为 `(x,y)`，速度最后一维为 `(u,v)`。
动能在组装积分点计算，壁面残差和稳态残差在输出节点计算。
散度诊断来自两次独立矩阵收缩的混合导数，而非直接填零。

## 验证

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m pytest packages/models/tests/test_cavity.py packages/models/tests/test_stream_navier_stokes.py -q
```

包含连续多项式制造解：`psi=s(t)*(psi_L+0.2*f(x)*g(y))`，
`f=16x²(1−x)²`、`g=y²(1−y)²`，压力取 `p=−|u|²/2`，
外力由解析多项式导数计算，独立于离散 BSPF 微分矩阵。
它同时检查非零内部流场、黏性、对流、非齐次边界惯性及时间二阶收敛。
该外力仅在测试中出现，实际方腔没有外力。

网格比较应固定 Re、dt、T、启动时间并关闭提前停止。例如分别运行
`--n 33/41/49 --steady-tol 0 --T 30`，输出至不同目录，然后：

```bash
python examples/pde/cavity_compare.py build/cavity_n33 build/cavity_2d build/cavity_n49
```

比较器只使用三套网格精确共有的物理节点，不插值；报告的是采样差异和
动能差异，不是全域最大误差或精确解误差。

### 本次实测（CPU / float64）

`Re=100, dt=0.01, T=30, ramp_time=1`，三套网格均运行至相同时刻。
上述测试命令得到 **11 passed**，其中 6 项为新增方腔测试。

| 指标 | 结果 |
|---|---:|
| 41² 最终最大散度 | 1.55e-15 |
| 41² 最终壁面速度误差 | 5.77e-15 |
| 41² 最终最大速度时间导数 | 1.69e-8 |
| 41² 最终动能 | 1.882491e-2 |
| 33² → 41² 共有节点速度最大差异 | 4.6591e-5 |
| 41² → 49² 共有节点速度最大差异 | 8.0387e-6 |
| 33² → 41² 动能绝对差异 | 1.8380e-9 |
| 41² → 49² 动能绝对差异 | 2.9621e-10 |

共有节点为每个方向 `0,1/8,...,1`，共 9×9 点。默认提前停止设置下，
41² 算例在 `t=27` 首次通过采样检查的 `1e-7` 稳态阈值。
这些结果支持本例的网格一致性，不构成标准恒速顶盖基准验证，也未验证高 Re。
