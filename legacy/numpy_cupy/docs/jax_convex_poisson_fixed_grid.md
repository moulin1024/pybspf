# 凸区域 Poisson：固定网格输出

`ConvexPoissonGridPlan` 在内部保留矩形 BSPF 系数表示，对外只返回指定
笛卡尔网格上的解、区域掩码和诊断信息。目标网格可以非均匀，可以不同于
背景构造节点；改变目标网格不改变内部近似空间。

当前版本的**张量加速作用于输出求值**。Poisson 求解仍复用已验证的
`ConvexPoissonPlan` 稠密 SVD 后端，包括它的几何积分、H^(3/2) 边界范数、
H2 系数缩放和截断阈值。首次构造的稠密计算/存储成本没有消失。
此前 `IterativeConvexPlan` 在高精度测试中尚未稳定收敛，不作为此接口的后端。
新加入的 [`TensorConvexPoissonPlan`](jax_convex_poisson_tensor.md) 保持原曲边积分，
提供两层粗空间修正，可通过 `backend="tensor"` 显式选择。它仍处于实验阶段，
未收敛默认报错，不能视为参考后端的等精度替代品。

## 用法

```python
import jax
import numpy as np
from bspf_jax import ConvexPoissonGridPlan
from bspf_jax.embedded_poisson import benchmark_domains

jax.config.update("jax_enable_x64", True)
domain = benchmark_domains()[0]
x = np.linspace(-1.2, 1.2, 65)
y = np.linspace(-1.2, 1.2, 49)
plan = ConvexPoissonGridPlan(domain, x, y, nodes=33)

# -Delta u = -0.2, u|Gamma = 1 + 0.2*x + 0.3*y + 0.1*x^2
def f(p):
    return np.full(len(p), -0.2)

def g(p):
    return 1 + 0.2*p[:, 0] + 0.3*p[:, 1] + 0.1*p[:, 0]**2

result = plan.solve(f, g)
U = result.values           # (49, 65)，U[j,i] 对应 (x[i], y[j])
inside = result.inside      # 同形状 bool，包含落在真实边界上的点
assert np.isnan(U[~inside]).all()
```

`nodes=33` 控制内部 BSPF 近似；`len(x)=65,len(y)=49` 控制输出。
返回对象没有系数、任意点求值或导数接口。不在物理区域内的网格值用 `NaN`
标记，不把辅助矩形上的延拓解释成物理解。

源项和边界数据仍通过回调，在求解器需要的积分点/边界点上提供。这个接口
不把仅给在输出网格上的数组自动插值成 PDE 数据，也没有引入这类插值误差。

同一 plan 可重复调用 `solve(f,g)`，复用稠密分解和固定网格因子。若已有
`ConvexPoissonPlan`，用以下方式避免重复构造求解器：

```python
grid_plan = ConvexPoissonGridPlan.from_plan(existing_plan, x, y)
result = grid_plan.solve(f, g)
```

## 张量计算与成本

内部系数为 `C`，提前缓存 `Bx[a,i]=phi_i(x[a])` 和 `By[b,j]=phi_j(y[b])`。
输出是 `(Bx @ C @ By.T).T`，并依据解析 B-spline 截线交点构造的掩码
将区域外标为 `NaN`。计算不使用二维插值，不组装二维求值矩阵。
相同的 x、y 轴共享同一份基矩阵；较短目标轴优先收缩。

设内部每方向有 N 个基函数，输出为 Mx × My：

- 缓存基矩阵占用 O((Mx+My)N)，输出本身占用 O(Mx My)。
- 两次矩阵乘法需要 O(min(Mx,My)N² + Mx My N) 运算。
- 当 Mx、My 与 N 同阶时，求值为 O(N³)。
- 逐点张量求值需要 O(P N²)，其中 P 为需要输出的物理区域内点数。
- 这里只比较输出成本；内部稠密 Poisson 分解的存储仍可能为 O(N⁴)。

复用一维 MPFR 基函数装配，保证与现有 BSPF 表示一致；装配时会产生现有
例程的导数临时量，但输出缓存和每次求值只保留/计算函数值。

## 验证与复现

```bash
PYTHONPATH=jax/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python examples/pde/convex_poisson_fixed_grid.py --nodes 33 --grid 33

# 需要已有 build/convex_poisson_n65 的参考解；此命令只测输出，不重求 PDE。
PYTHONPATH=jax/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python examples/pde/benchmark_convex_poisson_grid.py --grids 65 129 257

PYTHONPATH=jax/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m pytest -q jax/tests/test_convex_poisson_grid.py
```

示例将网格、掩码和数值解保存为 `grid_solution.npz`，不保存内部系数。
测试覆盖非零 Dirichlet 数据、与原求值结果的一致性、重复求解不重新装配
输出基矩阵、两种收缩顺序、非均匀轴、单点输出、边界切点和无效输入。

性能脚本对张量求值与逐点求值均预先缓存基矩阵，仅计算函数值，报告
21 次运行的中位数。其提速不能解释成整个 Poisson 求解的提速。

2026-09-18 本机单线程测试，使用原 N=65、kmax=12π 的已求解系数：

| 输出网格 | 区域内点数 | 张量求值 | 逐点求值 | 输出提速 | MMS 相对采样误差 |
|---|---:|---:|---:|---:|---:|
| 65×65 | 1383 | 0.0466 ms | 0.449 ms | 9.6× | 2.34e-9 |
| 129×129 | 5555 | 0.125 ms | 2.08 ms | 16.7× | 2.31e-9 |
| 257×257 | 22293 | 0.394 ms | 7.84 ms | 19.9× | 2.31e-9 |

张量与逐点求值的最大差异不超过 1.31e-13。257×257 输出的基矩阵缓存
为 133640 字节，相应逐点双因子缓存为 23184720 字节；这些数字均不包括
求解器分解。目标网格因子初始化耗时分别约 0.42、0.82、1.57 秒，不包含
在上述重复求值计时中。

另外实际从 PDE 数据重新运行 N=33、33×33 输出网格、kmax=4π 的端到端
示例，得到相对采样误差 3.33e-11。该次首次设置约 55.2 秒，分解完成后的
求解约 27.9 ms，输出约 0.0364 ms。首次设置仍是当前实现的主要成本。
