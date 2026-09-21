# 固定光滑磁通边界上的 BSPF Grad–Shafranov 验证

本阶段求解轴对称、静态、各向同性压力的固定边界问题，物理区域严格位于 `R>0`。
场使用已有 BSPF 空间；验证域直接使用解析闭合磁通面，不把边界拟合误差混入场误差。

实现入口：`bspf_models.plasma.grad_shafranov.FixedBoundaryGSPlan`。与原矩形计算壁面上的
`tokamak_equilibrium.solve_equilibrium` 不同，这里直接在给定曲边等离子体域上
施加磁通 Dirichlet 条件。该版本提供给定源项的线性 GS 求解及 Solov'ev 常数剖面接口；
没有加入一般非线性剖面迭代、自由边界更新或 X 点。

## 方程和符号

采用磁通定义

```
B_R = -psi_Z/R,
B_Z =  psi_R/R,
B_phi = F(psi)/R.
```

对应

```
Delta* psi = psi_RR - psi_R/R + psi_ZZ,
-Delta* psi = mu0 R² p'(psi) + F(psi)F'(psi),
psi|Gamma = 0.
```

Solov'ev 模型令 `p'` 和 `FF'` 为常数，方程因此成为线性椭圆问题。
参考：[Li–Zhu 的解析平衡与数值验证论文](https://arxiv.org/abs/1906.05534)
及[轴对称平衡方程推导](https://magnetohydrodynamics.physics.wisc.edu/lecture10.html)。
下面选取这一解析解族中的显式成员，并在测试中独立核对导数、源项和力平衡。

## 两个解析平衡及其真实固定边界

令

```
H(R,Z) = R² log(R/R0) - (R²-R0²)/2 - Z²,
psi(R,Z) = psi_a - a(R²-R0²)² - (bR²+c)Z² - d H(R,Z).
```

基本恒等式为

```
Delta* (R²-R0²)² = 8R²,
Delta* (R² Z²) = 2R²,
Delta* Z² = 2,
Delta* H = 0.
```

因此，无论 `d` 是否为零，

```
-Delta* psi = (8a+2b)R² + 2c,
p' = (8a+2b)/mu0,
FF' = 2c.
```

本例固定 `R0=2, psi_a=0.2, a=0.1, b=0.15, c=0.2, mu0=1, F_edge=3`：
参数采用归一化单位。

- 多项式平衡：`d=0`。
- 对数平衡：`d=0.08`。它不是有限次多项式，用于检查非多项式场的逼近。

两者都有非零的两项源：`p'=1.1`、`FF'=0.4`，即

```
-Delta* psi = 1.1 R² + 0.4,
p(psi) = 1.1 psi,
F(psi) = sqrt(9 + 0.8 psi).
```

域内 `psi>0`，边界 `psi=0`；压力非负，F²为正。磁轴均为 `(R,Z)=(2,0)`，
轴上磁通为 0.2。两例**各自使用自身的 psi=0 边界**，没有在任意曲线上施加变化的
解析磁通后把它称为等离子体边界。径向范围分别约为
`[1.608038,2.326846]`、`[1.620011,2.320213]`。

`SolovevFluxDomain` 使用解析径向截面和受保护 Newton 求根构造边界。
切向与曲率由隐函数微分计算，法向来自真实曲线。
凸性采用包围矩形上 Hessian 正定性的保守充分下界检查，而不只是抽查几个曲率点。
边界为光滑解析曲线，测试中的磁通面几何残差约为双精度舍入量级。

## BSPF 离散

内部基坐标为 `x=R-R0`、`z=Z`，默认辅助盒 `[-1.2,1.2]²`。
这里的 `1/R` 始终是 `1/(R0+x)`，不能误写为 `1/x`。
要求辅助盒也完全位于正大半径区域。

使用原有每方向 N 维 unrestricted BSPF 空间及其解析一、二阶导数，求

```
min_{psi_N}
  ||-Delta* psi_N - S||²_L2(Omega)
  + ||psi_N-g||²_H^(3/2)(Gamma).
```

真实体积分采用解析切片上的复合 Gauss 规则，并通过端点映射消除截面高度的平方根
行为；边界按等弧长采样。背景盒上的真实 H2 Gram 矩阵用于系数右缩放，再进行
截断 SVD，默认相对阈值 `1e-13`。不对辅助盒边缘施加物理边界条件。

这是直接强形式残差离散，沿用现有曲边 BSPF Poisson 路线；它不等于原轴对称
矩形求解器的加权弱形式张量分离算法。当前以可验证的准确性为主，采用稠密矩阵。
同一几何的分解可以复用给多个源项和边界数据。

域几何在构造时已经给定。解析解只用于定义已知的几何和求解后的误差验证。
正式求解只输入 `p'`、`FF'`、`mu0` 和边界常数；没有将域内精确磁通、梯度或 Hessian
作为拟合数据加入系统。

## 使用

```python
import jax
import numpy as np
from bspf_models.plasma.solovev import SolovevEquilibrium
from bspf_models.plasma.solovev import SolovevFluxDomain
from bspf_models.plasma.grad_shafranov import FixedBoundaryGSPlan

jax.config.update("jax_enable_x64", True)
exact = SolovevEquilibrium(logarithmic=0.08)
domain = SolovevFluxDomain(exact)
plan = FixedBoundaryGSPlan(domain, major_radius=2.0, nodes=33)
solution = plan.solve_solovev(
    p_prime=exact.p_prime, ff_prime=exact.ff_prime, mu0=exact.mu0,
    boundary_flux=0.0,
)
points = np.array([[1.9, 0.1], [2.2, -0.15]])
psi, grad_psi, delta_star_psi = solution.evaluate(points)
B = solution.magnetic_field(points, exact.toroidal_function)
diagnostics = solution.validate(exact.source)
```

`evaluate` 第三项为 **正号 Delta* psi**，因此应与 `-source` 比较；不是普通 Laplacian。
`magnetic_field` 的分量顺序为 `(B_R,B_phi,B_Z)`。
所有公开数据回调和求值点使用物理 `(R,Z)`；`domain` 使用局部 `(R-R0,Z)`。

给定其他源项可调用 `plan.solve(source, boundary_flux)`。也支持已有的光滑凸
`SplineDomain`，但给定曲线是否为某个物理解的磁通面取决于边界条件和问题本身。
求解器主动拒绝本阶段不支持的非凸几何与不满足 `R>0` 的设置。

## 验证量和复现

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python \
  -m pytest -q packages/models/tests/test_grad_shafranov.py

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python \
  examples/pde/solovev_bspf_validation.py

MPLCONFIGDIR=/tmp/pybspf-gs-mpl python examples/pde/render_solovev_bspf.py
```

默认 `N=17,25,33,49`，每例体积分 Q16（4096 点）、边界 512 个等弧长点。
独立验证采用偏移均匀点云，边界采用加倍数量并偏移的等弧长点；最高分辨率额外
使用 Q24（9216 点）做独立几何求积。

报告：

- 磁通相对 L2、相对轴上磁通的最大误差；
- 极向磁场相对 L2，以及总磁场误差，避免强环向场掩盖极向场误差；
- 磁通的 H2 误差与真实 GS 残差；
- 独立边界磁通和法向极向磁场残差；
- 数值磁轴位置、轴上磁通；
- 加密求积下的总等离子体电流 `I_p=integral J_phi dR dZ`，其中
  `J_phi=-Delta*psi/(mu0 R)`；
- 力平衡 `J×B-grad p` 的相对残差。这与 GS 残差有代数联系，不是另一套独立精度证明。

这些数值是独立采样/求积下的误差估计，不是严格连续误差证书。
随 N 增长出现舍入或截断平台时如实报告，不要求机器精度附近仍严格单调收敛。
首次构造成本中一维 BSPF 基会跨几何缓存，故两例的 setup 不能直接当成相同冷启动
条件比较；重复 RHS 计时不含输出求值。

结果写入 `build/solovev_bspf/results.json`、`summary.md` 和系数/验证值 NPZ。
另保存源码 SHA256；图为 `convergence.png`、`equilibria.png`。

## 已完成结果（2026-09-19）

四档分辨率、两种解析平衡共 8 个案例全部完成，保存的源码哈希与当前实现一致。
新增 GS 测试及原曲边 Poisson 回归测试合计 **11 项通过**。

以下为每方向 N=49，在两个域各自的 11301 / 11157 个独立点上的相对误差：

| 平衡 | 磁通 L2 | 极向磁场 L2 | 磁通 H2 | GS 残差 L2 |
|---|---:|---:|---:|---:|
| 四次多项式 | 1.08e-14 | 1.38e-14 | 3.60e-13 | 2.65e-13 |
| 含对数项 | 2.27e-14 | 1.14e-14 | 2.78e-13 | 1.99e-13 |

独立边界上最大磁通误差分别为 `4.54e-15`、`6.64e-15`，最大法向极向磁场分别
约 `2.99e-14`、`2.89e-14`。数值磁轴位置误差均小于 `1.5e-14`。

Q24 加密求积给出的 GS 相对残差分别为 `2.61e-13`、`1.95e-13`，与点云结果吻合。
两例总电流分别为 `1.34706462905439`、`1.38625449634269`；与相同独立求积下的
解析电流比较，相对误差分别为 `1.63e-15`、`8.77e-16`。这不是对积分绝对值的
严格舍入误差认证。

从 N=17 到 N=25，GS 残差由约 `1e-10` 降到 `1e-13`，随后进入截断/浮点平台；
N=33、49 不再严格单调改善。对这两个平滑算例，N=25 已足够高精度，不能把
更高 N 的微小波动解释为算法失去收敛，也不能外推到所有 GS 剖面与几何。

已有分解的 N=49 单 RHS 约 5 ms，不含磁场输出求值。构造包含 MPFR 基函数求值
和稠密 SVD；本阶段验证准确性，不把其称为近线性复杂度的快速 GS 求解器。

固定算子多右端项现可使用 `plan.compile_response()`，直接返回求积点上的磁通；
见 [快速响应实现与公平计时](gs_fast_response.md)。初始 SVD 仍是原路径。
