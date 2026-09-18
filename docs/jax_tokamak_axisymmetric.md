# BSPF 轴对称托卡马克：可变形的线性垂直不稳定模

此文档描述保留的外部流体版本。新的 [等离子体—真空界面耦合](jax_tokamak_vacuum.md)
已经独立实现，外部不再有流体自由度；工程导向案例见 [被动壁稳定化](jax_tokamak_confined.md)。

实现入口：[`tokamak_axisymmetric.py`](../examples/pde/tokamak_axisymmetric.py)。
磁平衡与线性模型分别位于
[`tokamak_equilibrium.py`](../jax/src/bspf_jax/tokamak_equilibrium.py) 和
[`tokamak_linear.py`](../jax/src/bspf_jax/tokamak_linear.py)。

## 已实现的物理范围

- 矩形柱坐标截面 `R∈[1,3], Z∈[−1.6,1.6]`，完整保留环向几何因子。
- 无顶盖运动、无基态流动；12 个矩形域外的轴对称环形细线圈，电流固定。
- 自洽 Grad–Shafranov 平衡，等离子体轮廓由中心连通区域的 `psi=0` 磁通面决定。
- 四个二维扰动场同时演化：极向速度流函数、环向速度、极向磁通、环向磁场。
- 不可压、轴对称 `n=0`、线性化。用上下反射奇偶性选取垂直模分支，允许内部流动和变形。
- 外层矩形边界静止、固定极向磁通和环向磁场；线性计算中不移动线圈、不设置位移反馈或外加驱动力。

**外部区是低密度、高电阻介质近似真空，尚未实现精确真空界面耦合。**
外边界的影响也没有做域尺寸收敛。本例没有电阻真空室壳层、反馈控制、
温度方程、可压缩效应、触壁、电流淬灭或非线性 VDE。
“完整二维场”指上述不可压模型的全部三个速度/磁场分量，不指完整扩展 MHD。

所有量归一化，`mu0=1`，没有指定真实装置、密度、磁场和长度单位，因此增长率
不能直接换算为 s⁻¹，也不能作为任何实际托卡马克的预测。

## 自洽平衡与固定线圈

采用

\[
B_R=-\psi_Z/R,\quad B_Z=\psi_R/R,\quad B_\phi=F_0/R,
\quad F_0=6,
\]
\[
-\Delta^*\psi=Rj_\phi,\qquad
\Delta^*=\partial_{RR}-R^{-1}\partial_R+\partial_{ZZ}.
\]

中心连通的正磁通区域中，`j_phi=alpha*R*psi²`，`p=alpha*psi³/3`；其余区域
`j_phi=p=0`。迭代时用总等离子体电流 `Ip=∫j_phi dR dZ=1` 确定 alpha。
平衡区中的 `j×B=grad(p)`，外部极向场为真空场。

线圈磁通和磁场由圆电流环的椭圆积分 Green 函数计算。示例先将 6 对上下对称
线圈拟合到一个真空多极场，再冻结所得电流；这一步用于定义示意装置，而非
把运动中的等离子体位置作为控制目标。确切线圈位置和电流保存在 summary.json。

总磁通拆成已知线圈场与 BSPF 等离子体响应。矩形外边界上响应取零；这是
固定磁通的有限域约束，并非无限空间自由边界。
等离子体本身的轮廓未作为固定几何边界施加，它与矩形外壁之间有外部区。

弱 GS 算子为

\[
\int \frac{\nabla\psi\cdot\nabla v}{R}\,dR\,dZ
=\int j_\phi v\,dR\,dZ.
\]

用原有 BSPF 基构造加权径向质量/刚度与轴向因子，通过一维广义特征分解求解。
平衡初始化保持上下对称，避免不稳定平衡在迭代中发生数值漂移；迭代收敛率
不是物理增长率。等离子体接触外壁或平衡不收敛时，程序报错而非输出动画。

## 线性场方程

背景流动为零。速度与磁场扰动满足

\[
\rho_0\partial_t\delta u=-\nabla\delta p
+\delta j\times B_0+j_0\times\delta B+\nabla\cdot(2\mu\epsilon(\delta u)),
\]
\[
\partial_t\delta B=\nabla\times(\delta u\times B_0)
-\nabla\times(\eta\delta j),\qquad
\nabla\cdot\delta u=\nabla\cdot\delta B=0.
\]

极向速度表示为 `u_R=−chi_Z/R, u_Z=chi_R/R`，极向磁场同样用磁通表示。
环向速度、环向磁场都是独立二维未知量；没有把它们冻结或省略。
无散度速度检验空间消去压力，积分包含柱坐标权重 R（共同的 2π 因子略去）。

默认等离子体内 `rho=1, eta=0`，外部 `rho=0.01, eta=1`，动力黏性 `mu=1e-4`。
这些外部系数在平衡界面上分段定义；线性质量矩阵使用背景密度。
这是一种有限外部惯性和电流弛豫近似，不自动等价于严格的 `j=0` 真空。

速度无滑移条件只施加在矩形外壁，**不施加在等离子体轮廓上**。
扰动极向磁通和环向磁场在外边界固定，外部线圈电流不变。
环向场采用 Dirichlet 条件，不是通用理想导电壁所要求的切向电场自然条件，
因此本例不能称为严格的理想导电壁模型。
磁通重建 `psi0+delta_psi` 的中心闭合零磁通轮廓随扰动变化；真空中的其他
同值开放磁通支不是等离子体轮廓，渲染时予以区分。

耦合组装成 `M y_t=A y`。磁场增量导致的 Lorentz 力与感应项采用互为伴随的
离散，满足线性扰动能量收支：

\[
\frac{dE_1}{dt}=\int\delta u\cdot(j_0\times\delta B)\,R\,dR\,dZ
-D_\mu-D_\eta.
\]

其中平衡电流项提供可能的自由能，不是人为加上的增长率或运动源项。
实现中混合 Galerkin 电流与强导数电流并非逐点相同，因此另行检查 GS 强残差。

## 特征模与独立时间推进

先求 `A v=gamma M v`，并在原始方程中验证特征残差。选择正实增长率的垂直模
作为小初始扰动，再用 Crank–Nicolson 隐式格式真正推进矩阵方程，保存各帧场量。
动画不是把指定几何轮廓平移，也不是直接把 `exp(gamma*t)` 当作时间求解结果。
图中的指数曲线仅作独立特征值预测参考。

初始平均竖直模态位移为 `1e-4*a`，结束于约 `0.02*a`。位移由增长模的
`xi=u/gamma` 定义，等离子体内按体积加权；报告的 centroid_z 是这一线性
位移诊断，不是额外求解的非线性密度界面。最后最大局部位移约 `0.03*a`，
因此只解释为小扰动增长，不能外推为有限幅度触壁轨迹。
零扰动对照在齐次线性系统中严格保持零，这只检查没有额外驱动，**不能替代
对平衡精度的独立检查**。

## 本次结果

中心等离子体伸长比约 1.635，归一化小半径约 0.5375。

| BSPF 节点 | 正增长率 gamma |
|---|---:|
| 33² | 0.1571022 |
| 41² | 0.1529961 |
| 49² | 0.1492362 |

49² 的时间拟合增长率为 **0.1492389**，与特征值相差 `1.85e-5`（相对）。
特征残差 `1.04e-11`；GS 弱残差 `2.95e-10`，强导数电流相对 L2 残差约 `1.22e-3`。
流体位移场相对均匀竖直平移的体积加权 L2 差异约 35%，表明内部流动并未被
约束成刚体。该指标包含径向和环向位移分量，不是边界形变量的百分数。

固定平衡和线圈，仅修改外部近似：

| 49² 外部配置 | gamma |
|---|---:|
| rho=0.01, eta=1 | 0.1492362 |
| rho=0.005, eta=1 | 0.1504203 |
| rho=0.01, eta=2 | 0.1563014 |

41²→49² 的增长率变化约 2.46%；外部电阻加倍产生约 4.73% 的变化。
**存在增长模的定性结论在这些检查中保持，但增长率尚未达到精确真空/网格
收敛。** 没有完成外边界位置、外部密度/电阻极限及真实装置验证。

另一个较圆的线圈配置（伸长比约 1.154）在 `gamma=0.05` 附近检查的 12 个
垂直分支特征值均为衰减模；这不是对完整谱的全局稳定性证明。

新增 **7 项测试通过**：圆线圈场的独立差分检验、解析多项式 GS 解、自由轮廓
与真空电流、完整三分量能量交换、独立 `curl(u×B)` 感应检验、柱坐标散度/壁面条件、
增长特征对和外部参数变化。

## 复现

从仓库根目录运行，依赖 JAX、NumPy、SciPy、gmpy2；渲染另需 Matplotlib 和 imageio-ffmpeg。

```bash
PYTHONPATH=jax/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python examples/pde/tokamak_axisymmetric.py --n 49 --sensitivity \
  --out build/tokamak_axisymmetric_n49

MPLCONFIGDIR=/tmp/bspf-mpl python examples/pde/render_tokamak_axisymmetric.py \
  build/tokamak_axisymmetric_n49

PYTHONPATH=jax/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python examples/pde/tokamak_axisymmetric.py --n 33 --control \
  --out build/tokamak_axisymmetric_control

PYTHONPATH=jax/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m pytest jax/tests/test_tokamak.py -q
```

主算例输出 `summary.json`、`evolution.npz`、本地调试缓存 `model.pkl`，渲染输出
`tokamak_vertical.png` 和 `tokamak_vertical.mp4`。代码使用稠密小规模特征/线性代数，
尚不是大网格可扩展实现；提高 n 会显著增加内存与分解开销。
只有自行生成且可信的 pickle 缓存才应加载，并须在加载前启用 JAX float64。

物理背景参考：[轴对称 VDE 三代码基准](https://arxiv.org/abs/1908.02387)、
[固定与自由边界磁平衡说明](https://freegs.readthedocs.io/en/stable/creating_equilibria.html)。
本例的 BSPF 求解器自行实现，没有调用这些程序求解，也未声称复现其 NSTX 基准。
