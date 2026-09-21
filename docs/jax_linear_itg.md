# 标准 BSPF 线性 ITG 原型

实现了一个静电、无碰撞、绝热电子的线性 ITG 模型，含平行运动、常曲率磁漂移、固定温度/密度梯度和完整 Bessel FLR。径向区间有限，使用标准 BSPF；径向网格限制在129点以内。

这是**无磁剪切、常局部系数的简化模型**，尚非 Cyclone 托卡马克基准、全局非线性湍流或输运雪崩。背景梯度取参考磁面处的局部值，不把一个空间变化很大的 Maxwellian 当成已实现的全局平衡。单个非零 $k_y$ 和单个 $k_\parallel$ 谐波在线性情况下独立演化，径向保留全部 $N_x-2$ 个自由度，两个速度变量均参与动力学。

物理方程可与 [局部曲率场中的 GK 色散研究](https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/an-analytical-form-of-the-dispersion-function-for-local-linear-gyrokinetics-in-a-curved-magnetic-field/7F750F4624B9C87D2FEC98413D50DF1A) 和 [局部线性 ITG 模型](https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/local-gyrokinetic-collisional-theory-of-the-iontemperature-gradient-mode/D109ED9613512699DA65195FB0BF704B) 对照。本项目没有把这些不同参数/几何下的结果当成自己的数值基准。

## 方程和符号

速度 Maxwell 权重为

$$d\Lambda=\frac{e^{-v^2/2}}{\sqrt{2\pi}}\,dv\,e^{-\mu}\,d\mu,
\qquad \epsilon=v^2/2+\mu.$$

用归一化非绝热分布 $h/F_0=g+\mathcal J_\mu\phi$，演化互补扰动 $g$。它不是完整粒子坐标下的 $\delta f/F_0$。定义

$$\Omega=k_\parallel v+\rho k_y\kappa(v^2+\mu),\qquad
\omega_*^T=\rho k_y[a_n+a_T(\epsilon-3/2)],$$

$$\partial_t g=-i\Omega(g+\mathcal J_\mu\phi)
+i\omega_*^T\mathcal J_\mu\phi,$$

$$[1+\tau-\Gamma]\phi=\int \mathcal J_\mu g\,d\Lambda,
\qquad \Gamma=\int\mathcal J_\mu^2\,d\Lambda.$$

$\tau=T_i/T_e$。正的 $\kappa$ 和 $a_T$ 对应本文选择的坏曲率约定；径向漂移取 $v_{Ex}=+i\rho k_y\mathcal J_\mu\phi$，$a_n,a_T$ 表示向外递减背景的正对数梯度系数。所有参数和频率都是模型的归一化量；$x,y$ 使用参考回旋半径尺度、时间使用参考平行长度/热速度尺度，梯度和曲率系数含该长度归一化。`rho` 控制相对于径向网格尺度的 FLR 半径，默认1。

默认参数：$L_x=12$，$k_y=0.3$，$k_\parallel=0.1$，$\kappa=0.2$，$a_n=0$，$a_T=4$，$\rho=\tau=1$。非零 $k_y$ 模态采用绝热电子响应；当前 API 拒绝 $k_y=0$，避免把未实现的核心带状模态闭合混进来。

固定梯度驱动由**自洽电势**产生，RHS 不读取任何解析增长率、参考解或外部制造源。提供参考频率只能设置用于检验的初值；普通密度初值也可以增长。

## 标准 BSPF 与非周期径向 FLR

1. 在 $[0,L_x]$ 上对扰动施加齐次 Dirichlet 条件，两端互不识别。
2. 使用7次样条、16个样条基、9点端点差分，样条结点采用50%均匀—余弦混合并对齐均匀采样。每一列通过标准 `decompose` 小型 KKT 求解获得 BSPF 表示，**不使用 FastAxis 的预计算系数映射**。
3. 以全部 $N_x-2$ 个离散正弦列作为齐次边界子空间的良态坐标。它们只是节点上的可逆基变换：求积点上的值与导数均由 BSPF 计算，不能直接替换成解析正弦/余弦。
4. 对 BSPF 函数和导数作分段 Gauss 求积，构造 $M=Q^TWQ$ 和 $K=G^TWG$。小型径向广义特征问题 $KU=MU\Lambda$ 给出 $U^TMU=I$。
5. 定义

$$\mathcal J_\mu=J_0\!\left(\rho\sqrt{2\mu(-\Delta_{x,D}+k_y^2)}\right).$$

在上述径向特征基中，$J_0$ 的自变量使用**数值 BSPF 特征值**，$\Gamma$ 和极化分母用同一速度求积构造。这使回旋平均与电荷沉积在质量内积下互为伴随。

这里明确采用 Dirichlet 拉普拉斯的谱函数作为有限区域 FLR 闭合，连续极限可理解为径向奇反射延拓。它不是粒子撞墙/鞘层模型，也不是此前波包的平行出射边界。真实全局径向边界与缓冲层需要后续重新验证，不能由这个数学闭合直接推断壁面物理。

为保证正确性，本阶段有意保留小型稠密**径向**质量、刚度和特征变换矩阵；并非先前的 FastAxis 后端。运行时在径向特征坐标中应用 FLR 和动力学，不组装整个相空间矩阵。由于局部系数恒定，不同径向模态在这个线性原型中独立；更一般的背景剖面会使其耦合。

## 自由能及粒子输运的含义

演化加权变量 $X=\sqrt{w_vw_\mu}\,g$，避免速度尾部除以极小权重。令 $b=\sqrt w J_0$、$D=1+\tau-\Gamma$，单个复振幅的自由能约定为

$$W=\frac12\sum_r\left(\sum_j|X_{rj}|^2+D_r|\phi_r|^2\right)>0.$$

对应实谐波的空间平均再乘共同的 $1/2$。离散算子满足

$$\dot W=P_n+P_T=a_n\,\mathcal F_n+a_T\,\mathcal Q_T,$$

其中 $\mathcal Q_T$ 用 $\epsilon-3/2$ 加权，是与温度梯度共轭的热通量。密度、温度驱动功分别输出。梯度关闭时，平行运动与磁漂移只重新分配自由能，不产生增长的正定自由能。RK4 同阶段积分驱动功，检查 $W(t)-W(0)-\int(P_n+P_T)dt$，没有事后能量修补。

单个非零 $k_y$ 扰动的总一阶粒子数在周期 $y$ 平均后恒为零，不能把它宣传为非平凡的粒子收支验证。诊断中的径向粒子/热通量是二阶相关量；目前不反馈到背景剖面，后续必须加入零模态和剖面方程。

## 独立参考与实测结果

参考径向波数使用连续 Dirichlet 解 $k_x=m\pi/L_x$。非绝热 $h$ 形式的色散关系为

$$1+\tau=\int J_0^2\frac{\omega-\omega_*^T}{\omega-\Omega}\,d\Lambda,
\qquad e^{-i\omega t},\quad \gamma=\operatorname{Im}\omega.$$

参考模块不导入生产 RHS 或 BSPF 径向算子。提供三层对照：

- 独立离散速度矩阵的特征值（小速度网格，仅用于测试）；
- 独立 NumPy/SciPy 求积的复色散根；
- 解析完成平行速度积分（复误差函数），再对磁矩作自适应积分的**连续速度参考**。上半平面限定明确：没有将离散速度实轴极点当成 Landau 阻尼模态。

默认第一径向模态，48×32速度求积，$\Delta t=0.01$、$t=12$：

| 量 | 时间推进 | 同速度离散参考 | 连续速度参考 |
|---|---:|---:|---:|
| $\omega_r$ |0.3208994299643|0.3208994299640|0.3209591074543|
| $\gamma$ |0.1639919025732|0.1639919025735|0.1640395824059|

时间推进对同离散色散根的差约 $4\times10^{-13}$，但默认速度求积的复频率误差约 $7.64\times10^{-5}$。二者不能混淆。速度求积加密到256×192时，该误差降到 $2.23\times10^{-9}$；这个较细速度结果来自独立色散求积，并未运行对应完整径向时序。

普通密度初值、32×24速度网格、$t=60$：最后三分之一时段拟合 $\gamma=0.16389594$，参考0.16390211；频率误差约 $2.93\times10^{-6}$。它验证增长不是通过不断灌入解析本征解制造的。

默认本征初值的自由能收支缺陷除以运行中的最大自由能为 $3.45\times10^{-12}$，除以初始自由能为 $1.77\times10^{-10}$。普通初值长时增长的这两个数分别为 $5.65\times10^{-11}$ 和约0.100；指数增长使“除以初始能量”的误差指标很敏感，两种归一化均保存在 JSON 中。

### 径向和时间收敛

第一径向模态的本征函数在49点已经达到约 $10^{-14}$ 的绝对 $L^2$ 平台，因此不从它拟合高阶。改用独立的第7径向模态测试完整 FLR 作用（$\mu=0.4$）：

| $N_x$ | FLR 作用的相对 $L^2$ 误差 |
|---:|---:|
|49|$1.4656\times10^{-7}$|
|65|$1.6097\times10^{-8}$|
|97|$1.1111\times10^{-9}$|
|129|$1.4467\times10^{-10}$|

这验证有限径向区间的 BSPF/FLR 算子，而不是把解析正弦拉普拉斯直接用于生产计算。单个最低径向模态的增长率已不受径向分辨率限制。

另以小速度网格进行 $\Delta t=0.2,0.1,0.05$ 的 RK4 检验，增长率误差分别为 $5.52\times10^{-8}$、$3.59\times10^{-9}$、$2.29\times10^{-10}$，呈约四阶下降。

### 梯度与波数扫描

绘制连续参考的 $a_T=1.7$ 至8、$k_y=0.15$ 至1的增长率和频率；并对若干梯度和波数额外运行 BSPF 时间推进，标为独立交叉验证点。数值点使用48×32速度网格，故不应要求其与连续参考逐位相同。

固定其余默认参数，跟踪同一 ITG 分支到正增长率 $0.003,0.001,0.0003,0.0001$，对应 $a_T$ 约1.65827、1.63958、1.63309、1.63124；外推该分支的边际点为 **$a_{T,c}\approx1.63$**。这是这一简化几何、波数、密度梯度和分支的结果，不是所有 ITG 模式的全局稳定阈值，更不是托卡马克普适临界梯度。

## 使用与复现

```python

import bspf_models.kinetic.linear_itg as bspf_linear_itg
import jax
jax.config.update('jax_enable_x64', True)
import pybspf as b

radial = bspf_linear_itg.plan_itg_radial(65)
p = bspf_linear_itg.plan_linear_itg(radial, a_t=4.0, ky=0.3)
x0 = bspf_linear_itg.linear_itg_initial(p)  # 普通密度初值，不使用参考根
history, times, work = bspf_linear_itg.integrate_linear_itg(
    p, x0, 0.01, steps=1200, save_every=20)
phi_modes, _ = bspf_linear_itg.linear_itg_fields(p, history[-1])
phi_at_nodes = p.radial.samples @ phi_modes
```

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/bspf-itg-mpl \
  python examples/pde/linear_itg.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m pytest packages/models/tests/test_linear_itg.py -q
```

`build/linear_itg/` 保存 `report.json`、`reference_scans.json`、`solution_129.npz`、`generic_seed.npz`、`validation.png/pdf` 和 `onset_and_seed.png`。第一个时序文件保存电势、能量、驱动功及径向形状，不是完整五维分布快照。

下一阶段是非线性 $E\times B$ 模态耦合和正确的带状模态电子闭合。之后再加入加热/排热及自洽背景演化，研究输运雪崩；当前模型不会发生非线性饱和。

本轮线性 ITG、开放波包和原 slab GK 的相关回归测试共19项通过。
