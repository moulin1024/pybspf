# BSPF 二维绕流与出口缓冲层

后续固定网格的伪影诊断、AAA–lightning Stokes 独立参考及可选解析壁面因子，见
[固定网格绕流伪影诊断](jax_immersed_flow_artifacts.md)。下文实测表是原 SVD 方案；
小壁面残差不代表高精度内场，新诊断给出了真实速度/涡量误差。
进一步的 `wall_method="rational"` 已将有理边界修正一致纳入完整 NS 算子，见
[合成空间与非定常 MMS 验证](jax_hybrid_rational_flow.md)。

实现：`packages/models/src/bspf_models/fluids/immersed_flow.py`，运行：
`examples/pde/immersed_channel_flow.py`。沿用之前的偏心解析椭圆孔洞，
求解二维不可压 Navier–Stokes。这里只计算速度/涡量，未重构压力和阻力。
这是宿主 NumPy/SciPy 稠密参考版，并非大规模生产求解器。

## 几何、流动和边界

- 物理区域：`[-1,3] × [-1,1]`，扣去中心 `(0.19,-0.13)`、
  半轴 `(0.31,0.23)` 的固定椭圆。
- 出口缓冲层：`3<x<5`；真正的出流条件在 `x=5`。
- 入流：`u=1-y², v=0`，平均速度 `Umean=2/3`。
- 上下壁面和椭圆壁面：静止无滑移。
- 默认 `Re=Umean*(2b)/nu=20`，故 `nu=0.0153333333333333`。
- 初态：满足入流与所有无滑移条件的离散 Stokes–缓冲层解。
  保持入口通流后，加入对流项演化到 `t=20`。
- 默认每方向背景节点 `73 × 33`，覆盖含缓冲层的整个 `6 × 2` 盒子；
  `dt=0.02`。显示/验证网格 `401 × 161` 不增加求解自由度。

## 孔内场与无滑移壁面

使用同一个全盒 BSPF 流函数

\[
 \psi=\psi_0(y)+B c,\qquad
 (u,v)=(\psi_y,-\psi_x),\qquad
 \psi_0=y-y^3/3+2/3.
\]

散度为混合导数的恒等消去，未再用分量速度裁剪或压力投影。
`grid()` 返回的 divergence 表示这个解析恒等式，不能把恒等返回值
冒充独立的数值误差测量；独立质量检查采用多个截面上的速度积分。

- 原 BSPF y 方向夹持空间保证上下壁面零扰动速度。
- 原 x 空间只对左端的值/一阶导数做零空间约束，右端保持开放。
- 在等弧长孔壁点上对两分量速度构造 SVD 约束：
  `C*c=-U0`，得到一个提升和约束零空间 `c=c_lift+Z*q`。
- 保留奇异值阈值为 `1e-10`；未保留方向导致的孔壁残差通过错位弧长点独立检查。
- **不指定孔壁流函数常数**。它由联合求解决定，从而保留双连通区域
  所需的绕孔流动自由度；不能把一个任意的常数当作额外物理边界条件。
- 孔内流函数仅为光滑辅助延拓，不参与质量、惯性或黏性积分。
  可视化中孔内速度遮罩，不把辅助场画成流体运动。

该速度原型继承光滑全盒表示的思想，但不直接逐步调用前一个标量 Dirichlet
Poisson 求解器。带孔质量矩阵、黏性矩阵和边界约束需要联合构造。
复用的是原 BSPF 空间、原导数、张量求值和原 cavity 时间推进框架。
矩形惯性谱 `lambda_x+lambda_y` 用于初始尺度变换；实际带孔隐式步是缓存
Cholesky 求解，不能声称仍然只需逐模态相除。

## 流体域 Galerkin 与出口

对相同无散度测试速度 w，解

\[
 (\partial_t\mathbf u,\mathbf w)_\Omega
 +((\mathbf u\cdot\nabla)\mathbf u,\mathbf w)_\Omega
 +\nu(\nabla\mathbf u,\nabla\mathbf w)_\Omega
 +(\sigma(\mathbf u-\mathbf U),\mathbf w)_\Omega
 =\langle\min(u_n,0)\mathbf u,\mathbf w\rangle_{x=5},
\]

其中 `U=(1-y²,0)`。对应末端边界为

\[
 \nu\partial_n\mathbf u-p\mathbf n=\min(u_n,0)\mathbf u.
\]

正常出流时右端为零牵引，局部回流时附加耗散。使用的是与
`nu*Laplacian(u)` 弱式一致的 Laplacian 牵引，未替换成对称应力牵引。
对流采用 advective 形式，所以这里没有额外的动压边界修正。
这不是严格无反射边界。

正权积分完全在物理流体中：左右矩形片加椭圆上方/下方两个曲边片。
曲边片用 `x=cx+a*cos(theta)`，y 从真实椭圆线性映射到水平壁面，
保留解析 Jacobian。矩形片在缓冲层起点再拆分。
没有使用网格阶梯几何，也没有对孔内外指示函数做谱微分。

默认积分密度因子从初试的 1.5 提高到 2.5，解析无障碍通流检查的误差
显著改善。矩形 BSPF 含分片样条，曲边积分的光滑性、积分密度和非线性
混叠仍需检查，不将“解析几何”自动当作谱精度的积分证明。

约束后用物理 `M+D` 能量矩阵筛去相对特征值小于 `1e-11` 的不可观测方向，
并做尺度归一化。全部步骤只在固定几何的首次构造执行。

## 平滑缓冲层

\[
 \sigma(x)=3S((x-3)/2),\qquad
 S(s)=\frac{e^{-1/s}}{e^{-1/s}+e^{-1/(1-s)}}\quad(0<s<1),
\]

在 `s<=0` 时为 0，`s>=1` 时为 1。沿用原流函数求解器的 C∞ 平滑函数。
物理区域中阻尼严格为零，缓冲层末端平滑饱和。

缓冲力为 `-sigma*(u-U)`，衰减的是尾流扰动，不是把净通流降到零。
其对扰动速度的功为

\[
 -\int_\Omega\sigma|\mathbf u-\mathbf U|^2\,d\Omega\le0.
\]

离散缓冲矩阵为 `Su.T*W*sigma*Su + Sv.T*W*sigma*Sv`，正半定；
和物理黏性一起隐式推进。没有给物理区域额外添加人工黏性或滤波。
非局部不可压耦合仍可能使缓冲区影响上游，不能由 sigma 局部为零推出
上游解完全独立于缓冲参数。

缓冲检查报告的是 `omega - 2y`：抛物线通流本身的壁面剪切涡量应保留。
有限长度、有限强度的缓冲不能保证尾流扰动在出口严格为零；会报告剩余量。

## 时间推进与验证

原 cavity 的 IMEX midpoint 已提取为 `_flow_kernels.imex_midpoint`，
原 cavity 和新绕流共用该函数：半步后向 Euler 预测，整步中点对流及
Crank–Nicolson 黏性/阻尼。只构造一次 `M+dt*(nu D+S)/2` Cholesky。
对流仍有时间步限制；隐式阻尼没有移除对流 CFL 约束。

测试包括：

1. 曲边积分面积、一阶和二阶矩与矩形减椭圆的解析结果一致。
2. 无障碍 Poiseuille 精确解、缓冲支持区以及负扰动功。
3. 非平凡连续 NS MMS：独立解析对流、黏性、线性压力、时间项和缓冲体力，
   同时验证残差及二阶时间收敛；未用数值 RHS 制造体力。
4. 偏移孔壁点的无滑移误差、入流/上下壁条件、出口净通量。
5. 原 cavity 的物理边界和连续 MMS/时间阶回归。

运行结果另独立检查八个截面的通量，包括穿过椭圆的上下两个流体间隙。
已进行两档网格和积分密度比较；壁面满足到舍入量级不代表内场已空间收敛。

## 实测结果

最终参考结果在 `build/immersed_flow/dense/`，采用 73×33、积分因子 2.5、
dt=0.02、t=20、Re=20、缓冲长度 2、最大阻尼 3。

|指标|结果|
|---|---:|
|孔壁最大速度|6.535e-12|
|入口/上下壁最大误差|2.130e-14|
|八个截面最大相对流量误差|4.247e-14|
|最终加速度 L2|1.203e-07|

缓冲入口 x=3 到末端 x=5 的截面 L2：

|扰动|入口|末端|下降比例|
|---|---:|---:|---:|
|速度相对通流|0.17162|0.00642|96.26%|
|涡量相对通流剪切|0.69378|0.15200|78.09%|

扰动涡量沿缓冲区不是单调下降，出口仍有残余，不能声称完全吸收或无反射。
同样高积分密度下，49×25 与 73×33 的物理区速度相对差为 6.737%，
涡量相对差为 47.355%，尚未达到空间网格无关。
固定 73×33，将积分因子 1.5 提高到 2.5 后，速度相对变化为 0.037%。
完整区域比较在 `build/immersed_flow/comparison.json`。不能把很小的壁面/通量误差
当成内场误差；本次结果只确立了可运行的稳定绕流及缓冲出流原型。

本机单线程 BLAS 首次构造 115.4 s，推进和保存显示网格耗时
277.2 s；独立后验检查另计。保留 1932 个速度自由度，
曲边积分 17881 点。固定几何/时间步的 Cholesky 因子复用。

10 项绕流/cavity 测试通过，包含解析 NS MMS、二阶时间收敛和负缓冲功。

## 复现

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python examples/pde/immersed_channel_flow.py --nx 73 --ny 33 --quadrature-factor 2.5 --out build/immersed_flow/dense
MPLCONFIGDIR=/tmp/bspf-mpl python examples/pde/render_immersed_flow.py --out build/immersed_flow/dense --movie
python -m pytest -q packages/models/tests/test_immersed_flow.py packages/models/tests/test_cavity.py
```

可调整 `--buffer-length`、`--buffer-strength`。增加长度时应相应增加 nx，
保留物理区域分辨率；修改阻尼强度需要重新构造当前计划。
`--buffer-length 0` 禁用缓冲并在原物理出口施加牵引边界。
`--quadrature-factor 1.5` 可复现初试积分对照。

结果含 `summary.json`、`history.json`、`fields.npz`、`flow.png`、
`diagnostics.png`，指定 `--movie` 后另生成 `flow.mp4`。

## 范围

默认是 Re=20 层流，出现稳定尾流/回流区，不把它称为已验证的卡门涡街。
当前没有压力、阻力/升力、运动边界或 MHD 的实现，也没有给出完整 NS 的
谱收敛结论。稠密组装及多次稠密矩阵乘法使大网格成本很高；本版用于建立
正确性和稳定性的参考结果。
