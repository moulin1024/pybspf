# 非周期 BSPF：无源离子声波包与温和端点结点加密

这是有自洽电场的、无体积制造源项的线性静电回旋动理学算例：离子声波包沿均匀磁场传播，从长度 6 的开放磁力线两端流出。电子采用绝热响应，保留一个非零垂直波数的完整 Bessel FLR 响应。物理背景可参见 [Gkeyll 的离子声波说明](https://gkeyll.readthedocs.io/en/latest/dev/collision-models.html)。

此算例验证非周期 BSPF 空间离散及开放边界收支；它是线性化、对称性约化的 GK 子问题，不是完整非线性五维验证，也不是鞘层边界模型。

## 方程与独立参考解

取两个垂直模态 $k_\perp=0,1$，$\rho=0.8$、$\tau=1$。设

$$g_m(z,v,\mu)=J_0(k_m\rho\sqrt{2\mu})u_m(z,v),\qquad
\Gamma_m=\sum_j w_{\mu,j}J_0^2(k_m\rho\sqrt{2\mu_j}),\qquad
\lambda_m=\frac{\Gamma_m}{\tau+1-\Gamma_m}.$$

该速度形状在线性问题下构成不变子空间。令 $s_i=\sqrt{w_{v,i}}$、$x_i=s_i u_i$，则每个模态满足

$$\partial_t x+V\partial_z(x+\lambda s s^T x)=0,\qquad
\phi=\lambda s^T x.$$

初始波包为

$$x_m(z,v,0)=A_m s(v)e^{-(v-1.1)^2/4}
\left[\max\left(1-\left(\frac{z+0.4}{1.8}\right)^2,0\right)\right]^{10},
\quad A=(0.01,0.003).$$

这是紧支撑、$C^9$、非负的扰动波包，初始支撑严格位于区间内部。令 $H=I+\lambda ss^T$，对称矩阵 $H^{1/2}VH^{1/2}=UCU^T$ 给出实特征速度。用 $R=H^{-1/2}U$ 将初值变换到特征变量，每个分量直接平移 $z-c_jt$，得到独立参考解。两端**入射等离子体特征为零**，出射特征自由流出。边界不是周期拼接，也不是按单粒子速度正负设置的吸收壁。

参考解对给定速度求积系统精确，对连续速度方程不宣称解析精确；速度加密单独检验。默认验证使用 192 点 Legendre 求积覆盖 $v\in[-8,8]$，含 Maxwell 权重；磁矩用 32 点 Laguerre 求积。192 对 384 点的电势最大差为 $9.04\times10^{-14}$；截断从 8 扩至 10 的差为 $6.44\times10^{-16}$。

## 真正使用 BSPF 的部分

$z$ 方向通过 `plan_fast_axis` 构造非周期 BSPF 弱导数、质量逆和边界提升。FFT 只处理样条扣除后的周期残差；非周期端点信息由样条和数值通量处理。运行时没有显式 $N_z\times N_z$ 轴矩阵。速度特征分解是独立的小型速度矩阵。

7 次样条、16 个基函数、12 点逐网格单元 Gauss 求积；RK4 的 $\Delta t=0.0005$，终止时间 1.2。未加滤波、截负或收支修补。

原均匀样条结点基线：

| $N_z$ | 分布相对 $L^2$ 误差 |
|---:|---:|
|49|$5.4004\times10^{-6}$|
|65|$3.8830\times10^{-7}$|
|97|$9.3706\times10^{-9}$|
|129|$6.9053\times10^{-10}$|
|193|$7.7361\times10^{-10}$|

49 到 129 呈高阶收敛，193 出现平台，不能把这些结果宣称为任意细网格的持续高阶收敛。129 上时间步减半误差仍为 $6.9047\times10^{-10}$；改为 5 次样条误差为 $3.7400\times10^{-8}$；提高求积至 16 点结果为 $6.9403\times10^{-10}$。

## 温和端点加密

`sample_aligned_knots(..., endpoint_blend=0.5)` 使用

$$\xi_j=(1-\alpha)\frac{j}{K}+\frac{\alpha}{2}
\left[1-\cos\left(\pi\frac{j}{K}\right)\right],\qquad \alpha=0.5,$$

再将结点对齐到最近的均匀采样位置。样条基数、次数、端点约束不变；FFT 采样网格仍均匀。保留所有内部区间，不将样条结点全部挤到端点。若网格太粗导致结点重合，会明确报错。

129 点时样条断点索引由 `0,14,28,43,57,71,85,100,114,128` 改为 `0,9,22,37,55,73,91,106,119,128`。采样到样条系数映射的二范数从 $4.42\times10^7$ 降为 $1.89\times10^7$；质量逆低秩核心的条件数却从约 126.24 变为 126.90。因此“降低某一误差放大因子”不能等同于“所有条件数改善”。

温和加密的分布误差依次为：49 点 $5.2380\times10^{-6}$、65 点 $3.7971\times10^{-7}$、97 点 $9.0244\times10^{-9}$、129 点 $6.7606\times10^{-10}$、193 点 $5.2431\times10^{-10}$、257 点 $1.2027\times10^{-9}$。同一 257 点均匀结点误差为 $4.3482\times10^{-9}$，温和加密降低约 3.6 倍。因此它改善了部分误差放大与中等分辨率表现，但**没有消除细网格演化误差平台**。257 点时，独立参考函数的 BSPF 插值误差为 $3.15\times10^{-12}$，明显小于演化误差；不能只根据插值精度宣称 PDE 已恢复高阶收敛。

细网格实际演化对比见 `build/open_slab_packet/mild_knots.json`、`uniform_knots.json` 和 `knot_comparison.png`。两条曲线分别固定一种结点规则，不拼接出人为更好的收敛曲线。

## 收支与边界验证

对每个模态的自由能使用 $\Gamma_m$ 和实余弦平均系数 $(1,1/2)$：

$$W=\frac12\sum_m\Gamma_m c_m\int(x_m^Tx_m+\lambda_m n_m^2)\,dz.$$

同时累计出射通量和入射数值罚项，检验

$$N(t)-N(0)+\int_0^t\mathcal F_N\,dt=0,\qquad
W(t)-W(0)+\int_0^t(\mathcal F_W+\mathcal D_{\rm boundary})\,dt=0.$$

这里能量指线性扰动的**自由能**，粒子量指空间平均模态的扰动粒子数。罚项单独记录，不混入物理出射能流。初始和终态的参考积分还通过不完全 beta 函数独立计算，避免只检查同一套离散公式的自洽性。

均匀结点 129 点时，扰动粒子流出约 4.256%，自由能流出约 2.087%；相对粒子/自由能收支缺陷约 $1.13\times10^{-11}$ / $5.66\times10^{-13}$，罚项约 $2.10\times10^{-18}$。错误地使用周期参考解会造成约 19.0% 的分布误差。节点上的总分布下界 $F/F_0$ 约 0.993；这只是采样点检查，不是连续域正性证明。

## 复现

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python examples/pde/open_slab_packet.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python examples/pde/validate_open_packet_knots.py
python -m pytest packages/models/tests/test_open_slab_packet.py tests/test_fast_axis.py -q
```

条件与插值扫描可另运行 `examples/pde/diagnose_open_packet_knots.py`。

输出在 `build/open_slab_packet/`，包含原始 JSON、129 点时序 NPZ 和静态图。

## 回查简单 BSPF 求导：端点估计与快速系数映射

参考历史的 [求导收敛诊断](convergence_diagnosis.md) 和 [压力构造误差诊断](pressure_accuracy_diagnosis.md)，对样条结点固定为 50% 混合加密的算例，只改变端点估计。原始一维 notebook 使用 `tanh` 结点加密及样条正则化，但那些参数不是已经证明适用于本波包的修复。

简单非周期函数 $f(z)=\exp(z/3)+\sin(1.3z)$，257 个相同评价节点，7 次样条、16 个基函数、9 点有限差分端点估计：

| 系数计算路径 | 一阶导数最大绝对误差 |
|---|---:|
| 标准 `differentiate`，按输入构造 RHS 后解小型约束系统 | $3.3662\times10^{-13}$ |
| `FastAxis.coefficient_map @ f`，随后同样计算样条与 Fourier 导数 | $1.1093\times10^{-9}$ |

两者样条系数最大差约 $2.86\times10^{-5}$。这直接显示当前预计算系数映射路径损失了精度；并非 BSPF 在该网格上只能达到 $10^{-9}$。该对照同时涉及标准未缩放求解与快速轴行缩放预计算路径，尚未完全分离系数预计算误差与应用时消去的各自贡献，也没有证明波包平台只有这一来源。

Chebyshev 端点估计已经采用增广 QR。12 模态、24 点窗口时，257 点波包误差从 $1.2027\times10^{-9}$ 降到 $1.1544\times10^{-10}$；但129点同一窗口误差为 $6.2809\times10^{-8}$，比9点有限差分基线更差。257 点继续加宽至32点后，波包误差回升至 $4.0470\times10^{-10}$。因此不能用一个更宽的固定窗口直接替代所有网格默认值。高阶端点权重的舍入放大和有限阶拟合的逼近偏差必须分别检查。

工厂现支持 `endpoint_method`、`boundary_points` 和 `chebyshev_modes`，历史默认不变。所有选项仍使用轴级快速演化，没有引入稠密空间轴矩阵。下一步应在保留该结构的前提下检验更稳定的分解式系数应用、端点尺度平衡及预计算精度，而不是仅增加结点加密或修改时间步。

复现脚本：

- `examples/pde/diagnose_fast_axis_derivative.py`：相同节点的标准/预计算系数路径对照。
- `examples/pde/diagnose_open_packet_endpoints.py`：129、257点，有限差分9点与Chebyshev 12模态的16/24/32点窗口对照；包含真实波包演化。
- 结果：`build/open_slab_packet/same_nodes_derivative.json` 与 `endpoint_comparison.json`。

## 固定保守窗口的完整重扫

固定 Chebyshev 12 模态、16 点窗口，温和结点加密系数0.5；其他参数不变。所有网格重新时间推进至 t=1.2，未拼接选择不同窗口。

| N | 分布相对 L2 误差 | 相邻实测阶数 |
|---:|---:|---:|
|49|5.13258e-05|—|
|65|2.26464e-06|10.85|
|97|2.63152e-08|10.99|
|129|1.12445e-09|10.96|
|193|6.36589e-10|1.40|
|257|1.18342e-09|-2.16|

49至129点约11阶的实测下降不延续至细网格；193至257误差回升。固定16点窗口未消除平台，257点与FD9的1.20e-9接近。粒子和自由能最大相对收支缺陷分别为3.19e-10、6.06e-12。

复现：`python examples/pde/validate_open_packet_conservative.py`。结果保存在 `build/open_slab_packet/conservative_window.json`、`conservative_window.png`、`conservative_window.pdf`，257点完整检查时序为`conservative_window_257.npz`。
