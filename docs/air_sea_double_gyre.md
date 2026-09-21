# 水平二维双环流海洋—大气混合层双向耦合：第一版

**版本说明：当前默认时间耦合已升级为四阶 MRI-GARK-ERK45a，见
[四阶方法、阶条件与运行验证](air_sea_mri4.md)。** 本页保留物理模型说明和第一版
一阶 `lagged` 方案的历史结果；下面的复现命令显式选择该旧方案。

本例的二维是 **水平 x–y 二维**：x 向东、y 向北，两个流体各自垂直平均。
海洋与大气占据相同的水平投影，通过风应力、显热、水汽交换耦合。没有 z 网格，
不解析竖直边界层涡旋；它不是“下方海洋、上方大气”的 x–z 截面模拟。

已实现可运行的 24 小时启动算例。图中的两个环流由求解动量方程产生，
不是指定解析 Double-Gyre 速度场。24 小时结果不是充分自旋的平衡环流。

## 运行

从仓库根目录执行；现有 Python 环境需要 JAX、NumPy、SciPy、gmpy2、matplotlib。
如需安装，使用 `python -m pip install -e 'jax[weak-ns]' matplotlib`。

```sh
PYTHONPATH=jax/src OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/bspf-mpl \
  python examples/pde/air_sea_double_gyre.py \
  --n 33 --hours 24 --dt-air 60 --dt-ocean 300 --window 600 \
  --method lagged --out build/air_sea_double_gyre
```

快速试跑可用 `--hours 1`；输出目录由 `--out` 指定。模拟时长必须是耦合窗口的整数倍，
窗口必须同时是大气步长和海洋步长的整数倍。

默认输出到 `build/air_sea_double_gyre/`：

- `coupled.png`：海洋流函数、纬向风、SST、比湿、热通量及平均温度演变。
- `budgets.png`：热量、水分和大气纬向动量收支残差。
- `state.npz`：最近输出时刻的模态系数和节点场；场数组采用 `(nx, ny)`。
- `snapshots.npz`：逐小时场快照；`history.json`：诊断和累计收支。
- `summary.json`：物理参数、步长、边界、方法限制及最终统计。

`state.npz` 和 `history.json` 是检查点资料；当前命令行没有自动重启选项。
重新运行相同输出目录会覆盖这些算例输出。默认输出网格包含两侧端点；
大气 x 周期端点代表同一物理位置，不能在积分时重复计数。守恒诊断使用求积点，
不使用画图节点的简单平均。

### 24 小时、48 帧 MP4

```sh
PYTHONPATH=jax/src OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/bspf-mpl \
  python examples/pde/air_sea_double_gyre.py --method lagged --hours 24 --output-hours 0.5 \
  --no-plot --out build/air_sea_double_gyre_48frames

PYTHONPATH=jax/src OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/bspf-mpl \
  python examples/pde/render_air_sea_mp4.py
```

视频取 `0.5, 1.0, …, 24.0 h` 的 48 个模拟状态，不包含初始零时刻，不做时间插值。
默认 8 fps、6 秒、1600×1000、H.264/yuv420p，输出为
`build/air_sea_double_gyre_48frames/air_sea_24h_48frames.mp4`。
六个面板显示海洋流函数和流速箭头、大气纬向风和风矢量、SST、大气温度、比湿、
向上显热与潜热通量。色标和箭头尺度在全片固定。`--fps` 可以修改播放速度。
同名 JSON 保存源快照索引、逐帧物理时刻、色标范围及 ffprobe 实测帧数。
渲染需要系统中可调用的 `ffmpeg` 和 `ffprobe`。

## 边界与空间离散

| 分量 | x 方向 | y 方向 | 离散 |
| --- | --- | --- | --- |
| 海洋速度 | 封闭无滑移 | 封闭无滑移 | 原 BSPF 空间的夹持流函数 Galerkin |
| 海洋 SST | 无扩散通量 | 无扩散通量 | 完整 BSPF 标量空间，自然 Neumann 条件 |
| 大气速度 | 周期 | 不可穿透、自由滑移 | x 实 Fourier、y 正弦流函数，另含平均纬向风 |
| 大气温度、比湿 | 周期 | 无法向通量 | x 实 Fourier、y 余弦 Galerkin |

大气不是 y 周期。南北端不相接，因此两层都能使用连续的 β 平面参数。
大气空间是与 BSPF 海洋共用求积的混合谱实现，不能称作所有方向均采用完整 BSPF。
所有非线性乘积在共同的高阶 Gauss 积分点计算，再分别投影到各自空间。
每轴默认约 456 个积分点（n=33/49/65 时）；积分点并非独立的解自由度。

海洋的强形式散度由流函数混合偏导相消；大气的平均纬向风也无散。
小散度残差只证明离散约束，不证明完整 PDE 误差小。

## 动力学与标量方程

长度 `L=1000 km`；内部空间坐标为 `ξ=x/L, η=y/L`，时间单位 s，速度单位 m/s。
导数显式乘 `1/L` 或 `1/L²`。物理流函数为 `L` 乘内部流函数，单位 m²/s。

两层的水平速度满足无散、旋转正压方程，压力通过无散试验函数消去：

\[
\partial_t\boldsymbol u_o+\boldsymbol u_o\cdot\nabla\boldsymbol u_o
+f\hat z\times\boldsymbol u_o
=-\nabla\pi_o+A_o\Delta\boldsymbol u_o-r_o\boldsymbol u_o
+\boldsymbol\tau/(\rho_oH_o),
\]

\[
\partial_t\boldsymbol u_a+\boldsymbol u_a\cdot\nabla\boldsymbol u_a
+f\hat z\times\boldsymbol u_a
=-\nabla\pi_a+A_a\Delta\boldsymbol u_a
+\gamma_u(\boldsymbol U_{target}-\boldsymbol u_a)
-\boldsymbol\tau/(\rho_ah_a).
\]

其中 `f=f0+β(y−L/2)`，默认 `f0=1e−4 s⁻¹`、`β=2e−11 m⁻¹s⁻¹`。
大气目标风为 `U_target=(-8 cos(2πy/L), 0) m/s`，恢复时间 6 小时。
初始大气等于目标风，海洋从静止开始；只施加 bulk 风应力，**不叠加第二套规定应力**。
该目标风产生南北反号的应力旋度，但二次阻力下应力分布不严格等于单一余弦。

海洋混合层温度、大气温度和比湿满足守恒平流扩散：

\[
D_oT_s=K_o\Delta T_s+(Q_{rad}-H_s-L_vE)/C_o,
\]

\[
D_aT_a=K_a\Delta T_a+H_s/C_a+\gamma_T(T_{target}-T_a),
\]

\[
D_aq_a=K_a\Delta q_a+E/M_a+\gamma_q(q_{target}-q_a).
\]

`C_o=ρ_o c_po h_m`，`C_a=ρ_a c_pa h_a`，`M_a=ρ_a h_a`。
默认海洋动力水深 `H_o=1000 m`，热力混合层 `h_m=50 m`，大气层厚 `h_a=1000 m`。
温度恢复时间 1 天，比湿恢复时间 2 天，`Q_rad=100 W/m²`。恢复目标固定为初始大气场。
这些外部源汇都进入诊断账本。

初始 `T_s=290+4 cos(πy/L)+0.5 cos(2πx/L) K`，`T_a=T_s−2 K`，
`q_a=0.75 q_sat(T_a)`。保存的温度模态是相对 `290 K` 的异常，输出字段为绝对温度。
默认海洋动量/标量扩散率为 2000/1000 m²/s，大气均为 20000 m²/s。
这是粗网格上较平滑的第一版参数，并未校准为真实盆地。

## 界面通量与记账

\[
\boldsymbol U_r=\boldsymbol u_a-\boldsymbol u_o,\quad
\boldsymbol\tau=\rho_a C_D|\boldsymbol U_r|\boldsymbol U_r,
\]

\[
H_s=\rho_a c_{pa}C_H|\boldsymbol U_r|(T_s-T_a),\quad
E=\rho_a C_E|\boldsymbol U_r|[0.98q_{sat}(T_s)-q_a].
\]

`C_D=0.0013`，`C_H=C_E=0.0012`。`q_sat` 使用液态水 Bolton 型饱和蒸气压近似，
常压 101325 Pa；0.98 表示简化海水盐度蒸气压修正。
首版把层平均大气量当作 bulk 代表值，未实现参考高度转换、Monin–Obukhov 稳定度
或 COARE。这是可控的概念验证闭合，不是经过验证的真实近海面通量产品。

正应力向海洋，正显热及水汽向大气。每个积分点只计算一次通量，再同时生成双方的
弱载荷；双方投影空间不同，但域积分一致。负 E 表示向海面的水汽回输。

热力预算变量为

\[
\mathcal H=C_o\langle T_s-T_{ref}\rangle
+C_a\langle T_a-T_{ref}\rangle+L_vM_a\langle q_a\rangle.
\]

界面显热与潜热在该预算中抵消。大气温度只接收显热；蒸发潜热通过比湿进入大气
焓，不能再给温度加一次 `L_vE`。热力预算未包括动能及黏性/阻力耗散回热。

海洋体积固定，蒸发失水只累计到诊断水库 `ΔW_o=−∫〈E〉dt`。
因此可以核验 `M_aΔ〈q_a〉+ΔW_o` 与外部水源相符，但不能声称已求解海洋淡水、
盐度和自由表面。动量方面，大气纬向均值满足应力与恢复力的独立预算；
封闭海洋还有壁面压力、黏性和底摩擦反力，不能要求两层总水平动量恒定。

## 快—慢推进

默认大气步长 60 s、海洋步长 300 s、耦合窗口 600 s：每窗口大气 10 步、海洋 2 步。

1. 冻结窗口起点的海流与 SST。
2. 大气使用 RK4 子步进；每个 RK 阶段重新计算相对风、显热和水汽通量。
3. 用完全相同的 RK 权重累计界面交换及外部源汇。
4. 海洋使用累计通量除以窗口长度得到的平均载荷，RK4 推进到相同物理时刻。

这套算法交换量守恒，**耦合时间精度是一阶**。内部 RK4 不会消除海洋状态滞后的误差。
没有实现界面隐式迭代、二阶预测校正或自适应步长；显式黏性、平流及交换仍有稳定性限制。
网格加密、扩散系数改变或混合层变浅时，必须重新检查时间步。

## 验证

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=jax/src OPENBLAS_NUM_THREADS=1 \
  python -m pytest -q jax/tests/test_air_sea.py

PYTHONPATH=jax/src OPENBLAS_NUM_THREADS=1 \
  python examples/pde/validate_air_sea_double_gyre.py
```

测试覆盖：通量方向与相对风不变性、阻力功、潜热不重复记账、非法步长拒绝；
独立随机系数的周期/自由滑移/无滑移边界；解析周期平流扩散 RHS；常数保持、
标量质量及耗散；旋转无黏流的能量功率；不同空间的通量积分一致性；
SST 和海流对通量的双向影响；多速率完整推进的热量、水分和大气纬向动量预算。

加密脚本用更快调整的 `H_o=50 m, h_m=5 m, h_a=300 m` 案例做 1 小时耦合窗口
600/300/150/75 s 对比，固定内部步长 15/75 s；另在 150 s 窗口下减半内部步长。
空间敏感性采用默认物理参数、33/49/65 节点，共同积分点比较 1 小时场。
脚本在不收敛或预算越界时失败，结果写到 `build/air_sea_validation/validation.json`。
这些短时对比不构成长期统计或平衡双环流的空间收敛证明。

本次实际验证：新增 16 项测试通过；连同现有流函数回归共 **21 项通过**。
窗口相邻差值的观测阶为 **1.0168、1.0083**，
与一阶耦合一致；固定 150 s 窗口、内部步长减半后的归一化场差约 `1.32e−14`，
远小于窗口滞后误差。该归一化组合范数将海流、风速、SST、大气温度、比湿的
RMS 差分别除以 `0.1 m/s, 8 m/s, 1 K, 1 K, 0.01 kg/kg` 后取欧氏范数。
33→49 节点的组合差为 `2.33e−4`，49→65 为 `1.20e−4`，呈下降趋势；
海流 RMS 差分别为 `1.49e−5`、`1.04e−5 m/s`。
这也说明 33 节点只适合启动演示，不能据此宣称已充分解析西边界层。

默认 33 节点、24 小时运行已得到：

| 指标 | 数值 |
| --- | ---: |
| 最大海流速度 | 0.0056004 m/s |
| 最大风速 | 6.66287 m/s |
| 海洋流函数范围 | 约 −704 至 +704 m²/s |
| 平均 SST 增量 | 0.016566 K |
| 平均大气温度增量 | 0.471599 K |
| 最大热力预算残差 | 2.17e−7 J/m² |
| 最大水分预算残差 | 2.98e−14 kg/m² |
| 最大大气纬向动量预算残差 | 5.21e−18 kg/(m·s) |

## 接口和后续扩展

```python
import jax
from bspf_jax.air_sea import (
    AirSeaConfig, plan_air_sea, plan_air_sea_stepper,
    initial_air_sea_state, air_sea_step,
)

jax.config.update("jax_enable_x64", True)
plan = plan_air_sea(n=33, config=AirSeaConfig())
stepper = plan_air_sea_stepper(dt_air=60, dt_ocean=300, window=600, method="lagged")
state = initial_air_sea_state(plan)
advance = jax.jit(lambda s: air_sea_step(plan, stepper, s))
state, budget = advance(state)
```

`budget.exchange` 是窗口累计冲量/热量/水分，而不是瞬时通量。
物理配置通过 `AirSeaConfig` 修改；需重新建 plan 以同步初始场和目标场。
公开入口在 `bspf_jax.air_sea`，不修改现有包顶层 API。

首版还没有温湿度浮力驱动、稳定度相关阻力、云凝结、分层海洋或垂向混合闭合。
温湿度会反馈到显热/蒸发，但不会通过浮力改变水平风；海流通过相对风反馈应力。
大气周期延拓覆盖封闭海盆，海洋两条东西岸的 SST 后续可能不匹配，因此周期大气
所受的热力强迫可能在接缝不光滑；首版没有人为平滑这个差异，长时/高分辨率研究
需要评估周期拼接敏感性或采用扩展的大气区域。

后续已实现四阶多速率耦合，见 [MRI-GARK4](air_sea_mri4.md)；局地隐式温湿交换、
独立海洋自旋及空间/时间自适应尚未实现。
进一步研究真实边界层时再增加垂向结构，而不是把当前二维场解读为三维湍流结果。

背景资料：
[MITgcm 风驱环流教程](https://mitgcm.readthedocs.io/en/latest/examples/barotropic_gyre/barotropic_gyre.html)、
[海气 Schwarz 耦合研究](https://gmd.copernicus.org/articles/14/2959/2021/)、
[NOAA COARE](https://github.com/NOAA-PSL/COARE-algorithm)。
本例没有依赖或调用这三个外部模型/算法。
