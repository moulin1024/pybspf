# BSPF 体内表示与有理边界修正：非定常 NS 原型

已实现 `ImmersedFlowPlan(wall_method="rational")`。这是**固定几何的合成近似空间**，
不是每次求解后叠加一个稳态 Stokes 流场。
质量、黏性、对流、缓冲与出口项均作用于完整合成基函数。
本版仍是 NumPy/SciPy 稠密参考实现，尚未实现矩阵自由的快速体内/边界分块求解。

实现位置：

- `jax/src/bspf_jax/rational_stokes.py`：可复用的齐次 Stokes 边界延拓。
- `jax/src/bspf_jax/immersed_flow.py`：合成空间、完整算子及原 IMEX midpoint 推进。
- `examples/pde/hybrid_flow_mms.py`：非零体力 Stokes 与非定常、非线性 NS MMS。
- `jax/tests/test_hybrid_flow.py`：含有理时间项的 MMS、时间阶、缓冲、通量和流函数单值性。

## 为什么不是把 NS 当作稳态 Stokes 求解

令 B_j 为原 BSPF 无散速度基函数，满足入口及上下壁齐次速度条件，孔壁迹为 γB_j。
预先构造线性延拓 E：给定零净通量的孔壁速度，返回满足该速度的齐次 Stokes 场，
入口/上下壁速度为零，右端 Laplacian 牵引为零。定义

\[
 R_j=-E(\gamma B_j),\qquad W_j=B_j+R_j,
\]
\[
 u_L=U_0-E(\gamma U_0),\qquad
 u_h(t)=u_L+\sum_j a_j(t)W_j.
\]

E 的作用是构造满足边界条件的基函数。B_j 不满足齐次 Stokes，
合成流场也没有被限制为齐次 Stokes。固定几何时 W_j 不随时间变化，

\[
 \partial_tu_h=\sum_j\dot a_j(B_j+R_j).
\]

离散 NS 是

\[
 M\dot a+\nu K a+Sa+C(a)=F+B_{\rm out}(a)-S_L,
\]
\[
 M_{ij}=(W_j,W_i)_\Omega,\quad
 K_{ij}=(\nabla W_j,\nabla W_i)_\Omega,\quad
 C_i(a)=((u_h\cdot\nabla)u_h,W_i)_\Omega.
\]

例如 M 包含 BB、BR、RB、RR 四部分。代码通过完整速度/导数矩阵直接组装，
没有丢掉有理部分的时间导数或非线性交叉项。隐式步仍是原框架的
`M + dt*(nu*K+S)/2`；不会用一个稳态 Stokes 反算代替该隐式系统。

因此“用稳态 Stokes 构造固定基底”本身不妨碍解 NS；
“每步求完体内问题，再把一个稳态修正贴上去且不计其时间项”才是不一致的流程。

理想连续情形，T=I-Eγ 在满足齐次孔壁条件的速度上是恒等映射。
若 E 在所用迹范数下有界，则 T 不会从连续目标空间中排除一般 NS 解。
有限精度的边界延拓、积分和有限 BSPF 空间仍然产生误差，必须分别验证。
这里没有给出一般 NS 的谱收敛定理。

## 边界与耦合实现

矩形、偏心椭圆、抛物线入口、水平无滑移壁和原出流条件沿用之前算例。
有理延拓采用 Goursat 表示、AAA 孔内极点、Laurent 项及角点外部 lightning 极点，
数学构造参考 [Xue–Waters–Trefethen](https://doi.org/10.1137/23M1576876)。
AAA 使用 [SciPy 1.15 及之后的接口](https://docs.scipy.org/doc/scipy-1.15.0/reference/generated/scipy.interpolate.AAA.html)。

- 边界超定系统仅在几何构造时分解。
- SVD 保留分解形式计算响应：先投影 RHS，再除奇异值，最后乘右奇异向量。
  初试显式形成伪逆会引入约 1e-5 的消减误差；保留因子后独立孔壁检查回到约 1e-13。
- 原 BSPF 孔壁迹的 SVD 用于压缩响应映射，不用于强迫 BSPF 系数落入孔壁约束零空间。
  73×33 时保留 160 个迹方向；没有在每次时间步重新求边界最小二乘。
- 删除有理表示中造成净源/汇的对数列，使流函数绕孔单值。
  不固定孔壁流函数常数；绕孔所需的自由度仍由解确定。
- 孔内有理极点是辅助表示的一部分。物理求值仅定义于流体及其边界；
  `grid()` 在孔内返回 NaN，`evaluate()` 拒绝孔内点。
  BSPF 体内部分仍是原矩形表示，但完整合成解不再强制在孔内正则。

有理系数是 BSPF 系数的预计算线性函数，并未另加一套随时间独立推进的未知量。
本例保留 2059 个体内未知量，背景仍为 73×33。
速度空间在物理 H¹ 内积下归一化；原归一化前的能量矩阵条件数约 3.69e9，
边界 SVD 保留子空间条件数约 4.38e12。不能把归一化后的易求解系统
当作原始表示天然良态的证明；本例需要 SVD、尺度变换及独立边界残差检查。

### 一个解析积分恒等式

固定提升 u_L 是齐次 Stokes，出口牵引为零，测试速度 W_i 在所有速度壁上为零，
所以 Green 恒等式给出

\[
 \nu(\nabla u_L,\nabla W_i)_\Omega=0.
\]

代码使用该恒等式处理固定提升的黏性载荷，避免体积分误差污染已知齐次解；
同时报告直接积分得到的残差 `quadrature_diffusion_lift_norm`。
这只处理一个已知、固定提升。未知量的质量、黏性和对流矩阵仍包含完整有理场。
有限精度的边界误差意味着该恒等式在数值上也只近似成立，因此需独立壁面验证。

## 验证不能只用无体力 Stokes

无体力 Stokes 在此构造下由 u_L 给出，单测它相当于只测有理边界求解器。
本次使用非零体力和非定常 MMS：

\[
 \phi(x,y)=25t_x^2(1-t_x)^3(1-y^2)^2
 \{\cos(\pi x/3+0.4)\sin(\pi y/2+0.2)
 +0.3\sin(2\pi x/3-\pi y+0.1)\},\quad t_x=(x+1)/6,
\]
\[
 v_B=\operatorname{curl}\phi,\quad v=v_B-E_*(\gamma v_B),\quad
 u_*(t)=u_{L,*}+\sin(1.3t)v.
\]

E_* 使用更高阶的有理参考。体力由 φ 的解析自动微分、有理场解析导数及连续 NS
方程生成。记 s=sin(1.3t)，连续体力为

\[
 f=s'v+(u_*\cdot\nabla)u_*-\nu s\Delta v_B+\sigma(u_*-U_0).
\]

有理部分的黏性与其压力梯度相消；时间项及对流项都保留其完整贡献。
φ 在外部速度边界满足齐次条件，且出口二阶导数为零，故其出口牵引为零。
孔壁上的 v_B 非零，需要非平凡有理修正。

求解器只接收连续体力、几何和边界数据，没有收到精确 φ 系数、精确特解或
精确有理修正系数。质量投影只用于事后表示误差诊断；时间推进从 s(0)=0 开始。
稳态强迫 Stokes 则只给 `f=-nu*Laplacian(v_B)`，用于检验 BSPF 体内解与边界耦合。

## 当前实测与范围

73×33、Re=20、积分因子 2.5，结果在 `build/immersed_flow/hybrid/mms/`：

- 扰动速度的离散 L² 表示误差：6.229e-12。
- 非零体力 Stokes：独立网格速度相对 L² 误差 4.381e-7，涡量 6.789e-6。
- 非定常 NS 的连续 MMS 弱残差，M⁻¹ 对偶范数：1.929e-5。
- 故意漏掉有理时间项：同一残差变为 0.4535。
- t=0.4，dt=0.08/0.04/0.02：速度绝对 L² 误差
  5.794e-4 / 1.431e-4 / 3.558e-5，符合二阶时间收敛。
- 最终独立孔壁最大速度：7.223e-14。

积分因子 4 的固定网格对照在 `build/immersed_flow/hybrid/mms_q4/`；
它只增加物理积分点，不改变未知量或显示网格。

| 同一 73×33 空间 | 积分因子 2.5 | 积分因子 4 |
|---|---:|---:|
| 强迫 Stokes 速度相对 L² | 4.381e-7 | 5.284e-11 |
| 强迫 Stokes 涡量相对 L² | 6.789e-6 | 8.218e-10 |
| 连续 NS 弱残差 | 1.929e-5 | 2.366e-9 |
| 漏掉有理时间项的 NS 残差 | 0.4535 | 0.4535 |
| 构造时间 / s | 154.5 | 312.4 |

积分因子 4 的最终 NS 孔壁最大速度为 7.285e-14；时间误差依次为
5.7943e-4、1.4313e-4、3.5576e-5，仍由二阶时间离散主导。
这组对照确认体积分曾是空间误差瓶颈，不需要增加求解自由度。
高阶积分有实际计算成本：本机 dt=0.02 的 20 步耗时约 13.8 s（包括步内首次计算）。
结果图由 `examples/pde/render_hybrid_flow_mms.py` 生成；上排是强迫 Stokes 空间误差，
下排包括非定常 NS 时间收敛，二者不得混作 NS 最终时刻的空间误差。

另有 33×25 的多项式 MMS 回归，启用原缓冲层：连续弱残差 1.334e-8，
漏掉有理时间项后为 0.1167，时间收敛保持二阶。
并进行零体力 NS 短时推进的壁面/通量检查。与原绕流、cavity 和独立有理参考测试一起
共 16 项通过。尚未用本构造完成原 Re=20 的 t=20 长时绕流验收，也未验证高 Re 稳定性。

这些结果支持固定几何下迁移到非定常 NS 的一致性，不等同于已经证明一般 NS 的
近谱精度。空间表示、体积分、时间误差及高 Re 稳定性需要分别控制。

## 使用

```python
import jax
jax.config.update("jax_enable_x64", True)
from bspf_jax import ImmersedFlowPlan

plan = ImmersedFlowPlan(nx=73, ny=33, wall_method="rational")
step = plan.stepper(0.02)
state = plan.stokes_state.copy()
state = step.step(state, 0.0)  # 完整 NS，默认缓冲层已启用
```

```sh
pip install -e 'jax[rational-flow]'
export PYTHONPATH=jax/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python examples/pde/hybrid_flow_mms.py
python examples/pde/hybrid_flow_mms.py --quadrature-factor 4 --out build/immersed_flow/hybrid/mms_q4
MPLCONFIGDIR=/tmp/bspf-mpl python examples/pde/render_hybrid_flow_mms.py
python examples/pde/immersed_channel_flow.py --wall-method rational --out build/immersed_flow/hybrid/channel
python -m pytest -q jax/tests/test_hybrid_flow.py jax/tests/test_immersed_flow.py jax/tests/test_cavity.py jax/tests/test_lightning_stokes_reference.py
```

`rational_options` 可以设置 degree、corner_poles、laurent、samples、rcond。
当前实现限定固定矩形加一个椭圆孔洞；移动几何时还需处理时间变化的基函数，
不能直接使用这里固定 W_j 的时间项公式。
