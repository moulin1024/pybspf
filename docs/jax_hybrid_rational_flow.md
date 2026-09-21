# BSPF 体内表示与有理边界修正：非定常 NS 原型

已实现 `ImmersedFlowPlan(wall_method="rational")`。这是**固定几何的合成近似空间**，
不是每次求解后叠加一个稳态 Stokes 流场。
质量、黏性、对流、缓冲与出口项均作用于完整合成基函数。
本版仍是 NumPy/SciPy 稠密参考实现，尚未实现矩阵自由的快速体内/边界分块求解。

实现位置：

- `packages/models/src/bspf_models/fluids/rational_stokes.py`：可复用的齐次 Stokes 边界延拓。
- `packages/models/src/bspf_models/fluids/immersed_flow.py`：合成空间、完整算子及原 IMEX midpoint 推进。
- `examples/pde/hybrid_flow_mms.py`：非零体力 Stokes 与非定常、非线性 NS MMS。
- `packages/models/tests/test_hybrid_flow.py`：含有理时间项的 MMS、时间阶、缓冲、通量和流函数单值性。

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
共 16 项通过。随后完成原 Re=20 的 t=20 绕流重算，结果见下；尚未验证高 Re 稳定性。

这些结果支持固定几何下迁移到非定常 NS 的一致性，不等同于已经证明一般 NS 的
近谱精度。空间表示、体积分、时间误差及高 Re 稳定性需要分别控制。

### 原绕流重算：Re=20，t=20

结果位于 `build/immersed_flow/hybrid/channel/`。固定 73×33、2059 个未知量，
dt=0.02；积分因子 4，44445 个物理积分点。原出口缓冲层长度 2、强度 3 保持不变。
由带缓冲的 Stokes 初态推进完整 NS，保存 41 帧实际计算流场。

- 最终最大速度：1.33988269；速度变化率 L²：1.70755e-7。
- 独立孔壁最大速度：5.89732e-13；外部速度边界最大误差：1.79806e-10。
- 八个独立截面的最大相对通量误差：4.04643e-12。
- 缓冲层起点至出口，速度扰动截面 L² 从 0.172929 降至 0.00646226；
  涡量扰动从 0.722914 降至 0.0280002。出口附近涡量扰动并非严格单调下降。
- 构造耗时 309.6 s；推进及逐帧场求值耗时 1762.1 s，不含最终独立边界检查。

`render_hybrid_channel.py` 用相同显示点和色标比较三个方案，并在同一组 27846 个
内部点上计算六阶差分空间导数及二阶后向时间差分的涡量方程残差。
该区域排除壁面、孔边邻域和缓冲层，诊断没有使用求解器的 Galerkin 矩阵。

|方案|积分因子|采样涡量方程残差 RMS|最大绝对残差|
|---|---:|---:|---:|
|原 SVD|2.5|11.2189|96.5231|
|解析壁面因子|2.5|3.32357|54.7328|
|有理边界修正|4|0.479997|3.71579|

图中原来明显的孔前振荡及尾流截面波纹显著减弱，采样残差相对原 SVD 降低约 23.4 倍。
这不是相对精确 NS 解的误差，也受差分诊断误差影响；不能据此声称一般 NS 已达到近谱精度。
两组旧结果与新结果的积分设置不同，因此该对照衡量整体改进，不能单独分离空间与积分的贡献。

```sh
python examples/pde/immersed_channel_flow.py --wall-method rational --quadrature-factor 4 --out build/immersed_flow/hybrid/channel
MPLCONFIGDIR=/tmp/bspf-mpl python examples/pde/render_immersed_flow.py --out build/immersed_flow/hybrid/channel --movie
# 需要此前 dense/ 与 factor/ 对照结果
MPLCONFIGDIR=/tmp/bspf-mpl python examples/pde/render_hybrid_channel.py
```

主程序现已增加 `latest.npz` 原子覆盖及逐帧 `history.json`，后续运行可直接读取中间场。
本次进程启动时仍使用旧输出逻辑，故完整场在 t=20 才落盘。

### 与已有 AAA–lightning 参考的比较

`examples/pde/compare_hybrid_lightning.py` 使用已验证收敛的 120 阶参考和相同的
401×161 独立采样网格。已有参考是**无缓冲、无惯性的 Stokes 解**；当前计算是
**带缓冲的 Re=20 NS 解**，因此其场差包含模型差异及数值误差，不能作为 NS 精度指标。
物理区域 x≤3 上速度/涡量相对 L² 场差为 30.1032% / 67.2578%；全盒上为
25.3744% / 62.6581%。仅限制比较区域并不能消除下游缓冲对全局解的影响。

另作匹配方程的检查：同时去掉惯性与缓冲时，新方法的解就是固定 Stokes 提升
u_L=U_0-E(γU_0)，无需推进 BSPF 体内系数。该提升相对独立 AAA 参考的结果为：

|量|相对 L² 误差|最大绝对误差|
|---|---:|---:|
|速度|8.91548e-13|8.35826e-12|
|涡量|2.74777e-11|2.28203e-9|

最大差与 AAA 参考自身 96→120 阶的差异处于同一量级。
这一检查只验证齐次 Stokes 边界提升，不验证一般 NS 的体内解、非线性或时间误差。
评估实际 Re=20 绕流的数值误差仍需要具有同样惯性、缓冲与出口条件的独立 NS 参考。
结果及图在 `build/immersed_flow/hybrid/channel/aaa_comparison/`。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/bspf-mpl python examples/pde/compare_hybrid_lightning.py
```

## 使用

```python
import jax
jax.config.update("jax_enable_x64", True)
from bspf_models.fluids.immersed_flow import ImmersedFlowPlan

plan = ImmersedFlowPlan(nx=73, ny=33, wall_method="rational")
step = plan.stepper(0.02)
state = plan.stokes_state.copy()
state = step.step(state, 0.0)  # 完整 NS，默认缓冲层已启用
```

```sh
pip install -e '.[host,notebook,test]' -e './packages/models[precision,test]' -e './packages/sim[air-sea,test]'
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python examples/pde/hybrid_flow_mms.py
python examples/pde/hybrid_flow_mms.py --quadrature-factor 4 --out build/immersed_flow/hybrid/mms_q4
MPLCONFIGDIR=/tmp/bspf-mpl python examples/pde/render_hybrid_flow_mms.py
python examples/pde/immersed_channel_flow.py --wall-method rational --out build/immersed_flow/hybrid/channel
python -m pytest -q packages/models/tests/test_hybrid_flow.py packages/models/tests/test_immersed_flow.py packages/models/tests/test_cavity.py packages/models/tests/test_lightning_stokes_reference.py
```

`rational_options` 可以设置 degree、corner_poles、laurent、samples、rcond。
当前实现限定固定矩形加一个椭圆孔洞；移动几何时还需处理时间变化的基函数，
不能直接使用这里固定 W_j 的时间项公式。


## GPU evolution

`examples/pde/immersed_channel_flow.py --backend gpu --wall-method rational`
uses GPU-resident state, Galerkin matrices, volume convection, outlet backflow,
Cholesky factorizations, both IMEX midpoint solves, and diagnostic reductions.
The backend requires a GPU and does not silently fall back to CPU. Geometry,
MPFR basis construction, rational boundary assembly, and plot reconstruction
remain host preprocessing/postprocessing. GPU mode also accelerates dense volume
assembly; `--basis-workers 4` (the example default) parallelizes MPFR point rows
without changing precision. Use `--basis-workers 1` for serial setup. Snapshots are explicitly downloaded
at output times; the evolving state stays on the device.

Library use: `step = plan.stepper(dt, device=jax.devices("gpu")[0])`, then
`state = step.initial_state` and `state = step.step(state, time)`.
Optional forcing callbacks must be JAX-traceable and return reduced force loads.
`step.diagnostics(state)` returns device scalars; use `jax.device_get` to log them.

On the Raven CUDA 13.0.1 environment, pip cuBLAS 13.4.1.1 combined with the
system cuSolver caused even random-matrix GPU QR to fail. A process-local
preload of the matching system cuBLAS restored reduced and complete QR without
moving those factorizations to NumPy:

```bash
LD_PRELOAD=/mpcdf/soft/SLE_15/packages/x86_64/cuda/13.0.1/lib64/libcublas.so.13 \
JAX_PLATFORM_NAME=gpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
PYTHONPATH=/tmp/pybspf-gpu-deps \
python examples/pde/immersed_channel_flow.py --backend gpu \
  --wall-method rational --out build/immersed_flow/hybrid/channel
```

The temporary dependency path above contains `gmpy2`; normally install the
package's `rational-flow` extra instead. The library preload is specific to
this environment, not a portable requirement.

Regression coverage: `packages/models/tests/test_immersed_flow_gpu.py` compares five steps
and diagnostics with the host implementation, checks device placement, and
runs evolution/diagnostics under JAX's implicit-transfer guard.

Profiling measurements and reproduction: [GPU performance report](jax_gpu_flow_performance.md).


### Opt-in GPU double-precision basis evaluation

The reference basis evaluator remains 113-bit MPFR. To evaluate the basis,
its derivatives, and its sensitive transforms in GPU FP64 instead:

```bash
python examples/pde/immersed_channel_flow.py --backend gpu \
  --wall-method rational --basis-precision float64 --out build/immersed_flow/fp64
```

The corresponding library option is
`ImmersedFlowPlan(..., assembly_device=jax.devices("gpu")[0], basis_precision="float64")`.
FP64 evaluation is also used for subsequent field reconstruction and independent
boundary checks. It bypasses MPFR and the basis worker pool; `--basis-workers`
has no effect in this mode. Summary files record the selected precision and zero
active basis workers. The profiling driver accepts the same precision flag.

This option changes cancellation-sensitive arithmetic and can reduce accuracy;
it does not silently fall back to MPFR. The GPU evaluator uses float64 throughout,
a compact spline recurrence, and fixed batches to reuse compiled kernels.
Use `--basis-precision mpfr` (the default) for the original precision.


GPU assembly also runs the rational boundary SVD, obstacle-trace SVD, and volume
energy eigendecomposition on the selected device. This is automatic with
`--backend gpu` and works with either basis-precision setting. The sampled-wall
`svd` method also uses a GPU SVD. CPU/GPU rank thresholds are the same, and device
decomposition errors do not trigger a CPU fallback. Geometry still has host
stages; GPU time stepping is unchanged.


GPU volume setup also performs rational row/coefficient products, tensor
construction, correction mapping, and scaling on device. Unnormalized operators
stay there for Gram assembly. Rational recurrences and stream-row construction
still run on the host; the factored-wall geometry path remains unchanged.


### GPU rational basis construction and evaluation

GPU volume setup now evaluates the rational Arnoldi recurrences, their first
and second derivatives, and stream rows on the GPU. The public host evaluator
remains available for reference and output reconstruction. Warm rational
volume evaluation fell from 2.07 s to 0.15 s on the A100; cold complete setup
remains about 20.3 s because of JIT compilation.

The initial Arnoldi construction can also run on GPU:
`rational_options={"basis_construction": "gpu"}` with `assembly_device` set.
For the example use `--backend gpu --wall-method rational
--basis-precision float64 --rational-basis-construction gpu`.
This option preserves both modified Gram–Schmidt passes but is slower for the
current small recurrence tables (22.7 s complete cold setup), so CPU construction
remains the default. AAA poles and initial boundary matrix assembly remain on
CPU. GPU Arnoldi preserves velocity/wall/flux accuracy, with some additional
roundoff sensitivity in vorticity at lightning corners; see
[jax_gpu_flow_performance.md](jax_gpu_flow_performance.md) for measurements.
