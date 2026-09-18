# KH 的高阶能量稳定闭合（JAX，已否决的精度对照）

此 SBP 分支因改变 BSPF 空间离散、损失精度而被用户否决，不作为 BSPF 修复方案推荐。默认 BSPF 未切换到此分支；当前研究转向保留 BSPF 插值空间的一维弱形式对照。

## 实现范围

`plan_navier_stokes2d(pressure_plan, closure="sbp84")` 增加一条高阶、能量相容的 NS 路径，默认 `closure="bspf"` 保持原行为。

这不是把原 BSPF 入流前两行换成低阶差分，也不是保留 BSPF 谱精度的局部修补。新选项将空间导数换为 **SBP(8,4)：内部八阶、边界四阶**，同时匹配扩散、非线性对流和压力投影。保留的是已有压力方法的**消元＋张量分离直接求解思路**；新投影使用 SBP 算子，不能与原 BSPF 散度混用。均匀网格即可，不需要 Chebyshev 节点，每轴至少 24 点。

经典正定对角 SBP 权重来自 Mattsson & Nordström (2004), JCP 199, 503–540；[作者维护的算子库及系数来源](https://github.com/ranocha/SummationByPartsOperators.jl/blob/main/src/SBP_coefficients/MattssonNordstr%C3%B6m2004.jl)。本实现不照搬某一组边界模板：固定八阶内部模板与对角权重，通过四次多项式精确性约束，计算最小范数反对称边界块。其自由参数选择可能与上述库不同。

仅替换对流端点、保留原 BSPF 压力投影，不能推出整个 NS 的能量估计。将原 BSPF 矩阵直接反对称化也不够：本次预实验中，即使补上低阶多项式精确性，光滑正弦函数的导数误差仍不随加密下降，因此未采用这种修补。

## 能量恒等式

每轴构造正定对角矩阵 \(H\)，导数满足

\[
HD+D^T H=B,\qquad B=\operatorname{diag}(-1,0,\ldots,0,1).
\]

这要求边界与内部耦合匹配；在 KH 中上下半平面的入流方向相反，因此两个端点都需要相容闭合。仍通过投影输出零壁面速度增量，强施加固定 Dirichlet 速度；未添加缓冲层或 SAT 阻尼。

直接采用 \(D^2\) 虽然满足能量估计，但中心导数对 Nyquist 奇偶模态的耗散不足，初次 KH 末态出现明显网格条纹。最终实现采用相容扩散：

\[
L=D^2-H^{-1}R,\qquad R=\frac{1}{256h}K_5^TK_5\succeq0,
\]

其中 \(K_5\) 是未除以步长的五阶前向差分。修正消去四次及以下多项式，内部截断误差为 \(O(h^8)\)，边界为 \(O(h^3)\)；内部 Nyquist 模态的扩散符号为 \(-4/h^2\)。它随物理黏性 \(\nu\) 一起进入方程，不是缓冲层，也不是每步滤波。

对齐次边界的一维对流扩散方程：

\[
\dot w=-aDw+\nu Lw,\qquad w_0=w_{N-1}=0,
\]
\[
\frac{d}{dt}\frac12 w^THw=-\nu(Dw)^TH(Dw)-\nu w^TRw\le0.
\]

这个结论对 \(a\) 正负均成立，强于“所测试的特征值没有正实部”。相容扩散的内部精度为八阶，但端点邻域的二阶导数截断误差一般只有三阶，不能把整个 NS 称作全局八阶或谱收敛。

非线性对流采用分裂形式：

\[
C(u)=-\frac12\sum_{j=x,y}\big[\operatorname{diag}(u_j)D_ju+D_j(u_ju)\big].
\]

取 \(M=H_x\otimes H_y\)，齐次速度边界下有 \(\langle u,C(u)\rangle_M=0\)。对流离散的这一抵消不依赖离散乘积法则，也不要求舍入误差下散度严格为零。

新投影 \(P\) 是到“零壁面、全网格 SBP 散度为零”的速度子空间的 \(M\)-正交投影，满足

\[
P^2=P,\quad MP=P^TM,\quad \|Pf\|_M\le\|f\|_M.
\]

因此，当 \(Pu=u\)、齐次壁面且无外力时，半离散 NS 严格满足

\[
\frac{d}{dt}\frac12\|u\|_M^2
=-\nu\big(\|D_xu\|_M^2+\|D_yu\|_M^2\big)
-\nu u^T(R_x\otimes H_y+H_x\otimes R_y)u\le0.
\]

这是**半离散**恒等式：RK4 仍受时间步稳定限制。KH 使用非零固定剪切边界与维持基流的外力，其扰动能量可以从基流获得能量；不能要求 KH 能量单调下降。均匀基流的无散扰动则满足相同的耗散估计。

## 压力仍是零精修直接法

令每轴的加权散度矩阵为

\[
B_x=H_x^{1/2}D_x[:,I_x]H_{x,I}^{-1/2},\qquad I_x=1,\ldots,n_x-2.
\]

先将内部速度正交投影到两个端面散度约束的零空间：以 \(Z_x\) 为 \(B_x[\{0,-1\},:]\) 的正交零空间基，\(C_x=Z_xZ_x^T\)。然后剩余压力矩阵是

\[
S_x\otimes I+I\otimes S_y,\qquad
S_x=(B_x[I_x,:]Z_x)(B_x[I_x,:]Z_x)^T.
\]

初始化时，对一维矩阵做 SVD，得到正交特征变换和非负特征值；每次投影只做张量变换、除以特征值和、反变换及梯度校正。每轴两个压力零模态，共四个张量规范模态置零。用 SVD 显式表示这两个零模态，避免通过阈值猜测小特征值。

没有 Krylov、AMG、压力迭代或迭代精修，也没有构建二维全局矩阵。存储为 \(O(n_x^2+n_y^2+n_xn_y)\)，一次应用为 \(O(n_x^2n_y+n_xn_y^2)\)。目前该新路径尚未接入压缩变换。

`ns_project_velocity` 返回速度与约束诊断，不返回可直接解释为物理压力的场；这里的内部乘子只用于投影，不做壁面压力梯度补全。为了兼容现有构造接口，仍传入原 pressure plan，但新分支只使用其中的网格和掩码，原压力预计算目前存在冗余。

## 使用方式

```python
import jax
import bspf_jax as b
jax.config.update("jax_enable_x64", True)

# p 为现有的 PressurePoisson2DPlan
ns = b.plan_navier_stokes2d(p, viscosity=0.002, closure="sbp84")
u, base, initial_diag = b.kh_initial_velocity(ns, perturbation=0.03)
force = -b.ns_raw_rhs(ns, base)
u, diag = b.ns_rk4_step(ns, u, 0.002, force)  # 不传 sponge
residual = b.ns_divergence(ns, u)
```

选用该分支后，使用 `ns_divergence` 和 `ns_project_velocity`，不要用 `pressure_divergence(p, u)` 评价新空间离散的约束。`ns_vorticity` 自动使用所选导数。旧默认分支继续使用原 BSPF 压力投影及其历史精修设置。

## 验证与复现

单元测试覆盖：SBP 矩阵恒等式、边界四阶/内部八阶多项式精确性、光滑函数的边界收敛、正负流向的加权耗散、投影与独立二维稠密约束 SVD 的对照、幂等性、正交性、全网格散度和壁面约束、非线性动能恒等式，以及均匀基流线性扰动能量恒等式。另运行原 NS 的三个回归测试。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=jax/src \
  python -m pytest jax/tests/test_ns_energy_closure.py jax/tests/test_navier_stokes.py -q

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=jax/src \
  python scratch/diagnose_kh_stability.py --closure sbp84 \
  --nx 160 --ny 112 --mode evolve --T 6 --dt 0.002

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=jax/src \
  python scratch/diagnose_kh_stability.py --closure sbp84 \
  --nx 160 --ny 112 --mode evolve --T 6 --dt 0.001
```

新结果输出到 `build/kh_stability/sbp84`，与原 BSPF 对照分目录。KH 的区域、黏性、初始扰动公式和固定边界与原诊断一致；两种离散各自投影初值并计算维持基流的外力，因此初始离散场不是逐点完全相同。本次对照证明稳定性改善，不等价于证明两种算法在此网格上的物理精度相同。

## 本次数值结果

采用相容扩散的最终版本，原网格 `160×112`、`ν=0.002`、无任何 sponge：

| 项目 | 结果 |
|---|---:|
| SBP 恒等式最大矩阵误差 | `1.11e-16` |
| x 向一维对流扩散最大特征值实部 | `-1.5863`（原 BSPF 为 `+10.0670`） |
| x 向加权能量对称部最大特征值 | `-5.4831e-4` |
| `dt=0.002` 完成时间 | `T=6` |
| 末态最大速度 | `1.51049` |
| 全程最大散度 | `1.95e-11` |
| 全程壁面速度误差 | `0` |
| 所有阶段压力约束检查 | 通过 |
| 与 `dt=0.001` 末态扰动的相对差 | `1.21e-11` |
| 主要 KH 线性模态增长率 | `1.35854` |
| 该模态的边界区域能量占比 | `2.255%` |

对照原 BSPF 无 sponge 在 `T≈2.35` 达到最大速度 15 的停止阈值，新分支跨过原失稳时段，并保持显著 KH 增长。增长率不能直接当成无限域 KH 的精确色散关系；这里是有限域、有黏性、固定边界模型。

在 `64×48` 的均匀流控制问题中，所计算的六个主要物理特征值实部均为负，最大约 `-0.24568`，特征对残差均小于 `1e-7`。能量定理提供与网格无关的稳定性依据，这个小网格谱测试只是补充核验。谱诊断将不满足约束的补空间平移到 `-50`，以免大量非物理零特征值干扰 ARPACK；真实无散速度子空间上的算子不变。该平移仅存在于诊断代码中，不用于 NS 时间推进。

光滑函数 `exp(x)` 的一阶导数最大误差，在 40、80、160 点上分别为 `1.336e-5`、`8.482e-7`、`5.342e-8`，每次约下降 16 倍，符合边界四阶精度。

![无缓冲层 KH 对照](../build/kh_stability/sbp84/comparison.png)

### 仍需区分的边界误差

初次使用 `D@D` 扩散的结果保存在 `build/kh_stability/sbp84_d1_squared`，仅作诊断，不代表最终实现。其高频指标 `||Δ_x^5 u||/(32||u-base||)` 在 `T=6` 为 `0.348`；加入相容扩散后降至 `0.110`，但末态图的出流区域仍有网格振荡，不能声称已经完全消除所有边界误差。

强制左右两侧速度始终等于基流，会约束到达出流端的涡结构；本次结果提示晚期出流处理和空间分辨率仍是问题。以 `|U|≈1` 估计，粗网格的 `|U|h_x/ν≈18.9`，并未解析厚度约 `ν/|U|` 的对流扩散边界层。这个解释尚不等于完成了对每个出流振荡模态的因果验证。

能量稳定性保证不会凭空注入齐次问题的动能，并不保证单调性、无振荡或已达到空间收敛。后续若要高精度长时间 KH，需要单独处理出流条件或加密边界；若要保留 BSPF 谱精度，还需要进一步构造与 BSPF 相容的能量离散，这个有限阶分支并未解决该问题。
