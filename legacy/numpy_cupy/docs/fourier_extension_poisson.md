# Fourier extension Algorithm 1 与快速 Poisson 求解器

实现：`jax/src/bspf_jax/fourier_extension.py`、`fourier_poisson.py`。
这是 CPU NumPy/SciPy 实现，位于已有曲边 Poisson 实验包中；不提供 JAX JIT/GPU 路径。
入口是 `FourierPoissonPlan` 和 `examples/pde/fourier_extension_poisson.py`。
原 `compare_poisson_bases.py --family fourier` 仍用于共同稠密 PDE 框架的空间对照，
不调用本求解器，避免将两类实验混为一谈。

## 与论文一致的部分

复现 Matthysen–Huybrechs，*Function approximation on arbitrary domains using
Fourier extension frames*，本地 `1706.04848v1.md` 第 2.2 节 Algorithm 1：

1. 在周期盒 `[-T,T)^2` 上取 `m×m` 均匀网格，只保留物理域内的点。
2. 每方向取奇数 `n` 个连续 Fourier 模态，构成 `K=n²` 个系数。
3. `A` 是归一化二维 DFT 的子矩阵：`A c = sum c_k exp(i k·x)/m`。
   输入物理值相应缩放为 `b=f/m`。`A`、`A*` 通过补零、FFT、限制实现。
4. 令 `P=A A*−I`，随机低秩分解 `P A`，截断小奇异值后求 `P A y=P b`。
5. 用 `c=y+A*(b−A y)` 补全解。

代码不构造完整 `A`，不对完整采样矩阵做 SVD，也不在遇到困难时偷偷退回
完整 SVD。低秩分解仍然需要稠密的压缩矩阵和 SVD，这是论文算法的组成部分。
随机 range finder 采用分块采样、两次投影及归一化后的再正交化；独立 Gaussian
探针监测未覆盖的范围。`max_rank` 不够时明确报错。这是随机残差估计，非确定性证书。

QR/SVD 使用实张量 cos/sin 坐标；它与复指数坐标通过酉变换相连，保持奇异值和
算法不变。公开的 `forward`、`adjoint` 和返回的 Fourier 系数仍采用复指数坐标。
复数右端项得到支持；实数据的求值返回实数，消除浮点相消引入的虚部。

`cutoff` 是 `PA` 的**绝对**奇异值阈值，默认 `1e-12`，不是对 `A` 或 PDE 矩阵的
相对阈值。`PA` 的奇异值为 `σ(1−σ²)`，所以快速结果不必与对 `A` 做 TSVD 得到的
系数逐项相同。应比较拟合函数和残差，不能把不稳定的域外系数当成一致性标准。

## 新增的 Poisson 路线

这部分不是论文中的原算法，而是使用它构造的二维 Dirichlet 求解器：

```
−Δu=f in Ω,  u=g on Γ
f 在 Ω 内的样本
  → Algorithm 1 Fourier 延拓 F
  → 非零模态逐项除以 |k|²，得到周期特解
  → 加上 −F₀(x²+y²)/4，处理非零均值
  → 根据 g−u_particular|Γ 求调和修正
  → u=u_particular+h
```

**不能丢掉零模。** 物理域的 Dirichlet 问题没有右端项零均值要求。
上述二次多项式满足 `−Δ[−F₀(x²+y²)/4]=F₀`，避免把非零均值误当作不兼容。

调和修正使用 MFS（method of fundamental solutions）：

```
h(x) = a₀ + Σ_j a_j log|x−s_j|/(2π).
```

源点 `s_j` 从真实样条边界沿外法向移到域外；常数列显式保留。边界使用等弧长
过采样、列缩放、因子形式的 TSVD。没有显式乘出病态伪逆，避免巨量矩阵元素
与光滑边界数据相乘时的消去误差。域内值、梯度和 Hessian 均从同一展开解析求值。

第一版 `FourierPoissonPlan` 限于现有的**光滑凸、逆时针闭合三次 SplineDomain**；
构造时执行曲率、正则性和总转角检查。凸性保证法向偏移源点在域外。
FE 本身只接受布尔 mask，不限于凸域；这不意味着边界 MFS 部分已支持凹域、孔洞或角点。
MFS 的精度依赖源点数、源距离和边界截断，需独立加密检查。

## 使用

```python
import numpy as np
from bspf_jax import FourierPoissonPlan
from bspf_jax.embedded_poisson import benchmark_domains

domain = benchmark_domains()[0]
plan = FourierPoissonPlan(
    domain, modes=49, grid_size=196,
    source_count=192, boundary_count=384, source_distance=0.25,
)
solution = plan.solve(
    lambda p: -2*np.exp(p[:, 0]+p[:, 1]),
    lambda p: np.exp(p[:, 0]+p[:, 1]),
)
u, grad, laplacian = solution.evaluate(np.array([[0.1, 0.2]]))
hessian = solution.hessian(np.array([[0.1, 0.2]]))
residuals = solution.validate(
    lambda p: -2*np.exp(p[:, 0]+p[:, 1]),
    lambda p: np.exp(p[:, 0]+p[:, 1]),
)
# 同一 plan 可以重复求解不同 f、g，不必重新做分解。
```

`solve` 也接受在 `plan.extension.points` 和 `plan.boundary` 上的数组，或标量常数。
输入回调只在物理域内/真实边界调用，不需要域外 forcing 或域内精确解。
所有点数组形状为 `(count,2)`，坐标顺序 x、y。FFT 网格使用 `indexing='ij'`。

`evaluate` 的第三项是正号 `Δu`，因此应与 `−f` 比较。
`validate` 使用另一套真实几何求积与偏移边界点，报告 forcing 的 L2 残差和边界
H^(3/2) 残差。这些是离散诊断，不是严格连续误差上界。

## 速度、存储和适用范围

记 `K=n²`，域内样本数 `M`，过渡区秩 `r`。固定过采样时：

- FE 构造主要为 `O((M+K)r²)` 的低秩分解，另有 FFT 工作；二维正常边界下论文
  给出 `r=O(n log n)`，对应约 `O(K² log²K)`，不是构造阶段 `O(K log K)`。
- 已有分解后，一个 RHS 需要 `O((M+K)r)` 因子乘法和若干 FFT。
- 长期保存的 FE 因子为 `O((M+K)r)`；报告的 `factor_bytes` 不含构造临时矩阵，
  不能当作峰值内存。
- MFS 是仅边界上的稠密 TSVD。此实现没有 FMM。
- 非均匀点上的 Fourier 求值采用分块直接张量求和，没有 NUFFT；输出求值时间
  独立报告，不包含在重复 RHS 计时中。边界处的特解求值包含在 solve 计时中。

因此“快速”指论文的结构化 FE 路线与避免全体积 Poisson 稠密矩阵，**不承诺所有
规模的首次构造均比直接 SVD 快**。在小规模/很紧阈值时，过渡区秩可能占绝大多数。

默认 `grid_size=4*modes`，即每方向网格过采样 4 倍；这不等于论文的
`N_Ω/N_Λ=4`，后者取决于物理域面积。当前区域下约为 5.4 个域内样本/系数。
先前每方向 2 倍的试验只有约 1.35 个域内样本/系数，导致曲边附近的独立误差远大于
训练残差。不能仅据训练点小残差认定延拓、Poisson 或二阶导数已准确。

## 复现与验证

```sh
PYTHONPATH=jax/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python \
  -m pytest -q jax/tests/test_fourier_extension_poisson.py \
  jax/tests/test_poisson_basis_comparison.py

PYTHONPATH=jax/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python \
  examples/pde/fourier_extension_poisson.py \
  --out build/fourier_extension_poisson_final
```

基准包含 `n=17,33,49,65`，指数、多项式、高斯、有理函数、两组随机波。
所有测试使用共同的 8159 个全域偏移网格点和独立偏移边界点，报告值、梯度、H2、
Laplacian 误差。MMS 的域内精确 jets 只用于误差评估，正式 PDE 求解仅输入 f、g。
分解在同一 n 的所有 RHS 之间复用。

脚本保存 JSON、系数 NPZ、Markdown 汇总、环境版本和实现源码 SHA256。
另有同一采样问题的直接 TSVD 对照；直接法也使用实正交基，并提前保存截断后的
因子，避免重复 RHS 中不必要的列拷贝。对照用于核验 FE 实现与成本，不是原有
BSPF/Fourier PDE 离散的等精度性能排名。低分辨率失败案例也原样保留。

测试覆盖独立 DFT/伴随、直接 TSVD、复数模态导数、全盒极限、秩预算失败、非周期
Poisson、非零均值、纯调和边界、独立边界点、解析 MFS 导数、残差验证与 plan 复用。

## 本机实测（2026-09-19）

最终结果位于 `build/fourier_extension_poisson_final/summary.md`，图为同目录
`convergence.png`。最终实现源码哈希与运行记录逐项一致；新测试与原有空间对照测试
合计 **20 项通过**，四档分辨率、六类函数共 24 个案例全部完成。

每方向 65 模态、260×260 背景网格、23003 个域内采样、192 个 MFS 源点下：

| 精确解 | 全域独立相对 L2 | 全域独立相对 H2 |
|---|---:|---:|
| 指数 | 2.84e-15 | 2.74e-11 |
| 八次多项式 | 6.79e-15 | 3.24e-11 |
| 局部高斯 | 2.62e-15 | 6.21e-15 |
| 有理函数 | 4.94e-11 | 6.95e-8 |
| 低频随机波 | 2.81e-14 | 1.41e-11 |
| 高频随机波 | 4.14e-13 | 3.76e-11 |

首次构造约 **100.3 s**，已有分解的单 RHS 约 **16.7–17.7 ms**（不含输入采样与
输出求值），FE 长期因子约 **504 MiB**，不是峰值内存。不能把这组结果推广为
所有光滑函数均可达到机器精度，有理函数仍明显受分辨率限制。

同一 FE 问题在 n=49 时，Algorithm 1 构造约 24.0 s，直接 TSVD 约 15.2 s；
重复 RHS 分别约 6.46 ms、5.40 ms，独立点相对差异约 7.83e-14。
因此本次小中规模实验**尚未证明 FE 阶段比优化后的直接 TSVD 更快**。
新 Poisson 路线能够复用因子进行毫秒级求解；它与旧稠密 PDE 求解器的等精度、
同环境现场性能对照仍是另一项实验，不能用不同时间采集的历史计时宣称固定倍数提速。
