# JAX BSPF 非周期 Kelvin–Helmholtz 剪切层

本算例使用 `bspf_models.fluids.navier_stokes` 的二维不可压 NS 推进，以及
`bspf_models.elliptic.pressure` 的 BSPF masked pressure projection。两个方向均为有限区间，
不连接相对边界。FFT 只作为非周期 BSPF 导数中的计算分量，不代表周期边界条件。

## 模型与边界

区域为 `[-3,3] × [-1,1]`，基流为

```text
U0 = (tanh(y/δ), 0),  δ = 0.12,  ν = 0.002.
u_t = -(u·∇h)u + ν Δh u + f0 - σ(x,y)(u-U0) - ∇h p
div_h u = 0
u|boundary = U0|boundary.
```

四边都固定为基流速度：上下边为反向运动的切向速度，左右边同时包含相反方向的
进出流。左右净流量抵消。这里不是四边静止无滑移，也不是周期 KH benchmark。

固定体力 `f0 = -NS_raw(U0)` 抵消基流的离散黏性扩散及舍入量级的对流残差，
使 U0 成为离散稳态。对于连续 tanh 基流，其对应体力为 `(-ν U0'',0)`。
体力在初始化后不随数值解调整。

显式缓冲项使用 `σ = 80[(x/3)^16 + y^16]`，阻尼目标为固定基流。
中心区阻尼很弱，靠边界增强，抑制固定进出流边界激发的数值模态。
缓冲项被明确包含在压力投影之前。未采用周期回卷、额外频谱滤波或湍流模型。
一项无缓冲预试验出现了边界数值不稳定，未作为最终 KH 动画使用。

KH 初值形式参考 [NGSolve 的剪切层 benchmark](https://www.ngsolve.org/kh-benchmark/)
中的 tanh 基流加流函数扰动；本算例的有限域 Dirichlet 边界、维持体力与缓冲层
与该周期/自由滑移 benchmark 不同，因此不将它的参考数据当成本算例的精确解。

## 初始扰动与数值方法

```text
ψ = ε δ [1-(x/3)^2]^4 [1-y^2]^4 exp[-(y/(2δ))^2] cos(2πx/λ)
ε = 0.03, λ = 1.5
u(0) = U0 + Project(∂y ψ, -∂x ψ).
```

流函数及其一阶导数在边界消失。使用与 NS 相同的 BSPF D1 计算 curl，
再投影一次，使采样扰动满足离散散度及齐次边界增量约束。后续四阶 RK
每个阶段都投影加速度；由于边界增量严格为零，非零基流边界速度保持不变。

- 正式网格：160 × 112，均匀、包含端点；数组顺序 `(nx,ny,component)`。
- x64 JAX CPU，步长 0.002，积分至 T=6，共 3000 步。
- BSPF：q=9、degree=13、32 个样条基；Chebyshev M12/P16 端点估计。
- 梯度/散度共用相同 D1；黏性使用独立构造的 BSPF D2，而非 D1 平方。
- 缓存的一维 D1/D2 矩阵用于 NS 导数；压力使用一维张量分解，不形成全域矩阵。
- 阶段压力只求决定内部加速度的代表解，省略不影响该加速度的 wall completion。
  最终保存的压力重新求解并做 wall completion；墙面拟合残差单独记录。
- 所有阶段的压力返回状态必须成功。状态散度、有限性、边界残差与 CFL 均记录。

## 运行与文件

在仓库根目录运行：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
 MPLCONFIGDIR=/tmp/pybspf-mpl python scratch/run_kh_nonperiodic.py
```

绘图需要 matplotlib 和支持 libx264 的 ffmpeg。默认输出目录：
`examples/pde/results/kh_nonperiodic/`。

- `kh_nonperiodic.mp4`：301 个实际计算时刻，25 fps；涡量、横向速度、扰动增长与散度。
- `kh_nonperiodic.png`：末帧预览。
- `frames.npz`：可重新渲染的涡量与横向速度快照。
- `final_state.npz`：完整末态速度/压力、初值、基流、固定体力、缓冲系数。
- `diagnostics.json`：每个输出时刻的标量诊断。
- `summary.json`：参数、约束残差、增长量与验证结果。

计算、诊断及最终场为 float64；仅动画快照转为 float32 存储。
渲染时使用固定色标与双线性显示插值，没有生成额外的物理时间帧或修改数值场。
`--render-only` 可直接从保存的快照重建动画；`--no-video` 仅计算。
可通过 `--nx`、`--ny`、`--dt`、`--T`、`--out` 做独立复算。

## 验证

单元测试用解析多项式制造解检查含非零压力的 NS 加速度与涡量，检查稳态基流、
非零固定边界、散度以及 JIT 路径：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
 python -m pytest -c jax/pyproject.toml packages/models/tests/test_navier_stokes.py
```

正式运行另做两个对照：同一初值以半步长积分至 T=1，比较相对扰动的加权 L2
差异；去掉初始扰动后积分至 T=6，检查基流是否仍保持稳态。
这些验证用于检查实现与本算例的数值可信度，不等价于无界 KH 的增长率验证，
也不构成对更高 Reynolds 数或更细卷丝的网格收敛证明。

本次运行的结果：

| 指标 | 数值 |
| --- | ---: |
| 中心区域横向速度 RMS 增长倍数 | 78.31 |
| 全程最大状态散度 | 3.78e-12 |
| 全程最大边界速度误差 | 0 |
| 全程最大阶段 Schur 残差 | 4.56e-11 |
| 最大对流 CFL | 0.1084 |
| 半步长 T=1 相对扰动 L2 差异 | 1.43e-11 |
| 无扰动基流 T=6 最大变化 | 6.99e-14 |
| 128×80 与 160×112 的中心区末态涡量 L2 相对差异 | 1.30e-4 |
| 两网格中心区末态横向速度 L2 相对差异 | 7.63e-5 |

空间对比在 `|x|<2, |y|<0.5` 内进行，将细网格结果三次插值到粗网格节点。
粗网格对照使用相同的参数、步长和终止时刻，结果保存在 `coarse_frames.npz`
及 `coarse_final_state.npz`；这是一对网格的敏感性检查。可重新计算粗网格：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
 python scratch/run_kh_nonperiodic.py --nx 128 --ny 80 --no-video \
 --out examples/pde/results/kh_coarse
python scratch/compare_kh_grids.py \
 examples/pde/results/kh_coarse/frames.npz \
 examples/pde/results/kh_nonperiodic/frames.npz \
 examples/pde/results/kh_nonperiodic/spatial_sensitivity.json
```

末态压力 wall completion 的最大拟合残差为 8.51，并不为零；它不能作为
“所有墙面动量方程已逐点满足”的证据。这里直接施加的是固定速度边界，
速度增量和内部散度约束才是推进中严格检查的条件。该压力闭合的边界局限
仍然存在，并未被动画或上述速度收敛检查消除。

## 后续无缓冲稳定性诊断

在同一正式网格上的复算已将快速边界失稳定位到当前 BSPF 对流导数与强 Dirichlet 速度边界的组合，主要集中在入流端前几个内部节点。半步长不能消除它，无剪切匀速流也有同类正实部模态；详见[完整诊断、算子替换与收敛图](kh_boundary_instability.md)。因此本算例的缓冲层应视为针对当前边界离散的数值补偿，而不是 KH 物理所必需的项。

可选的能量稳定路径现已实现为 `closure="sbp84"`：使用内部八阶、边界四阶 SBP 导数、相容黏性及同一内积下的正交直接投影。它改变空间离散，不保留原 BSPF 谱精度；推导、无缓冲层对照及使用限制见[高阶能量稳定闭合](kh_energy_stable_closure.md)。
