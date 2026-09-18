# DCT＋分块压缩直接应用：实现与实测

后续的默认压缩布局已改为按层批处理，见[新布局实测](jax_layered_pressure.md)。本页保留旧 `layout="grouped"` 的首轮历史结果；当前基准脚本会比较两种布局并输出到 `build/layered_pressure_benchmark/`。

2026-09-18。已实现 JAX 可选后端，真正应用分块因子，不在求解时重建稠密 V、Vi。结论：**精度可以保持、较大网格的变换存储减少，但本机 CPU 上仍比稠密直接法慢**。保留为实验选项，不替换默认后端。

## 使用

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import bspf_jax as b

x = jnp.linspace(0, 1, 256)
plan = b.plan_pressure_poisson2d(
    x, x,
    endpoint_method="chebyshev", baseline_points=16, chebyshev_modes=12,
    transform_backend="dct_hodlr",
    compression_tolerance=1e-12,
    compression_leaf_size=16,
    protected_modes=8,
    compression_layout="grouped",  # 本页历史布局
)
project = jax.jit(lambda model, raw: b.project_pressure2d(
    model, raw, completion=False, refinement_steps=0,
))
# velocity, diagnostic = project(plan, raw)
# assert bool(diagnostic.converged)
```

也可以 `compressed = b.compress_pressure_plan(dense_plan)`，复用已有稠密计划的几何和特征分解。安装预处理依赖：`pip install -e 'jax[compression]'`。

压缩后端默认0次精修；稠密后端保留原有2次精修默认值。为公平比较，本报告双方都显式使用0次。没有外层迭代、没有自动回退、没有放宽收敛容差。压缩具有指定截断误差，因此属于近似直接法。

## 实现

- DCT-II 系数空间中进行最大重叠模态匹配，然后递归分解非对角块。
- 相同形状、相同秩的块批量相乘。秩向上取4的倍数，降低小算子数量；不会削弱压缩精度。
- 稠密叶块上限16；低秩表示比原块更大时保留原块。压缩计划不保留完整 V、Vi。
- 固定代数投影精确保留前8个按特征值模排序的模态，包括两个零模态；这不是迭代精修。
- SciPy/NumPy 仅用于可选的主机压缩预处理；求解时的 DCT、批量矩阵乘法、gather/scatter 和模态修正均使用 JAX。
- 当前 Chebyshev 算子有真正的复特征值和特征向量，不能简单丢弃虚部。实现支持复数变换，通过分别对实部和虚部做 DCT；只有严格为实数的特征系统才使用实数因子。
- 本次仍有稠密特征分解和 SVD 的设置成本，没有实现快速设置算法。

## CPU 实测

JAX 0.10.0，CpuDevice(0)，float64/complex128。均匀 [0,1] 网格，q=9、样条数32、degree=13、Chebyshev端点16样本/12模态，压缩容差1e-12，保护8模态。预热编译后同步计时9次，取中位数；测试进程外未同时运行本任务的数值测试。环境设置 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`，不声称这能把 XLA 的全部线程池限制为单线程。

| 网格 | 稠密张量直接逆 | 压缩张量直接逆 | 稠密完整投影 | 压缩完整投影 |
|---|---:|---:|---:|---:|
| 128² | 1.44 ms | 5.43 ms | 3.96 ms | 8.76 ms |
| 256² | 5.21 ms | 15.34 ms | 12.11 ms | 22.54 ms |
| 512² | 24.66 ms | 59.49 ms | 75.23 ms | 109.96 ms |

张量直接逆包含端点消元/恢复；完整投影还包含散度、梯度和残差诊断，使用 `completion=False`。本次没有GPU或三维性能测量。

| N | 因子标量数／两套原变换标量数 | 含索引和保护模态的变换字节数／原变换字节数 |
|---|---:|---:|
| 128 | 95.8% | 105.8% |
| 256 | 73.4% | 78.7% |
| 512 | 54.1% | 57.0% |

这里的字节数只比较一条坐标轴的 V 和 Vi 表示，不是整个压力计划或进程峰值内存。N=512 的这部分存储节省约43%。实际因子数与前期SVD分析略有不同，因为现在向上取整秩以批处理，并保留8个敏感模态。

## 精度

完整压力投影使用随机原始向量。另用平滑已知压力和随机已知压力构造 `div(Q grad p)`，在壁面梯度补全后检验压力误差。所有规模、两种后端的这些测试均通过默认 `converged` 检查。

| N | 压缩投影速度场相对稠密版差异 | 压缩版平滑压力相对误差 | 压缩版随机压力相对误差 |
|---|---:|---:|---:|
| 128 | 5.89e-13 | 2.99e-11 | 3.45e-11 |
| 256 | 2.56e-13 | 1.96e-10 | 1.17e-10 |
| 512 | 2.04e-13 | 3.44e-10 | 5.43e-10 |

N=512 的稠密版平滑/随机压力误差分别为3.44e-10、5.42e-10，说明这里压缩新增误差很小。误差表为已测右端的结果，不是任意数据保证。生产使用仍须检查 `converged`。

21项相关测试通过：新压缩变换测试，以及现有压力和NS回归测试。覆盖实际低秩因子、复数、非二次幂长度、矩形网格、JIT、vmap、输入反向导数、保护模态、壁面补全、不兼容右端和选项校验。

## 为什么暂未加速

因子总量减少不等于运行时间减少。本实现每条轴的正逆变换在 N=512 时共有36组块运算，并额外执行 DCT、索引读写和保护模态乘法。它们可能抵消了稠密矩阵乘法减少的工作；本报告没有通过 profiler 定量分摊各项开销，因此不把原因归结到单一算子。

下一步有针对性的优化是按树层重排/融合，降低 gather/scatter 和小矩阵调度开销，再测实际速度；不应仅凭减小存储或放宽压缩容差宣称加速。当前保持稠密后端为默认。

## 复现

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=jax/src \
MPLCONFIGDIR=/tmp/pybspf-mpl \
python3 scratch/benchmark_compressed_pressure.py

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=jax/src \
python3 -m pytest -q jax/tests/test_compressed_pressure.py \
  jax/tests/test_pressure.py jax/tests/test_navier_stokes.py
```

原始数据：`build/compressed_pressure_benchmark/results.json`。图：`build/compressed_pressure_benchmark/timings.png`。
