# JAX 压缩压力变换：按层批处理与连续布局

2026-09-18。将原先按形状/秩分组的索引块应用，改成整层兄弟块的批量矩阵乘法。求解仍是固定次数的直接代数应用，默认0次迭代精修。没有放宽压缩容差或压力残差容差。

## 使用与兼容性

```python
compressed = b.compress_pressure_plan(
    dense_plan,
    tolerance=1e-12,
    leaf_size=16,
    protected_modes=8,
    layout="layered",  # 新默认；旧实现为 "grouped"
)
project = jax.jit(lambda model, raw: b.project_pressure2d(
    model, raw, completion=False, refinement_steps=0,
))
velocity, diagnostic = project(compressed, raw)
assert bool(diagnostic.converged)
```

工厂对应选项是 `transform_backend="dct_hodlr", compression_layout="layered"`。压力求解器的整体默认后端仍是 `dense`；只改变选择压缩后端后的默认内部布局。SciPy 仅在压缩预处理使用；求解全部使用 JAX。

## 改动

原实现每组块要 gather 输入、做小矩阵乘法，再 scatter-add 到输出。新布局在系数矩阵上补少量零，使同层块宽相同：

1. 叶块通过一次批量矩阵乘法应用。
2. 每层通过 reshape 和静态 reverse 交换相邻兄弟块输入。
3. 同层低秩因子采用该层所需的最大秩，批量做两次乘法；不划算的层改用稠密块。
4. 每层结果直接 reshape 后相加，没有块索引数组或 scatter-add。

补零只发生在系数矩阵的乘法内部，DCT、模态排列和压力离散仍使用原来的 m=N-2 个内部点。在 N=512 时，内部乘法从510补到512；并未更改物理网格。

保护模态的置零/覆盖也改为 concatenate，避免 scatter。模态排列仍需 gather，它位于 HODLR 分块乘法之外。本优化是按层批处理，并不意味着所有矩阵乘法融合成一个设备 kernel。

代价是每层共享最大秩以及少量补零，会增加因子存储和部分乘法。仍然保留 `layout="grouped"`，用同一流程对照。

## 验证

24项测试通过（压力、NS回归和新变换测试），包括：实数/复数、非二次幂长度67、矩形压力网格、JIT/vmap/反向导数、保护模态、零秩非对角块和单叶树。StableHLO 检查明确断言新 HODLR 块应用中不存在 `stablehlo.gather` 或 `stablehlo.scatter`。

## 复现

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/pybspf-mpl \
python3 scratch/benchmark_compressed_pressure.py

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 -m pytest -q packages/models/tests/test_compressed_pressure.py \
  packages/models/tests/test_pressure.py packages/models/tests/test_navier_stokes.py
```

新基准同时比较稠密、旧分组、新按层布局，输出到 `build/layered_pressure_benchmark/`。上一轮未经此次优化的历史结果保留在 `build/compressed_pressure_benchmark/`。

## 本机 CPU 结果

JAX 0.10.0，float64/complex128；均匀网格，Chebyshev端点16样本/12模态，q=9、样条数32、degree=13。预热后同步测量9次取中位数。计时时未并行运行本任务的数值测试。全部实现均关闭精修，完整投影不含壁面梯度补全。

| 网格 | 稠密完整投影 | 旧分组布局 | 新按层布局 | 相对旧布局耗时减少 |
|---|---:|---:|---:|---:|
| 128² | 3.61 ms | 9.01 ms | 6.97 ms | 22.6% |
| 256² | 13.20 ms | 22.25 ms | 22.02 ms | 1.0% |
| 512² | 75.72 ms | 106.28 ms | 96.73 ms | 9.0% |

按层布局在512²上减少约9%的完整投影时间，但仍慢于稠密直接法。256²上的约1%差异很小，不能据此宣称稳定加速。

| 网格 | 稠密张量逆 | 旧布局张量逆 | 新布局张量逆 |
|---|---:|---:|---:|
| 128² | 0.95 ms | 5.05 ms | 4.79 ms |
| 256² | 5.15 ms | 15.44 ms | 17.78 ms |
| 512² | 29.65 ms | 57.82 ms | 53.82 ms |

单独张量逆在256²上本次反而变慢，说明减少批次并不保证每个规模、每个内核都加速。512²的新布局张量逆约53.82 ms，旧布局57.82 ms。

| N | 旧→新分块批次（每轴正逆合计） | 旧→新变换存储字节数 | 投影场相对稠密版差异 |
|---|---:|---:|---:|
| 128 | 15 → 8 | 537264 → 542176 | 5.037e-13 |
| 256 | 25 → 10 | 1625120 → 1674720 | 1.514e-13 |
| 512 | 36 → 12 | 4743136 → 4922848 | 1.795e-13 |

存储统计包括因子、模态排列和保护模态，不是进程峰值内存。512²的新布局这部分比旧压缩版增加约3.8%，仍比原稠密变换少约40.9%。

所有规模、所有后端的随机向量投影，以及带壁面补全的平滑/随机压力测试均通过默认收敛判据。512²时，新布局平滑压力相对误差3.44e-10、随机压力相对误差5.44e-10；稠密基准分别为3.44e-10、5.43e-10。没有通过放宽容差换取速度。

512²完整投影首次调用（包含编译和执行）从旧布局约1.31 s降到新布局约0.57 s。它不是纯编译时间，单次数据不作为稳定性能保证。

另外以独立块应用微基准比较了在乘法前/后交换兄弟块，后交换未显示收益，因此保留乘法前交换。数据在 `layer_order.json`，复现脚本为 `scratch/benchmark_layer_order.py`（需要先运行特征变换分析生成NPZ）。

当前优化降低了分块索引和调度开销，但没有达到比稠密法更快的目标。GPU、三维和更大规模尚未测量。原始数据为 `build/layered_pressure_benchmark/results.json`，计时图为同目录的 `timings.png`。
