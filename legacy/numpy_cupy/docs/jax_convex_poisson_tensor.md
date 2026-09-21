# 无矩阵凸域 Poisson 与两层粗空间修正

入口是 `TensorConvexPoissonPlan`，也可通过固定网格接口的 `backend="tensor"`
选择。**这是实验后端，当前不是原 SVD 后端的等精度替代品。** 默认固定网格
接口仍用 `backend="reference"`。未满足所选收敛标准时，新后端默认抛出
`PoissonIterationError`，不会把达到迭代上限的结果静默当成成功。

## 保持与改变的部分

保持原背景 BSPF 空间、MPFR 一维基函数装配、`domain.volume_rule(volume_order)`
曲边积分、均匀弧长边界采样及其 H^(3/2) 范数。没有换成矩形点乘区域掩码。
物理最小二乘矩阵 A 与 `ConvexPoissonPlan` 的未缩放矩阵一致。

新增显式正则化，求解

\[
\min_c\ \|Ac-b\|_2^2+\lambda^2\|c\|_{H^2(B)}^2.
\]

因此 λ>0 时目标函数与原来的截断 SVD 有区别；不能宣称两者解完全相同。
λ 控制的是辅助矩形上延拓的能量，不是额外施加的物理 Neumann 边界条件。
MMS 精确解及其内部导数不进入求解器。

## 细层的张量作用

曲边积分按相同 x 坐标分组，一次计算该截线上的 `Bx @ C`、`Hxx @ C`，
再与每个积分点的 y 因子收缩。伴随用分组求和返回对应系数，不组装二维
PDE 矩阵。边界用配对的一维因子和 FFT 加权，伴随使用对应的逆 FFT 归一化。

完整矩形上的 H2 Gram 作用为

\[
G(C)=C+KC+CK+JC+CJ+2KCK,
\]

其中 J 是实际二阶导数 Gram，不用 K² 替代。正则化因子 L 由六个张量块
组成：C、K^(1/2)C、CK^(1/2)、J^(1/2)C、CJ^(1/2)、sqrt(2)K^(1/2)CK^(1/2)。
满足 LᵀL=G（浮点舍入精度内），因此不需要细层二维 Cholesky。

一维导数 Gram 中仅舍入量级的负特征值截到零，明显负值会报错。

## 与 abstract multigrid 的关系

当前实现是两层粗空间修正/deflation，不是多层 V-cycle，也没有证明网格无关
收敛。细层可分离 H2 缩放为 P，增广算子为 B=[A;λL]P。

提供两种粗空间：

- `coarse_space="projected"`：单独构造较小 BSPF 空间，再用完整矩形上的
  L2 投影转移到细层。不假设不同 N 的 BSPF 空间严格嵌套。
- `coarse_space="spectral"`：从细层质量正交、按一阶刚度特征值排序的基中
  选取低模态，形成精确嵌套的代数粗空间。此时 `coarse_nodes` 表示每方向
  保留的模态数，不代表重新构造了该节点数的背景网格。

设转移后的粗空间在细层预条件坐标中的正交基为 Z，对 **细层算子的投影**
做小规模分解 BZ=UΣVᵀ。令 Vc=ZV，Q=I-UUᵀ，W=I-VcVcᵀ，则先求

\[
\min_y\|QBW y-Qb\|_2,
\]

再恢复

\[
a=Wy+V_c\Sigma^{-1}U^T(b-BWy),\qquad c=Pa.
\]

粗层与细层共享同一份物理积分，不另行粗采样。`coarse_rcond` 只决定哪些
方向进入粗修正；被排除的粗方向仍属于细层补空间，不等同于删除物理自由度。
粗层覆盖整个小测试空间时直接使用粗解，不在舍入误差构成的补空间上迭代。

`preconditioner="box"` 是默认张量缩放。另有 `"bounding_box"` 对照选项，
使用区域包围区间内的质量归一化和导数能量；其质量特征值使用正下限以保持
满秩，不删除近似空间。当前测试没有显示它比默认选项更好。

## 使用与停止标准

```python
import jax
import numpy as np
from bspf_jax import ConvexPoissonGridPlan
from bspf_jax.embedded_poisson import benchmark_domains

jax.config.update("jax_enable_x64", True)
x = np.linspace(-1.2, 1.2, 65)
plan = ConvexPoissonGridPlan(
    benchmark_domains()[0], x, x,
    backend="tensor", nodes=65, coarse_nodes=33,
    coarse_space="spectral", regularization=1e-11,
    coarse_rcond=1e-8, maxiter=2000,
    cache_dir="build/convex_poisson_tensor/cache",
)
# f/g 回调契约与参考求解器相同；失败时默认报错。
# result = plan.solve(f, g)
```

默认 `data_tolerance=None`，采用 LSMR 的正则化最小二乘停止判断。
可显式指定 `data_tolerance`，改为要求物理数据残差
`||Ac-b||/||b||` 小于目标。此时先检查粗解，若已达标就不做细层迭代；
否则将相对目标换算到粗修正后的右端项尺度，再迭代并重新检查实际物理残差。
诊断中的 `convergence_criterion`、`optimization_converged` 和 `data_target_met`
分别说明采用了哪种判断。达到物理残差目标不等于正则化目标已求到驻点。

这个离散残差也不是解误差的承诺：还要独立检查积分充分性和边界采样。
提高 N 时，应同时验证 `volume_order` 与 `boundary_count` 是否足够。

研究脚本可用 `solve(..., allow_unconverged=True)` 获取标记为失败的迭代结果，
用于诊断；固定网格接口不启用这个例外。

## 缓存与成本

一维 BSPF 装配有进程内缓存；给定 `cache_dir` 后，还缓存一维因子、几何
因子和粗层分解。键包含离散参数、控制点、相关实现和依赖版本。粗层缓存
同时保存它所依赖的预条件变换，避免重新分解时特征向量符号变化导致不一致。
缓存是原子写入的数值 NPZ，不反序列化 pickle。几何缓存格式变化需要更新
代码中的 `tensor-geometry-v1` 版本标记。

没有细层二维 PDE 矩阵或二维 H2 Cholesky，但**粗层仍有稠密分解和粗空间
投影的存储/运算成本**。固定粗层大小时可降低细层分解规模；若粗层也快速
增大，这部分仍会昂贵。曲边积分的因子缓存是 O(PN)，不是完整张量网格的
O(N²)；不能把这里的全部内存成本笼统写成 O(N²)。

## 复现

```bash
PYTHONPATH=jax/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python examples/pde/convex_poisson_tensor.py \
  --nodes 33 --coarse 17 --volume-order 12 --boundary-count 256 --reference

PYTHONPATH=jax/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m pytest -q jax/tests/test_convex_poisson_tensor.py
```

基准分别记录冷构造、热缓存构造、迭代耗时、收敛状态，以及独立 MMS 的值、
梯度、Laplacian 和移位边界误差。参考分解的增量计时复用同一份一维/体积分
因子，明确不包含共享装配成本，不把它误解为参考后端的全部首次构造时间。

测试覆盖与原曲边积分逐点相同、显式 Kronecker 矩阵对照、算子伴随、完整
H2 Gram、粗算子投影一致性、同正则化目标的独立稠密解、持久缓存、固定网格
接口及未收敛异常。

## 当前实测结果：尚未实现等精度降本

2026-09-18，本机 CPU、单线程 BLAS。N=33、体积分阶数12、边界256点，
投影粗层17、λ=1e-10、粗层截断1e-6、迭代上限2000：

| 方法 | kmax=4π 解误差 | kmax=12π 解误差 | 每个右端项求解时间 |
|---|---:|---:|---:|
| 单层迭代 | 9.67e-7 | 3.56e-4 | 约4.6–4.9秒 |
| 两层投影粗修正 | 2.45e-8 | 1.44e-4 | 约15.2秒 |
| 同离散参考 SVD | 2.37e-10 | 1.26e-5 | 约0.014秒（已分解） |

两种迭代在这些运行中都达到迭代上限，不能视为收敛解。参考 SVD 的正则化
方式也不同；表格比较实际求解策略，不用于单独归因正则化误差。

该次两层冷构造46.54秒，分为一维装配19.03秒、几何因子16.59秒、粗层
10.92秒。同一进程中的热缓存重建为0.0157秒。共享一维/几何因子后，参考
二维装配与分解的额外成本为4.20秒；不能把这一增量计时与46.54秒直接相比。

N=65、体积分阶数16、边界1024点、粗层33、λ=1e-11、粗层截断1e-8：

| 粗空间 | 高波数12π解误差 | 1200步后的相对物理数据残差 | 状态 |
|---|---:|---:|---|
| 投影 BSPF 粗空间 | 1.07e-6 | 3.00e-7 | 达到迭代上限 |
| 细层低特征模态粗空间 | 8.01e-7 | 3.34e-7 | 未达到指定的2e-8残差目标 |

两次冷构造约117–118秒，细层一维/几何装配约96–101秒，粗层分解约17–21秒；
1200步迭代又用约72–80秒。高波数精度仍未达到既有参考结果约2.3e-9的水平。
低波数投影粗空间在1200步后达到约9.1e-12解误差，但默认最小二乘停止条件
仍未满足；不会因此把该运行重新标记为收敛。

因此，当前得到的是保持离散、可检验的两层实验框架和有效缓存，而不是一个
已经证实能保持原高波数精度、同时降低全新配置总成本的生产后端。粗修正每步
需要稠密投影，近零方向也没有被普通低频粗空间充分捕获；继续增加层数本身
不能保证解决这两个问题。

## 固定算子的多右端项问题

参考 `ConvexPoissonPlan` 的 SVD 只在构造时计算一次；重复调用同一个 plan 的
`solve(f,g)` 会复用分解，源项和 Dirichlet 边界值变化不触发重新分解。
总成本应按 `T_setup + 次数 × T_solve` 比较，而不是只看首次设置。
上面的 N=33 实测中，参考分解后的求解约0.014秒，两层迭代约15秒且未收敛，
因此固定域、多次求解场景当前应优先复用参考后端。

背景网格固定并不足够：若真实自由边界移动，曲边积分和边界算子仍会改变。
此外，当前参考类每次重新构造都会重做 SVD，没有自动跨实例/跨进程共享完整
参考分解；应把 plan 构造放在时间步或右端项循环之外。
