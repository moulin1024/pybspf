# 固定算子、多右端项的 BSPF GS 快速响应

此路径面向固定光滑边界、R>0 的后续剖面迭代：一次构造，反复从源项和边界磁通得到固定点上的磁通。不包含非线性迭代本身，也不加速初始稠密 SVD。

设加权 GS/边界算子经 H² 缩放后为 A=UΣVᵀ，H² 上三角根为 C，固定输出点的基函数矩阵为 E。原路径每次计算

    b → a = V Σ⁻¹ Uᵀ b → c = C⁻¹ a → ψ = E c。

快速路径一次计算 T=E C⁻¹ V，每步仅计算

    q = Σ⁻¹ Uᵀ b，ψ = T q。

不改变求积、BSPF 空间、边界范数或保留的奇异值；不引入额外低秩截断。Σ⁻¹ 保留在每次求解中，避免先形成完整稠密伪逆。训练残差仍计算为 b−UUᵀb。它是原离散求解器的代数重组，浮点运算顺序会有微小差异。

```python
import jax
jax.config.update("jax_enable_x64", True)
from bspf_jax import FixedBoundaryGSPlan, SolovevEquilibrium, SolovevFluxDomain

eq = SolovevEquilibrium(logarithmic=0.08)
plan = FixedBoundaryGSPlan(SolovevFluxDomain(eq), nodes=25,
                           volume_order=16, boundary_count=512)
fast = plan.compile_response()  # 默认在源项求积点输出磁通
result = fast.solve(eq.source, boundary_flux=0.0)
psi = result.flux
# 后续剖面更新在 fast.points（物理 R,Z）上构造新的 source，再调用 fast.solve。
# 若需任意点输出，可按需恢复系数：
solution = result.as_solution()
```

`plan.compile_response(points, derivatives=True)` 额外编译两个一阶和三个二阶导数，返回的 `gradient`、`hessian`、`delta_star` 和 `magnetic_field(F)` 可用于独立检验。完整导数响应约需磁通响应六倍内存；一般剖面更新只需默认的磁通响应。

默认求积点直接复用组装阶段的基函数，避免重复 MPFR 求值。新的自定义输出点在编译时求值一次。修改几何、离散空间或算子后必须重建 plan；改变源项和 Dirichlet 数据只需再次 solve。

## 复现与比较范围

```sh
PYTHONPATH=jax/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python examples/pde/benchmark_gs_response.py
PYTHONPATH=jax/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m pytest jax/tests/test_grad_shafranov.py -q
```

计时参考路径也缓存全部基函数，只计算磁通，不重复 MPFR、不计算多余导数。两条路径均包括源项/边界数据处理和训练残差。四组不同平滑源项及非零边界数据轮换，预热后交替计时，以中位数报告。结果保存至 `build/gs_response/results.json`。

额外响应存储为 8Mr 字节（M 个输出点、r 个保留模态），每步主成本 O((L+M)r)，L 为体积与边界方程数。初始 SVD、响应编译和存储成本仍存在；这不是近线性复杂度求解器。回本次数只比较新增编译时间与每步节省时间，两者共有的初始 GS 分解不计入增量回本。

验证覆盖零源项、非零边界、振荡源项、多右端项复用、系数恢复，以及独立采样点上的解析 Solov’ev 磁通、导数、GS 算子和磁场。`test_grad_shafranov.py` 共 10 项测试通过。

## 本机单线程实测

Q=16（4096 个体积点）、512 个边界点；80 次重复的中位数。

| N | 缓存参考 / ms | 编译响应 / ms | 加速 | 响应编译 / s | 响应 / MiB | 增量回本次数 |
|---|---:|---:|---:|---:|---:|---:|
| 25 | 1.453 | 1.289 | 1.13× | 0.060 | 14.16 | 367 |
| 33 | 2.443 | 1.879 | 1.30× | 0.166 | 21.22 | 295 |
| 49 | 6.356 | 3.070 | 2.07× | 0.682 | 38.06 | 207 |

共同的 GS 初始构造耗时依次为 24.11 s, 21.19 s, 35.61 s（顺序运行，含进程内缓存影响）。新增编译不会消除这些成本。

四组右端项的快速/参考最大磁通差异依次为 4.48e-14, 5.99e-15, 2.11e-15；解析 Solov’ev 磁通最大误差均小于 6e-15。

小规模仅有温和收益，几十步迭代未必收回编译成本；N=49 约需 208 次重复求解回本。多次平衡求解、参数扫描或更长迭代可复用同一响应。该表仅代表此几何、分辨率和单线程配置，不承诺普遍加速比。
