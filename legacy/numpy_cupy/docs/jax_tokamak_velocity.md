# 接入原 BSPF 流函数 NS 框架的线性 MHD

入口为 `examples/pde/tokamak_velocity_mhd.py`，主实现为
`jax/src/bspf_jax/tokamak_velocity.py`。本次继续保留已约定的线性、轴对称、
不可压物理范围，推进独立的速度和磁扰动；原位移模型作为独立对照保留。

## 实际复用的代码路径

| 原有能力 | 新路径中的调用 |
|---|---|
| 流函数 NS 的 BSPF 值、一阶/二阶导数及张量旋度 | `_stream_line`、`tensor_product`、`curl_from_gradient`，同一组函数也供原 `stream_ns_velocity/vorticity` 使用 |
| 流函数 NS 的广义特征变换和张量 Poisson/惯性反解 | 公共 `tensor_elliptic_solve` 同时供 `stream_ns_inertia_solve`、GS 平衡和曲坐标真空预条件器使用 |
| NS 四阶速度阶段推进 | 原 `ns_rk4_step`、`stream_ns_rk4_step` 和新 MHD 都调用 `rk4_stages` |
| 流函数 NS 的力载荷到加速度接口 | 新 MHD 的磁力载荷经过 `stream_ns_inertia_solve`；等离子体限制域已作质量正交化，因此此处惯性矩阵为单位阵 |

共享核位于 `_flow_kernels.py`，已接回旧 NS 代码并运行回归测试。
没有把固定矩形域的速度—压力投影器直接用于曲边界：本路径继承流函数 NS 的
无散试验空间，压力被消去，而非数值上漏掉不可压约束。

## 速度和磁扰动独立推进

令 q 表示界面及内部位移，v 表示速度，b 表示磁扰动的正交系数。
用 BSPF 导数构造 `curl(u cross B0)` 的离散算子 C，再推进

```
q_dot = v
b_dot = C v
M v_dot = -C^T b + P q - T^T S_v T q
```

P 是由基态电流与磁扰动耦合重新积分得到的对称载荷；T 是界面磁通迹算子，
S_v 是真空磁能反馈。每一个 RK4 阶段同时更新 q、v、b，磁力使用该阶段的 b。
运行时没有调用旧模型的 `-K q` 作为动量右端。测试特别验证：q=v=0、b≠0 时，
磁扰动仍能驱动速度，这区别于只改名包装位移方程。

对于理想磁通冻结的初始条件 `b=Cq`，新系统应等价于原线性位移模型。
这个恒等关系、两者的独立算子组装结果、实际轨迹和能量收支均分别核验。
总能量为 `E=(v^T M v+b^T b-q^T P q+q^T T^T S_v T q)/2`。
磁场用整个解析感应像空间的 QR 坐标表示，没有挑选单一增长模推进。

## 原张量椭圆反解怎样用于曲坐标真空

映射后的真空算子有变系数和交叉项，通常不能直接分离变量。
`_tensor_pcg.py` 将原 NS 中的张量反解作为可分离预条件器，通过 PCG 求解
实际的变系数方程，并以原方程残差判断收敛。默认相对残差阈值为 `2e-12`。
可以指定 `vacuum_solver="cholesky"` 保留稠密直接法作为对照。

当前示例仍组装了二维真空算子矩阵，尚未实现全程矩阵自由计算，不能据此声称
已经获得大规模求解的内存或速度优势。线性几何和固定线圈使真空响应可以在初始化
时求解并缓存，随后各速度阶段应用当前边界位移对应的响应；没有反复求解同一矩阵。

## 运行

```bash
export PYTHONPATH=jax/src
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python examples/pde/tokamak_velocity_mhd.py
python examples/pde/tokamak_velocity_mhd.py --far-wall --duration 30 --out build/tokamak_velocity_far
python -m pytest jax/tests/test_tokamak_velocity.py jax/tests/test_stream_navier_stokes.py jax/tests/test_navier_stokes.py jax/tests/test_cavity.py -q
```

输出包含 q、v、b 的真实时间序列、实际磁场与速度、不可压误差、磁通冻结误差、
能量变化，以及与旧位移模型的独立模态参考解的差异。
RK4 时间步需要满足 Alfvén 稳定性限制，代码会拒绝超限的步长。

## 运行结果

35项相关测试通过，包括原 NS/cavity 回归、GS 平衡、真空制造解以及新 MHD 测试。
默认 `dt=0.01`，近壁稳定构型推进到 `T=100`：

| 检查 | 数值 |
|---|---:|
| 新磁力/感应组装与旧位移刚度的相对差 | 3.06e-15 |
| 实际轨迹与旧离散位移系统精确模态解的相对差 | 3.50e-7 |
| 磁通冻结约束 `b=Cq` 的相对误差 | 2.63e-14 |
| 速度散度最大值 | 1.90e-16 |
| 相对能量漂移 | 1.45e-7 |
| 真空 PCG 最大迭代数 / 最大列相对残差 | 26 / 1.99e-12 |
| 界面磁通投影的轨迹相对误差 | 6.07e-4 |

远壁对照的实际增长率为0.160825619，与旧离散位移算子的参考增长率一致；
真空 PCG 最大57次迭代。零扰动保持零。另有独立时间加密测试验证 RK4 四阶收敛。
这些比较验证的是新旧离散系统的一致性及时间积分，不能代替物理空间收敛验证。
界面投影误差和更高阶位移方向的解析限制仍见统一 BSPF 文档。

此实现尚未加入非线性速度对流、非线性感应、黏性/电阻耗散、有限电阻壁或
非线性自由界面更新。复用已有 NS 算法并不等于已经完成完整非线性 NS–MHD。
