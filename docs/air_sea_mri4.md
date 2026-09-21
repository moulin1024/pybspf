# 四阶海气耦合：MRI-GARK-ERK45a / RK4

默认海气时间推进已由一阶海洋冻结方案改为 **四阶显式 MRI-GARK-ERK45a**。
该方法具有专门的快—慢耦合系数，不是把两个独立 RK4 的输出插值拼接。
海洋仍采用 BSPF，水平大气仍为 x 周期、y 不可穿透自由滑移；物理参数和标量方程
见 [模型说明](air_sea_double_gyre.md)。空间离散和物理闭合在本次升级中不变。

## 运行与 API

```sh
PYTHONPATH=jax/src OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/bspf-mpl \
  python examples/pde/air_sea_double_gyre.py --method mri-gark4 \
  --hours 24 --dt-ocean 300 --dt-air 60 --window 600 --output-hours 0.5

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=jax/src OPENBLAS_NUM_THREADS=1 \
  python -m pytest -q jax/tests/test_multirate.py jax/tests/test_air_sea.py

PYTHONPATH=jax/src OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/bspf-mpl \
  python examples/pde/validate_air_sea_mri4.py
```

默认模拟目录为 `build/air_sea_double_gyre_mri4/`，验证目录为
`build/air_sea_mri4_validation/`。不会覆盖已有的一阶演示或其 MP4。
需要制作新方案视频时，使用
`python examples/pde/render_air_sea_mp4.py --run build/air_sea_double_gyre_mri4`
（同样设置上面的 PYTHONPATH 和 MPLCONFIGDIR）。

```python
import jax
from bspf_jax.air_sea import (
    plan_air_sea, plan_air_sea_stepper, initial_air_sea_state, air_sea_step,
)
jax.config.update("jax_enable_x64", True)
p = plan_air_sea(n=33)
stepper = plan_air_sea_stepper(dt_air=60, dt_ocean=300, window=600)
state = initial_air_sea_state(p)
advance = jax.jit(lambda s: air_sea_step(p, stepper, s))
state, integrated_budget = advance(state)
```

旧方案仍可通过 `method="lagged"` 或 CLI `--method lagged` 选择。
旧加密脚本 `validate_air_sea_double_gyre.py` 已显式固定为该一阶对照。

## 时间参数的新含义

| 参数 | 四阶方案中的含义 |
| --- | --- |
| `dt_ocean` | MRI 慢宏步 H，每步求值 5 次海洋内部 RHS |
| `dt_air` | 快子步的上限，阶段边界必须对齐 |
| `inner_steps` | 每个 H/5 区间的 RK4 子步数 m=ceil(H/(5 dt_air)) |
| `effective_dt_air` | 实际快步长 h=H/(5m)，不大于指定上限 |
| `window` | 同步/输出分组，必须是 H 的整数倍 |

例如 H=300 s、dt_air=60 s 时，m=1，每个宏步有 5 个快区间、共 5 个 RK4 子步。
每个宏步有 **5 次慢 RHS 和 20m 次快 RHS**，不是每个快阶段都重算完整海洋内部动力。
H=150 s、dt_air=17 s 时，m=2，实际 h=15 s。

**现在只改变 window、保持 H 和 h 不变，不再改变时间离散。** 它只改变相同宏步的
分组方式；测试核验一个 600 s 分组与两个 300 s 分组一致。时间收敛实验必须细化 H，
并控制内部快积分误差，不能继续把“输出窗口减半”称作耦合加密。

## 为什么四阶且阶段相容

把整个状态 Y（海洋、大气和预算变量）拆成

\[
Y'=F_f(t,Y)+F_s(t,Y).
\]

- `F_s`：海洋内部平流、黏性、β 平面旋转、底摩擦，SST 平流扩散和外部辐射。
- `F_f`：大气动力与温湿输运、恢复项，以及 **双方的完整界面交换**。

该拆分按物理过程而非严格按流体划分。海洋受到的应力、显热和潜热也属于快分量，
这样每个快阶段都使用同一个正在演化的海流、SST、风、温度和比湿。
海洋内部动力只在 5 个慢节点计算，随后由 MRI 多项式强迫进入快区间。

慢节点为 `c=(0,1/5,2/5,3/5,4/5,1)`。在区间 i 内，令
`θ=(t−t_n−c_iH)/(H/5)`，求解修正快方程：

\[
v'=F_f(t,v)+5\sum_{j=0}^{i}
\bigl(\Gamma^{(0)}_{ij}+\theta\Gamma^{(1)}_{ij}\bigr)F_s(t_n+c_jH,Y_j),
\qquad v(t_n+c_iH)=Y_i.
\]

区间终点成为下一慢阶段 `Y_(i+1)`。**系数中的乘 5 来自除以 Δc=1/5**；
遗漏它或把 θ 换成整个宏步的时间坐标，会破坏阶条件。
修正快方程用经典 RK4 积分，在每个子步的 0、1/2、1/2、1 时刻使用对应的阶段状态
和线性多项式强迫。

有理系数保存在 `jax/src/bspf_jax/multirate.py`，出处为 Sandu 的
MRI-GARK-ERK45a；已与 SUNDIALS v7.2.1 主方法系数逐项核对。
未使用该表的三阶嵌入公式，也未添加自适应步长。

## 守恒不是事后修补

界面双方的载荷在同一快 RHS 内生成，因此应力作用与反作用、显热和潜热的交换
使用同一组阶段值和积分权重。海洋的慢内部项不包含另一份界面通量。

`WindowBudget` 作为附加 ODE 状态与流体一起推进；快、慢 RHS 各自都满足相应的线性
收支恒等式，所以 MRI 的阶段线性组合及内层 RK4 都保留这些恒等式，误差限于空间
积分和浮点运算。没有在步末强行改温度、比湿或能量以“修正”残差。
测试分别验证两个拆分 RHS 的热量、水分和大气纬向动量预算，再验证完整推进。

## 验证层级

1. **代数阶条件。** 独立 NumPy 记账展开实际 MRI+RK4 的全部慢/快导数事件，形成
   完整加性 RK tableau。枚举所有一至四阶的 72 个双色有根树，检验权重等于
   `1/tree_factorial`；同时检验显式性以及两分量的内部时间相容。
   m=1、2、3、7 全部通过，容差 3e−13。并非只检验经典 RK4 的八个无色条件。
2. **独立精确解。** 双向耦合、不对易的线性分裂与矩阵指数对比；非线性、显含时间
   的制造解检验阶段时间和四阶全局误差。另验证 pytree 附加预算的线性不变量。
3. **实际半离散海气方程。** 同一 n=33 空间模型，H=2400/1200/600/300/150 s，
   实际 h=H/10，积分至 4800 s。用相邻加密解差估计阶，并保留一阶 lagged 对照。
   参数为 H_o=50 m、h_m=5 m、h_a=300 m，其余与第一版一致。
4. **物理演示。** 默认水深/层厚下运行 24 小时，并核验热量、水分和大气纬向动量预算。

验证脚本把误差、收支和观测阶写入 JSON，并输出 `convergence.png`。
RMS 在质量归一化系数空间计算，避免把接近机器精度的温度误差放到约 290 K 的绝对值
中相减。组合范数的尺度仍为海流 0.1 m/s、风 8 m/s、两个温度各 1 K、比湿 0.01 kg/kg。

本次实际结果：相关 **31 项测试通过**，包括每个 m 下全部 72 个阶条件。

| 相邻宏步对 | MRI 四阶方案的组合 RMS 差 |
| --- | ---: |
| 2400 / 1200 s | 3.277e−9 |
| 1200 / 600 s | 1.992e−10 |
| 600 / 300 s | 1.227e−11 |
| 300 / 150 s | 7.608e−13 |

对应观测阶为 **4.0403、4.0213、4.0109**；旧一阶对照为
**1.0622、1.0294、1.0142**。这些值来自相邻解之差，不是假定细步结果为精确解。
默认 24 小时演示也已跑通，保留了半小时真实模拟快照供 48 帧视频使用。
该运行最大热力预算残差为 `2.63e−8 J/m²`，水分残差 `5.84e−14 kg/m²`，
大气纬向动量残差 `6.94e−18 kg/(m·s)`；均未采用步末守恒修正。

## 精度范围与限制

对足够光滑的半离散方程、稳定步长和固定步长比例，时间误差为 `O(H^4)`；固定 H
只细化快子步，最终会停在慢步误差。空间误差仍需独立网格加密评估。
本方法为显式方法，四阶不等于无条件稳定；海洋扩散、快过程和海气交换都可能限制步长。
相对风过零等非光滑情形可能影响高阶收敛，不能将光滑制造解结论无条件推广。
当前仍无隐式界面迭代、保正限制器、云凝结、垂向湍流或海洋斜压反馈。

方法资料：
[Sandu, arXiv:1808.02759](https://arxiv.org/abs/1808.02759)、
[SUNDIALS v7.2.1 主方法系数](https://github.com/LLNL/sundials/blob/v7.2.1/src/arkode/arkode_mri_tables.def)、
[SUNDIALS MRI 方程与时间归一化](https://sundials.readthedocs.io/en/v7.2.1/arkode/Mathematics_link.html#multirate-infinitesimal-mri-methods)。
运行时没有引入 SUNDIALS 依赖。
