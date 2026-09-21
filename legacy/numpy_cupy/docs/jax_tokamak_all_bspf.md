# 统一 BSPF 的等离子体—真空求解

进一步的代码复用已接入 [速度—感应推进路径](jax_tokamak_velocity.md)：它与旧 NS
共用微分/张量惯性/RK4 核，默认真空椭圆系统采用张量 Poisson 预条件 PCG。

`assemble_plasma_vacuum`、`tokamak_plasma_vacuum.py` 和 `tokamak_confined.py`
现在默认选择 `vacuum_method="bspf"`。磁平衡、等离子体位移、真空磁通的未知场
均在 BSPF 空间中离散。原来的 P1 有限元求解器只在明确指定 `fem` 时使用。

## 真空区的映射与弱形式

在等离子体和导电壁之间，以 `s∈[0,1]`、`theta∈[0,2*pi]` 参数化：

```
x(s,theta) = (2,0) + rho(s,theta) (cos(theta),sin(theta))
rho = a(theta) + s [b(theta)-a(theta)]
```

`a` 是平衡等离子体的射线半径，`b` 是固定外壁的射线半径。
两者均来自真实几何，未把真空区填成流体。设 `h=b-a`，物理大半径为 R，
真空磁能中的积分转化为

```
integral [A f_s^2 + 2 B f_s f_theta + C f_theta^2] ds dtheta
A = (rho^2 + rho_theta^2)/(h rho R)
B = -rho_theta/(rho R)
C = h/(rho R)
```

因此保留了曲坐标 Jacobian、交叉项和轴对称 `1/R` 权重。
径向使用已有 `_stream_line` 的齐次 Dirichlet BSPF 函数，另用该母空间包含的
线性多项式 `1-s` 和 `s` 提升边界数据。自由未知量通过能量最小化求得。

光滑贴体外壁的角向使用 BSPF 函数的周期 C2 子空间。矩形外壁在四条角点射线
上会使映射导数发生跳变，因此使用四个 BSPF 区段，并保证场值连续；不强行
要求参考坐标导数连续。每个区段也由齐次 BSPF 函数和多项式边界提升组成。

## 与等离子体耦合

从等离子体 BSPF 位移计算 `g=-xi dot grad(psi0)`，将其投影到真空角向迹空间。
真空内部系数被消去，得到对称的边界磁能矩阵，再装配到等离子体位移方程。
边界条件按有限维投影满足，因此需报告界面投影误差，不能称作连续界面上精确匹配。

默认光滑壁使用径向24、角向121个模式；矩形壁使用径向31、总计96个分片角向模式。
这与平衡的49个节点、等离子体的161个正交化位移自由度是不同的分辨率参数。
`vacuum_layers` 在 BSPF 后端只控制显示采样；求解精度由 BSPF 模式数与积分控制。
文件中保留的 `triangles` 只用于画图，不组装三角形有限元刚度矩阵。

曲线求值仍使用已核验的密集五次 Hermite 加速求值；这是一项求值近似。
固定线圈场采用解析电流环公式，时间推进采用隐式中点法。统一 BSPF 指的是
未知场的空间离散，并不意味着用 BSPF 替代时间积分或解析线圈模型。

## 复现

```bash
export PYTHONPATH=jax/src
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python examples/pde/tokamak_plasma_vacuum.py --out build/tokamak_vacuum_bspf
python examples/pde/tokamak_confined.py --out build/tokamak_confined_bspf
python examples/pde/render_tokamak_vacuum.py --out build/tokamak_vacuum_bspf
python examples/pde/render_tokamak_confined.py --out build/tokamak_confined_bspf
python -m pytest jax/tests/test_tokamak_vacuum.py jax/tests/test_tokamak_vacuum_bspf.py -q
```

`--vacuum-radial-modes` 和 `--vacuum-angular-modes` 可用于真空增长算例的空间收敛检查。
测试覆盖独立制造解 `psi=R^2` 与 `psi=Z`、物理坐标下的真空方程残差、角向周期性、
矩形角点区段连接、能量守恒，并禁止 BSPF 路径调用有限元求解器。

## 已运行的对照结果

相关9项测试通过。与保留的细网格 BSPF/FEM 版本比较：

| 指标 | BSPF/FEM | 统一 BSPF |
|---|---:|---:|
| 远矩形壁、F=6 的不稳定增长率 | 0.161247926 | 0.161347580 |
| 近壁稳定构型的主导垂直角频率 | 0.297177558 | 0.297170506 |

增长率差约0.0618%，角频率差约0.00237%。这是两种离散结果的差异，不能作为
对连续解的严格误差界。默认增长模的界面磁通相对投影误差为0.00510%；稳定轨迹
全时段的相对投影误差为0.0607%，相对能量漂移为5.93e-13。

对整个保留的位移空间，界面迹算子的 Frobenius 相对投影误差仍分别为远壁2.56%、
近壁0.898%，高阶方向的界面解析程度低于上述实际轨迹。研究其他初始扰动或高阶模时
需要继续检查角向截断，不能用当前增长率收敛替代所有场量的收敛验证。

若旧对照与新结果都已生成，可运行 `python examples/pde/tokamak_bspf_compare.py`
重算实际轨迹误差和比较表，结果写入 `build/tokamak_confined_bspf/bspf_comparison.json`。

这仍是轴对称不可压理想 MHD 的线性自由边界模型，支持光滑星形等离子体边界。
尚未加入非线性界面运动、X 点、有限电阻壁、主动反馈或非轴对称扰动。
