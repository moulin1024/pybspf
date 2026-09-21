# 等离子体—真空自由边界的线性耦合

**更新：默认真空求解器已统一为 BSPF。** 下文的 P1 有限元结果作为历史对照保留；
使用 `--vacuum-method fem` 可复现该后端。当前方法见
[全 BSPF 空间离散说明](jax_tokamak_all_bspf.md)。

入口为 `examples/pde/tokamak_plasma_vacuum.py`，核心为
`packages/models/src/bspf_models/plasma/tokamak_vacuum.py`。这条路径替代原来的外部流体近似：
真空中没有速度、密度、黏性、电阻率或动量方程。原有 halo 示例保留作对照。

## 方程及界面

沿用固定外部线圈、自洽 Grad–Shafranov 平衡和常数 `F=R B_phi`。
等离子体中使用不可压位移 `xi_R=-chi_Z/R, xi_Z=chi_R/R`，并保留 `xi_phi`。
理想感应扰动为 `Q=curl(xi cross B0)`，界面自由变形。

真空扰动满足 `Delta* delta_psi_v=0`，界面满足
`delta_psi_v=-xi dot grad(psi0)`，外部理想导体壁满足 `delta_psi_v=0`。
这里的零值是**磁通扰动**，不要求基态壁面磁通为常数。
外部固定线圈无扰动电流。本实现限制为上下反射的垂直模分支，真空环向
扰动 `delta F=0`；一般偶对称扰动需要另行处理整体环向磁通约束。

通过扩展能量原理耦合等离子体和真空：

```
2 delta W = integral_plasma (|Q|^2 - xi dot (J0 cross Q)) R dR dZ
            + integral_vacuum |grad(delta_psi_v)|^2 / R dR dZ
M q_ddot + K q = 0
```

本平衡在边缘 `p=j=0`，没有平衡面电流，因此表面能项为零；不可压约束消除了
体积中的压力散度项。真空磁能的变分提供界面应力反馈，未把界面固定，也没有
用显示遮罩代替真空模型。依据为 [Freidberg 的扩展能量原理，式 5.41–5.44](https://harvest.aps.org/v2/journals/articles/10.1103/RevModPhys.54.801/fulltext)。

## 离散与适用范围

- 等离子体：限制在真实等离子体区域内的 BSPF 张量函数；三分量位移。
- 边界：沿射线寻找中心闭合 `psi0=0` 曲线；仅支持相对 `(2,0)` 星形的光滑边界，无 X 点。
- 真空：贴体三角形 P1 有限元，消去内部磁通得到边界 Dirichlet-to-Neumann 能量矩阵。
  因此这是 BSPF/FEM 混合方法，不是全域纯 BSPF。
- 限制域质量矩阵用加权采样 SVD 正交化，默认舍弃相对质量小于 `1e-10` 的方向。
- BSPF 任意点求值使用密集采样的 C2 五次 Hermite 插值；与多精度直接求值独立核验。
- 时间演化用隐式中点法，实际推进位移和速度；无外部驱动。
- 采用线性化自由边界：在平衡域组装，显示 `x+xi`。没有非线性动网格或触壁演化。

## 运行与验证

```bash

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python examples/pde/tokamak_plasma_vacuum.py
python examples/pde/render_tokamak_vacuum.py
python examples/pde/tokamak_vacuum_convergence.py
python -m pytest packages/models/tests/test_tokamak.py packages/models/tests/test_tokamak_vacuum.py -q
```

默认平衡 n=49，位移截断 modes=14，角向256点、径向64点积分，真空48层。
远矩形壁算例 `gamma=0.161247926`，实际时间拟合 `0.161251411`；界面节点磁通
误差为零，真空弱方程相对残差 `6.52e-16`，外部流体自由度为零。
这些是离散方程误差，不代表连续解精度。

真空/积分网格从192角点、32层到256角点、48层，增长率变化约0.064%；
位移截断从14到16变化约0.00027%；平衡 n=41 到49变化约0.0027%。
质量阈值 `1e-9` 到 `1e-11` 的增长率差约0.0021%。
测试还包含制造真空解的二阶网格收敛、独立 curl/div 验证、一般混合模的能量守恒。

输出在 `build/tokamak_vacuum/`：模型缓存、时间轨迹、诊断、PNG、MP4。
`display_fields.npz` 中真空速度为 NaN，表示该变量不存在；并非把求得的外部速度隐藏。
缓存 pickle 只允许加载自己生成的可信文件。

## 工程导向构型

另见 [被动壁稳定化算例](jax_tokamak_confined.md)，用于研究壁间隙对垂直稳定性的影响。
