# 无驱动、无耗散的多模态非线性守恒测试

在[标准 BSPF 线性 ITG](jax_linear_itg.md)基础上加入多模态 E×B 非线性耦合。径向保持有限区间齐次 Dirichlet 条件、标准 BSPF 和完整数值 Bessel FLR，y、z 周期。密度/温度梯度、碰撞、人工耗散全部关闭。本测试验证离散动力学的守恒结构与非平凡模态转移，尚不是有驱动的 ITG 饱和或雪崩。

## 方程与闭合

沿用线性模型的速度权重、归一化和互补扰动 g，记 ψ=Jφ、X=√w g。在径向质量正交基和周期 Fourier 坐标中，

$$\partial_tX=-i\Omega(X+\sqrt w\,J\phi)+\rho B(\psi,X),\qquad
\Omega=k_zv+\rho k_y\kappa(v^2+\mu).$$

非线性括号在实空间周期坐标计算，连续约定为

$$\{a,b\}=a_xb_y-a_yb_x.$$

极化与回旋平均使用相同速度求积：

$$D\phi=\sum_{v,\mu}\sqrt w\,JX,\qquad
D=1-\Gamma+\tau(1-P_{00}),\quad \Gamma=\sum_\mu w_\mu J^2.$$

P₀₀ 只选中 ky=kz=0 的带状模态；ky=0、kz≠0 仍保留绝热电子响应。有限径向 Dirichlet 特征值为正，使带状模态极化分母也为正，不添加人为小分母。径向 FLR 仍是 Dirichlet 拉普拉斯的谱函数闭合，可理解为奇反射延拓，不代表鞘层边界。

## 离散守恒结构

径向求积值 Q、导数 G 均来自标准 BSPF，保留全部 Nx−2 个径向自由度。沿用适度端点结点加密和129点上限。运行时使用径向小矩阵变换，不组装完整相空间矩阵。

令内积包含径向 Gauss 权重与周期平均，定义

$$T(c,a,b)=\langle c,a_xb_y-a_yb_x\rangle_q,$$

$$\langle c,B(a,b)\rangle=
\frac{T(c,a,b)+T(a,b,c)+T(b,c,a)}3.$$

这个三线性形式对任意两个参数完全反对称；实现使用 Q、G 的同权重伴随和周期反对称导数。分部积分充分解析时它还原通常的 Poisson 括号。独立测试用解析正弦/余弦导数对照，避免只验证一个人为守恒算子。

y、z 使用严格 2/3 Fourier Galerkin 截断。输入初值与 RHS 位于同一截断空间；时间步后没有滤波、裁剪、能量投影或缺陷修补。

守恒量按径向积分、周期 y/z 平均归一化，Fourier 变换采用正交归一化：

$$S=\frac{1}{2N_yN_z}\sum X^2,\qquad
E=\frac{1}{2N_yN_z}\sum D|\widehat\phi|^2,\qquad W=S+E.$$

- 纯 E×B 非线性：S、E 分别守恒，可以发生带状/非带状场能转移。
- 加入平行运动和磁漂移：S 与 E 交换，总自由能 W 守恒。
- RK4 不精确保持二次不变量，因此另做时间步收敛；半离散功率由自动微分直接作用于生产 RHS 检查。

此处 S 是熵型二次量，不是完整热力学熵。当前 Dirichlet Galerkin 空间不包含常数测试函数，本次没有证明或声称非平凡总粒子数精确守恒；粒子/剖面收支仍需后续单独处理。

## 实测结果

基准 Nx=33，保留31个径向自由度；Ny=9、Nz=7，截断后两个周期方向均含0、±1、±2模态；Nv=6、Nμ=4。参数 ky,min=0.3、kz,min=0.1、ρ=τ=1、κ=0.2。三组径向/周期混合扰动含速度依赖，幅度0.7，初始没有带状场。这是有限幅度数值守恒测试，不作湍流分辨率收敛结论。

推进至 t=20，报告保存时刻的最大相对漂移：

| Δt | 纯非线性 W 漂移 | 全无驱动系统 W 漂移 |
|---:|---:|---:|
|0.2|6.72e−14|5.29e−6|
|0.1|5.37e−14|1.67e−7|
|0.05|5.30e−14|5.23e−9|

最细时间步纯非线性 S、E 漂移分别为8.05e−14、3.33e−15；状态相对初值变化20.03%，带状场能从舍入零增长到总场能的1.316%。因此守恒不是因为解静止或括号退化为零。

最终状态的相邻时间步差之比分别为15.993（纯非线性）与15.974（全系统），符合 RK4 四阶。全系统能量漂移本例约按五阶下降，不能据此把状态精度称为五阶。纯非线性能量漂移已到舍入平台。

全系统中段快照满足 dS/dt≈−2.24123e−4、dE/dt≈+2.24123e−4、dW/dt≈−4.36e−18，显示实际交换和抵消。JSON 中 normalized_rate 定义为相应功率除以 ||X||₂ ||RHS||₂，范数使用存储数组的欧氏范数。

## 复现与文件

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=jax/src python examples/pde/nonlinear_itg_conservation.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=jax/src python -m pytest -q jax/tests/test_nonlinear_itg.py
```

脚本启用 JAX 双精度。公共 API 位于 `bspf_jax.nonlinear_itg`，并从包顶层导出。输出在 `build/nonlinear_itg_conservation/`：`report.json`、`conservation.png/pdf`、`nonlinear.npz`、`full.npz`。NPZ 保存时间、五项诊断、Fourier 电势和最终状态。

测试覆盖带状闭合、三线性完全反对称、Fourier 支持、独立解析括号、随机多模态瞬时守恒、退化回原无驱动线性模型及带状模态生成。下一步恢复梯度驱动时，应在同一 W 诊断上加入驱动功预算，再评估可控碰撞与长时饱和。
