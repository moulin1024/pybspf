# 保留 FLR 的静电 slab 回旋动理学原型

实现：`packages/models/src/bspf_models/kinetic/gyrokinetic_slab.py`；可运行算例：
`examples/pde/gyrokinetic_slab.py`。

## 模型与归一化

均匀直磁场，周期 (x,y,z)，动力学离子、绝热电子、无碰撞、无背景梯度。
演化量 g=δF/F0，速度 v 以 sqrt(Ti/mi) 为单位，μ=μphysical B/Ti，
电势 φ=eΦ/Ti。平衡速度积分权为
exp(-v²/2)/sqrt(2π) × exp(-μ) dv dμ。
长度、时间及电势的参考尺度选为平行自由飞行与 E×B 括号系数为一；
参数 rho 表示该长度单位下的热拉莫尔半径，rho=0 用作形式上的无 FLR 对照。

方程为

\[
\partial_t g+v\partial_z(g+\mathcal J\phi)
 +\{\mathcal J\phi,g\}_{xy}=0,\qquad
\{a,b\}=a_xb_y-a_yb_x,
\]

\[
\mathcal J_{\mathbf k}(\mu)=J_0(k_\perp\rho\sqrt{2\mu}),\qquad
D_{\mathbf k}\phi_{\mathbf k}=\langle\mathcal J_{\mathbf k}g_{\mathbf k}\rangle_{v,\mu},
\quad D_{\mathbf k}=\tau+1-\Gamma_{0,\mathbf k},\quad \tau=T_i/T_e.
\]

连续 Maxwell 积分给 Γ0=I0(b)exp(-b)，b=kperp² rho²。
代码用同一 μ 积分规则计算 Γ0=Σwμ J0²，以保持离散场耦合一致性，
并用解析表达式测试积分精度。完整 Bessel 函数同时进入电势回旋平均和电荷沉积；
不使用小 b 展开。

电子对所有非恒定空间模作 Boltzmann 响应，**没有移除 zonal 模电子响应**。
全空间常数模 φ000=0，其均匀密度扰动由固定中性背景吸收。
这是明确选择的 slab 闭合；研究 zonal flows 时需要改为相应电子闭合并处理零模。

该方程是标准小扰动排序下的非线性 δf 原型：保留平行电场作用于 Maxwell
背景产生的驱动，以及 E×B 非线性；未加入高阶平行加速度作用于 δf 的项。
不含磁镜力、磁曲率、磁剪切、碰撞、电磁扰动或多动力学物种。
不能把它当成此前磁镜 full-f 模型的通用替代品。
场与分布联立的实现背景可参阅 [GS2 的静电算法说明](https://gyrokinetics.gitlab.io/gs2/page/user_manual/timestepping.html)；
本原型使用显式 RK4，而非该页面的隐式推进算法。

## 离散与守恒

所有空间轴使用周期 FFT，采用严格 |mode|<N/3 的谱截断去除二次混叠。
v 使用 Gauss–Hermite、μ 使用 Gauss–Laguerre；在这个排序下没有速度方向输运，
因此不用构造速度导数矩阵。电势逐 Fourier 模直接除以 D。
周期 slab 是先前快速轴路线的纯 Fourier 特例，本模块不调用非周期样条修正核。
生产代码没有空间或相空间 N×N 矩阵；独立线性测试中才组装小矩阵指数参考。

周期边界下，无边界通量。监测体积平均粒子扰动与二次自由能

\[
\delta N=\langle g\rangle_{x,v,\mu},\qquad
W=\tfrac12\langle g^2\rangle_{x,v,\mu}
 +\tfrac12\langle\phi D\phi\rangle_x.
\]

代入方程，平行项是周期全导数；括号项对 g 与 Jφ 的积分均为零，
故半离散 dW/dt=0。时间离散采用 RK4，不强制修正 W。
**W 是扰动自由能，不是 full-f 模型的总物理能量**。
步长须解析最大保留 kz v 和非线性输运频率；长时间需检查速度相混合分辨率及复现。

δf 可有正负；诊断报告 min(1+g) 而非要求 g 非负。
小扰动示例检查节点上的 F/F0>0，但此方法没有任意幅值下的全局保正保证，
也没有裁剪或套用旧的 log-f 更新。

## 运行

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python examples/pde/gyrokinetic_slab.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest packages/models/tests/test_gyrokinetic_slab.py -q
```

使用 JAX float64（算例内启用）。输出在 `build/gyrokinetic_slab/`：
`solution.npz` 包含 g、φ、无 FLR 对照和诊断；`report.json` 包含验收指标；
`validation.png` 显示电势、自由能交换、步长收敛及电势截面。

## 已运行的验证（2026-09-21）

四项新测试通过：Bessel 回旋平均与直接圆周积分对照、Γ0 与解析 Maxwell
积分对照和 rho=0 极限、自洽场残差与零模规范、含非线性项的粒子/自由能瞬时收支，
以及线性单模与独立矩阵指数参考。（部分检查合并在同一测试内。）
原有磁镜与平行输运的 13 项回归测试也通过。
线性对照针对离散速度系统，尚未声称完成连续 Landau 阻尼率或湍流谱基准。

非线性算例为 9×9×9×16×16，rho=0.8，tau=1，t=0…2，dt=0.005；
两组不同方向、不同速度依赖的初始模使 E×B 括号非零。

| 验证指标 | 实测 |
|---|---:|
| 最大绝对粒子扰动漂移 | 3.422e-18 |
| 最大相对自由能漂移 | 1.903e-12 |
| 步长减半后的相对自由能漂移 | 5.740e-14 |
| 节点最小 F/F0 | 0.909589 |
| 步长减半的末态最大 g 差异 | 9.455e-9 |
| 连续两次减半的误差比 | 16.062 |
| 初态非线性 RHS 欧氏范数 | 0.062326 |
| FLR / rho=0 末态最大电势差异 | 0.009279 |
| 保留模上的 Γ0 积分误差 | 2.443e-15 |

时间收敛检查使用 dt=0.01、0.005、0.0025。自由能从电势部分转移到分布部分，
总和保持稳定。FLR 对比保持相同初始 g，因此两者初始自洽电势本来就不同；
电势差异包含场闭合和演化两方面的改变，不能单独解读为阻尼率差异。
当前算例用于原型验收，尚未进行完整空间/速度分辨率扫描。

后续新增了[独立解析 MMS 与时间、空间、速度积分验证](jax_gyrokinetic_mms.md)，
包括具有非零非线性括号的制造解和解析自由能源项收支。
