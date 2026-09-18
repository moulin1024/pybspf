# BSPF 光滑凸域 Poisson 求解器

入口 `bspf_jax.ConvexPoissonPlan`。仅需指定网格上的解时，可使用
[`ConvexPoissonGridPlan`](jax_convex_poisson_fixed_grid.md)，缓存一维因子进行张量求值。
求解
−Δu=f 于光滑凸闭合 B-spline 域 Ω，u=g 于 Γ。
本实现优先提供可重复验证的高阶稠密参考求解器；不要求 Fourier continuation，
没有外域延拓方程，也没有使用 MMS 真实梯度作为边界条件。

## 离散设计

矩形B上的张量BSPF场 u=Bc，限制到Ω。默认使用**无外盒边界约束**的空间，
每方向49个自由度，B=[−1.2,1.2]²，端点 Chebyshev M=P=12。
基函数、梯度和二阶导数复用原有MPFR求值路径，运行矩阵是float64。

求解目标为

    ||sqrt(WΩ)(−ΔBc−f)||² + ||SΓ(BΓc−g)||²,

其中SΓ离散弧长H^{3/2}范数。不是任意调大Dirichlet罚参数。
对于凸域严格零迹误差，Miranda–Talenti估计控制Hessian；一般边界误差
由迹提升给出 ||e||H2 ≲ ||−Δe||L2+||e|Γ||H^{3/2}。
这些是连续稳定性依据，具体采样还须独立验证。

具体步骤：

1. 在每个三次样条段上构造曲率分子与速度平方的多项式，检查其端点和
   内部驻点，并积分总转角。拒绝凹域、反向/多圈或非正则边界。
   检查使用浮点容差，极端退化几何不属于精度认证范围。
2. 分段高阶积分计算弧长，数值求逆产生均匀弧长点；位置仍评价原始样条。
   默认边界点数为不小于8N的2次幂（N=49时512点）。
3. 对真实边界迹做归一化实FFT，第m模态乘(1+k_m²)^{3/4}，包含
   周长、点数及正负频率重数。FFT仅用于边界范数，不替换体场BSPF。
4. 使用背景盒实际H2 Gram作系数尺度：G=I+Kx⊗I+I⊗Ky+Jx⊗I
   +2Kx⊗Ky+I⊗Jy，其中J由二阶导数积分得到，绝不把J假定为K²。
5. 对右白化后的完整矩形最小二乘算子做稠密SVD，默认相对阈值1e-13。
   不形成正规方程用于求解，也不按域内质量矩阵先删除模式。
6. 同一几何计划可复用求解不同f、g。独立验证采用加密体积分和偏移弧长
   网格，报告PDE残差、边界H^{3/2}范数、最大边界误差及高频尾部。

## 使用

```python
import jax
import numpy as np
from bspf_jax import ConvexPoissonPlan
from bspf_jax.embedded_poisson import benchmark_domains

jax.config.update("jax_enable_x64", True)
domain = benchmark_domains()[0]
plan = ConvexPoissonPlan(domain, nodes=49)

def g(p):
    return 1 + .2*p[:, 0] + .3*p[:, 1] + .1*p[:, 0]**2

def f(p):
    return np.full(len(p), -.2)

solution = plan.solve(f, g)
u, gradient, laplacian = solution.evaluate(np.array([[0., 0.], [.2, .1]]))
print(solution.validate(f, g))
```

注意返回的laplacian是Δu，PDE采用−Δu=f。
API需在构建基函数前开启JAX float64；不会在导入时偷偷修改用户精度配置。

## 精度实验的解释

`examples/pde/convex_poisson.py` 比较普通L2边界范数、H^{3/2}边界范数、
H2系数尺度。低波数4π与高波数12π使用同一64波随机MMS。
`--cached-dirichlet` 仅供复用旧实验的2209维外盒Dirichlet空间做同空间对照，
不是默认2401维无约束空间；两类结果必须分开报告。

`convex_poisson_best_h2.py` 为旧2209维空间提供独立的H2拟合诊断：
它使用MMS值/梯度/完整Hessian，正式PDE求解器不读取这些数据。
该离散拟合是可达到精度的诊断，不是连续最佳逼近的严格下界。
高波数独立相对H2误差约3.09e-6，值1.28e-7、梯度2.71e-7；
低波数H2误差约6.65e-10。固定N的误差平台不能全部归因于边界处理。

同一旧2209维空间、同一512点弧长边界、最高波数12π的独立错位网格结果：

|方法|值相对误差|梯度相对误差|Laplacian相对误差|
|---|---:|---:|---:|
|L2边界＋列缩放|5.50e-7|4.73e-6|1.65e-6|
|H^{3/2}边界＋列缩放|9.29e-8|2.41e-7|1.43e-6|
|H^{3/2}边界＋H2尺度|8.70e-8|2.36e-7|1.37e-6|

第一、二行SVD阈值均1e-14；第三行为默认1e-13，所以第三行包含尺度与
截断共同改变，不能把其全部收益归给白化。单独更换边界范数已经显著
改善值/梯度，同时没有恶化Laplacian误差。

默认无外盒约束的2401维空间在N=49得到：

|最高波数|值相对误差|梯度相对误差|Laplacian相对误差|
|---|---:|---:|---:|
|4π|3.33e-12|2.34e-12|2.33e-11|
|12π|1.90e-8|6.93e-8|4.68e-7|

独立体积分阶20、偏移边界1024点进一步检查，高波数边界最大误差为
1.12e-9，H^{3/2}误差7.84e-7，PDE相对积分残差约4.22e-7。
这类不同空间结果不能作为仅改变边界范数的消融实验。

加密到默认无约束 N=65（4225自由度，1024边界点）：

|最高波数|值相对误差|梯度相对误差|Laplacian相对误差|
|---|---:|---:|---:|
|4π|3.24e-12|5.35e-12|5.41e-11|
|12π|2.31e-9|2.41e-9|1.38e-8|

高波数独立体积分PDE相对残差1.51e-8，独立偏移2048点边界最大误差
8.16e-12。边界H^{3/2}误差2.29e-9，其中高频尾部1.91e-9，已需注意
浮点求值噪声的影响。低波数值误差已接近平台，加密不再一致改善所有
低波数指标；不宣称任意数据都能达到机器精度，也不从两个N推断严格收敛阶。

最终3项组合测试覆盖几何/弧长、分数阶范数、真实H2矩阵、二次Poisson
及Hessian；此前与法向坐标测试合跑5项通过。ruff通过。

可运行：

```sh
export PYTHONPATH=jax/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python examples/pde/convex_poisson.py --single --out build/convex_poisson_unrestricted
python examples/pde/convex_poisson.py --nodes 65 --boundary-count 1024 --single --out build/convex_poisson_n65
python examples/pde/convex_poisson.py --cached-dirichlet
python examples/pde/convex_poisson_best_h2.py
MPLCONFIGDIR=/tmp/bspf-mpl python examples/pde/render_convex_poisson.py
python -m pytest -q jax/tests/test_convex_poisson.py
```

`--resume`只用于同一空间/参数的中断实验。缓存来自本地旧实验，必须可信。
所有JSON与独立点NPZ在指定build输出目录，结果不会自动提交到版本库。

## 当前实现边界

- 限定实数、二维、光滑凸样条、常系数Poisson、Dirichlet条件。
- 三次几何为C2，不宣称对任意数据都具备无限阶/指数收敛。
- 稠密SVD成本及存储较高；这是准确性优先版本，尚无大规模矩阵自由实现。
- 返回的投影残差是保留SVD空间上的残差。真实场可能存在系数相消，
  必须同时查看独立验证，不能仅凭线性代数残差宣称高精度。
- H2白化减少不合理的系数尺度差异，不消除域限制带来的全部近相关性。
  固定rcond或固定采样数量不构成任意分辨率下的精度保证。
- 目前实现了独立残差与边界尾部检查，尚未完成整个保留空间的采样范数
  广义特征值认证。高精度结论限于实测配置。
