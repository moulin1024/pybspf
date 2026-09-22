# 两条 B-spline 边界之间的 Poisson / Helmholtz Dirichlet 原型

实现：`packages/models/src/bspf_models/elliptic/spline_annulus.py`。
实验：`examples/pde/spline_annulus_convergence.py`。

目标是直接在两条精确闭合 B-spline 所界定的双联通区域内求解

```
L_sigma u = (-Delta + sigma) u = f,
u|Gamma_outer = g_outer,  u|Gamma_inner = g_inner.
```

`sigma=0` 是 Poisson；`sigma=alpha**2>0` 是修正 Helmholtz；
`sigma=-k**2<0` 是振荡型 Helmholtz。当前只支持常实系数、标量、双 Dirichlet。
返回值使用复数 dtype（包括实问题），不强行丢弃虚部。

这是 NumPy/SciPy 的模型层精度参考实现，不是新的 JAX 核心或三角形有限元。
采用精确 B-spline 几何、源项 Fourier extension 和高阶面板 Nyström。
尚未把体源项替换为完整的 BSPF 快速延拓，也没有 FFT/FMM 加速。

## 几何

`SplineBoundary` 接收 `scipy.interpolate.BSpline`，平面系数 `(n,2)`、axis=0，
次数至少为 2，闭合且一阶导数匹配。支持非均匀 knot、非整数参数区间和两种方向。
所有原始 knot 都必须是积分面板断点；细化只增加断点，不重新拟合几何。

两条曲线必须简单、正则、互不相交，内曲线严格包含于外曲线。构造器检查闭合、
knot 重数，采样检查速度与嵌套；**没有实现任意曲线的自交／相交严格认证**。
角点、NURBS 和不规则参数化不在本版支持范围内。

体内点分类由原始 spline 的竖直切片交点完成，不把几何折线化。
切片切向退化会明确报错；源项训练和验证采用错开的网格避免常见切线列。

## 势表示与奇异积分

分解 `u=v+w`：先构造 `L_sigma v ~= f`，再令 `L_sigma w=0`，边界数据为 `g-v`。

- Poisson：`w=S mu+c`，联合两条边界求解，并附加总电荷
  `integral_(both boundaries) mu ds=0`。单层势使用 `-log(r)/(2*pi)`。
  不照搬单连通双层势，所以孔洞中的对数调和模式可以被表示。
- 修正 Helmholtz：`w=S_alpha mu`，核 `K0(alpha*r)/(2*pi)`。
- 振荡型 Helmholtz：`w=D_k mu+i*eta*S_k mu`，`eta=max(1,k)`。
  核 `G=i*H0^(1)(k*r)/4`，法向总是物理区域的外法向（内边界指向孔洞）。
  从物理区域取迹时使用 `-mu/2+K mu+i*eta*S mu`。
  该组合避免依赖一个裸单层／双层表示的辅助空腔共振，但不能消除物理区域本身的
  Dirichlet 共振。没有声称在真实特征频率上问题仍然唯一。

密度是物理面密度 `mu`，弧长 Jacobian 显式进入积分；不同于旧 panel_poisson 的
参数密度约定。本模块独立实现，没有改变旧接口。

同一面板内将核拆为 `A(t)*log|t-tau|+B(t)`，使用 Legendre 对数积分矩做乘积积分。
一般次数 B-spline 在 knot span 内的 Taylor 多项式用于稳定计算几何差商。
Helmholtz 的 J0/J1、修正 Helmholtz 的 I0 因子进入对数系数，不能直接沿用纯 Laplace
的常数对数系数。大参数时自动缩短面板，使局部核分裂不跨越过大的参数尺度。

非自身面板、跨内外边界和靠近曲线的体内目标，根据几何距离递归细分求积。
该细分不增加密度自由度。距离判据使用局部几何采样，不是严格包围盒误差认证；
独立提高求积阶数检查实际精度。极近／在边界的体内调用可能报错，应使用 `boundary()`。

## 一般源项：只有物理区域内的 f

`FourierSourcePlan` 在包围盒内的物理域点上采样 f，拟合固定 Fourier 字典。
不向用户请求孔洞或外部区域的 f，也不向求解器传入制造解的特解。

训练点还包含两条边界向域内偏移的两层采样，补上笛卡尔网格与曲线之间的
空隙，避免把近壁拟合变成外推；所有点均经过物理域包含性检查。验证点采用
不同的参数偏移、法向距离和网格，并随分辨率增加近壁探针数。

`padding` 是 Fourier 周期长度与包围盒边长之比，默认保留 2.0。
固定 `modes=m` 时，每方向最大角波数为 `2*pi*m/(padding*box_width)`。
高频源项可显式选择 `padding=1.5` 提高物理带宽，但更小的 padding
不是普遍更优：它也压缩了非周期函数的延拓余量，低频高精度拟合可能需要
更多模态。必须用独立源项验证选择分辨率，不能仅凭带宽或训练残差接受拟合。
统计信息包含物理带宽、体内与近壁训练点数。

SVD 以分解顺序应用，避免显式病态伪逆引起的消减误差；rcond 默认 1e-12。
验证采用另一组错开网格，以及内外边界附近的独立点。超出源项容差就报错，
不把源项误差混入边界求解成功的声明。源项验证是采样检查，不是连续误差上界。
稠密 Fourier extension 仍可能病态，必须记录保留秩、条件数、独立验证误差。

每个已拟合 Fourier 模式的特解由 `(|omega|^2+sigma)^(-1)` 构造。
Poisson 零频用 `-|x-center|^2/4`；盒频率恰好共振时用
`i*(omega dot (x-center))*exp(i*omega dot (x-center))/(2*|omega|^2)`。
这样不会把背景盒的共振误认为物理区域共振。近机器精度的零分母使用同一极限形式。

## 接口示例

```python
from bspf_models.elliptic.spline_annulus import (
    SplineAnnulus, FourierSourcePlan, adaptive_dirichlet,
)

# outer_curve / inner_curve 是原始 scipy BSpline；不需要体网格。
domain = SplineAnnulus(outer_curve, inner_curve)
source_plan = FourierSourcePlan(domain, modes=12, samples=56)

solution, history = adaptive_dirichlet(
    domain,
    sigma=-k**2,  # 或 0.0、+alpha**2
    boundary_data=(g_outer, g_inner),
    source=f,
    source_plan=source_plan,
    order=10,
    tolerance=1e-9,
    source_tolerance=1e-7,
    max_refinements=8,
)
assert history[-1]['converged']
u = solution.interior(points_in_domain)
```

固定几何/参数/面板上，`AnnulusPanelPlan` 缓存 LU，支持多个 RHS；源项 SVD 也可复用。
自适应根据两条边界上的独立 Gauss 点和近端点探针残差局部二分。
`converged` 仅说明该采样指标达到容差；达到细分上限时返回 False。
源项指标、边界指标和矩阵条件数必须分别看待，尤其在共振附近。

## 实测结果（CPU，未限制单线程）

以下为加入近壁源项训练点之前的历史基线。高频修复及新的独立验证见
[随机波 MMS 记录](../examples/pde/SPLINE_ANNULUS_TURBULENT_MMS.md)。

外曲线由 12 个具有三瓣径向扰动的控制点定义，内曲线由偏心椭圆状的 8 个控制点定义。
控制点之间的实际曲线是原始三次 B-spline，不是解析椭圆。三种算子使用相同几何。

齐次验证使用孔洞内的基本解源，叠加另一非零齐次解。固定面板、order=6/10/14
分别有 120/200/280 个未知量。独立边界最大归一化误差：

| sigma | order 6 | order 10 | order 14 |
|---|---:|---:|---:|
| 0 | 1.53e-6 | 1.04e-7 | 1.45e-8 |
| +4 | 3.58e-6 | 4.37e-7 | 6.11e-8 |
| -4 | 4.19e-6 | 8.62e-8 | 1.17e-8 |

三次 spline 的 C2 接缝会限制全局光滑性；不把这些结果称为全局指数收敛。
节点对齐升阶和局部细分共同提供高精度。

非齐次制造解：

`u=exp(.3*x+.2*y)+.2*sin(3*x)*cos(2*y)`，

求解器只接收其解析 f 和边界限制 g。源项字典每方向 25 个频率，共 625 列，
2275 个域内训练点，SVD 保留 557 列方向；独立源项相对最大误差分别为
1.91e-9、5.69e-10、2.04e-9。

自适应（每面板 10 点、目标采样指标 1e-9）的额外验证结果：

| 算子 | 边界未知量 | 独立边界误差 | 体内误差 | 近壁误差 |
|---|---:|---:|---:|---:|
| Poisson | 460 | 8.26e-10 | 5.94e-12 | 9.35e-11 |
| 修正 Helmholtz, alpha=2 | 480 | 6.58e-10 | 1.14e-11 | 3.05e-11 |
| 振荡 Helmholtz, k=2 | 820 | 1.09e-9 | 7.99e-14 | 7.41e-12 |

误差归一化为 `max(abs(error))/max(1,max(abs(exact)))`；体内是一组固定稀疏探针，
不是连续域最大误差证明。近壁探针距两条曲线分别为 1e-2、1e-4、1e-6。
额外边界点包括原始 knot 旁 1e-6 处；振荡问题的额外误差略高于采样停止指标，
明确保留这个差异，不宣称所有点都小于 1e-9。

## 共振诊断

在 k=3..7 扫描条件数，并局部搜索得到离散病态候选 k≈5.00146177。
order=8/10/14 的 reciprocal condition estimates 约为 5.99e-9/5.81e-9/5.64e-9。
这是离散条件数极小点，不是经过认证的连续本征值。

在该候选处，齐次制造解的体内误差分别约为 8.17e-6/8.82e-7/3.44e-8，
尽管训练残差一直约 1e-15。这直接说明小代数残差不能保证近共振内部精度。
系统在 rcond<1e-13 时拒绝求解；高于此阈值也不代表误差已得到保证。
本次没有实现完整特征值求解、共振处兼容条件或正则化物理模型。

## 验证与边界

27 项新旧面板测试通过，包括：双边界独立 Dirichlet、孔洞对数模式、三种核、
独立求积、距壁面 1e-6、间距约 6% 的窄环、反向与非整数参数化、五次 spline 与
knot insertion、一般源项仅在物理域调用、源项欠分辨拒绝、背景盒共振特解、
自适应限额报告，以及旧单边界回归。

当前存储/分解是稠密 O(N^2)/O(N^3)，源项拟合也是稠密 SVD。
没有运行大规模或高波数性能验收，没有梯度/通量输出接口，也没有任意不光滑源项保证。
不能把本版描述为成熟快速求解器、全局谱收敛方法或所有任意几何的认证求解器。

## 复现

```sh
MPLCONFIGDIR=/tmp/bspf-mpl python examples/pde/spline_annulus_convergence.py --adaptive --resonance-scan
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q packages/models/tests/test_spline_annulus.py packages/models/tests/test_panel_poisson.py
```

输出：`build/spline_annulus/results.json`、`adaptive.json`、`resonance_scan.json`、
`convergence.png`、`geometry.png` 和运行源码指纹。

方法背景：

- [复杂域上的函数延拓与体势/边界势分解](https://arxiv.org/abs/2211.14537)
- [多连通域的 PUX 延拓](https://arxiv.org/abs/1712.08461)
- [高阶 Nyström 对数奇异积分](https://amath.colorado.edu/faculty/martinss/Nystrom/2011_quadrature_for_nystrom.pdf)
- [参数相关核分裂的数值限制](https://link.springer.com/article/10.1007/s10444-022-09927-5)

本版的域内 Fourier 最小二乘不是上述 PUX 算法；没有借用论文对特定方法的收敛保证。


层势求值现在通过 `AnnulusPanelPlan.apply_layer(points, density, block_size=512)`
分块直接作用于密度。`interior`、`boundary` 和 GS 迭代不再构造或缓存完整的
“目标点 × 边界未知数”矩阵。远场先计算求积点上的加权密度，近场/自作用仍使用
原有高精度求积。边界系统的小型稠密矩阵及 LU 保留；当前分块实现运行于 CPU。
