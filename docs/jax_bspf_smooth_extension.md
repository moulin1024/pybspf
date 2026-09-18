# BSPF 未知场平滑延拓：解析 B-spline 法向原型

实现：`jax/src/bspf_jax/smooth_extension.py`。这是受
[IBSE 方法](https://arxiv.org/abs/1506.07561)启发的模态 Galerkin 原型，
并非原文 Fourier / 正则化 delta 离散的复现。当前结果没有超过直接强残差求解。

## 精确几何与未知导数

逆时针、正则闭合曲线 C(t)=(x(t),y(t)) 的外法向为

    n(t) = (y'(t), -x'(t)) / sqrt(x'(t)^2 + y'(t)^2).

`SplineDomain.normal(t)` 支持任意标量或数组参数。直接求原始 B-spline
导数，不经过折线、差分法向或网格插值。BSPF 基函数的解析梯度和 Hessian 给出

    B1 = nx Bx + ny By
    B2 = nx^2 Bxx + 2 nx ny Bxy + ny^2 Byy.

B2 是沿界面点固定法线方向的二阶导数。曲线不必星形，凹陷处同样适用。
这里的“解析”指基于解析样条导数的浮点计算，不代表无限精度。
当前周期三次几何通常只有 C2 连续；存在任意点法向不意味着曲线 C∞。

同时求 u 与外延场 ξ，匹配 Bj(u−ξ)=0，j=0,…,k，支持 k=0,1,2。
求解器仅接收域内 f 和边界 g，**没有输入 MMS 的真实法向导数或域内解**。
真实解及其导数仅供独立验证。

## 离散系统与原有框架复用

矩形背景为 [-1.2,1.2]^2，u、ξ 使用同一齐次 Dirichlet BSPF 背景空间；
这是外部辅助盒的条件，物理边界另行施加 u=g。
复用 `_stream_line`、`stream_evaluate_line`、`tensor_product` 和
`tensor_elliptic_solve`。端点使用 Chebyshev M=12、P=12。

质量归一化后的矩形刚度为对角 K，表示 −Δ 的弱算子。记

    F = BΩᵀ WΩ f
    E = K − BΩᵀ WΩ (−ΔBΩ)
    S = sqrt(WΓ) B0
    C = stack_j [sqrt(WΓ) h^j Bj]
    H = (I + ell^2 K)^(k+1), ell=0.25.

联立系统是

    K u − E ξ + Sᵀ λ = F
    H ξ + Cᵀ μ = 0
    C(ξ−u) = 0
    S u = sqrt(WΓ) g.

消去体场形成界面 Schur 系统，行列缩放后以 SVD 截断 1e-12 求解。
H 是背景弱 Laplacian 的谱多项式；此选择及其辅助盒边界条件属于本原型。
有限阶匹配不会自动恢复矩形问题的谱精度。

## 随机波 MMS 验证结果

两种域：凸闭合样条，以及带凹陷的非星形单连通样条。
固定随机种子 20260918，最高波数分别 4π、12π。N=49，体积分阶16，
每样条段界面积分阶12，二阶法向匹配。独立域内101×103错位网格，
独立界面每段20点。表中为独立点离散相对 L2 误差：

| 区域 | kmax | 二阶延拓 | 同空间直接强残差控制 |
|---|---:|---:|---:|
| 凸域 | 4π | 2.665e-4 | 4.796e-10 |
| 凸域 | 12π | 6.520e-3 | 7.201e-6 |
| 非星形域 | 4π | 1.023e-3 | 1.809e-9 |
| 非星形域 | 12π | 1.763e-2 | 4.709e-6 |

控制组使用相同背景空间、体积及边界采样，直接最小化加权 PDE 与
Dirichlet 残差，列缩放后 SVD 截断1e-14。它与延拓系统的目标、缩放和
截断不同；此比较检验空间表达能力，并非仅改变一个参数的消融。

非星形高波数组在独立界面上的 u−ξ、∂n(u−ξ)、∂nn(u−ξ) RMS 分别为
1.35e-6、3.56e-6、8.31e-4，但 ∂n u 相对真实导数的绝对 RMS 仍为2.47。
**两未知场彼此匹配好，不等于物理解准确。**
同组 Schur 保留秩967/1536；凸域516/576，存在明显界面冗余/数值秩损失。
加密界面采样显著减小匹配误差，但没有一致改善体场误差。
N=33/49及较稀界面结果也保存在对应 build 目录，尚未证明网格收敛率。

因此解析法向已经被利用，几何差分误差被避免；目前仍需研究体场离散的
一致性、延拓阶数与界面乘子表示/条件数，不能将当前误差归咎于法向估计，
也不能宣称已达到矩形精度。下一步宜先做固定空间下的 Schur 截断和界面
密度基函数实验，再做体积分及网格收敛，分离这些误差来源。

## 复现与输出

```sh
export PYTHONPATH=jax/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python examples/pde/embedded_poisson_smooth_extension.py --nodes 49 --boundary-order 6 --out build/embedded_poisson_smooth_extension_n49
python examples/pde/embedded_poisson_smooth_extension.py --nodes 49 --boundary-order 12 --matching-orders 1 2 --reuse-cache --geometry-cache build/embedded_poisson_smooth_extension_n49 --out build/embedded_poisson_smooth_extension_dense
python examples/pde/embedded_poisson_smooth_extension_control.py
MPLCONFIGDIR=/tmp/bspf-mpl python examples/pde/render_embedded_poisson_smooth_extension.py
```

缓存是本地产生的 pickle，仅加载可信缓存。图、NPZ 和 JSON 在
`build/embedded_poisson_smooth_extension_dense/`。build 为忽略目录。
法向一/二阶算子独立差分验证、加权迹伴随验证、原有张量逆验证，连同
两域积分矩验证共5项通过。

## Fourier extension / continuation 文献对下一版的启发

这里应区分两条路线：

1. [Matthysen–Huybrechs, Fourier extension frames](https://arxiv.org/abs/1706.04848)：
   将包围盒上的 Fourier 函数限制到物理域，形成冗余表示，通过过采样最小二乘
   和正则化 SVD 求系数。小奇异值本身不证明物理场无法高精度恢复。
   [Adcock 等的稳定性分析](https://arxiv.org/abs/1206.4111)讨论了这一现象。
   当前 BSPF 直接强残差控制在构造思想上更接近这条路线。
   [Chen–Lin 的椭圆 PDE 配点工作](https://arxiv.org/abs/2211.06251)
   也使用 Fourier extension 过采样，并报告精度平台；因此不能从函数逼近
   的稳定性结论直接保证 BSPF PDE 离散的精度。
2. [Bruno–Paul, Two-dimensional Fourier Continuation](https://arxiv.org/abs/2010.03901)：
   在边界法向线上使用若干内侧样本进行一维高阶平滑归零延拓，再高阶转移到
   Cartesian 网格。其 Poisson 示例延拓右端，计算特解后加边界积分调和修正。
   这不是用已知 Dirichlet 数据直接得到未知解全部法向导数。

针对用户提出的解析 B-spline 法向，第二条路线给出更直接的几何用途：

    X(t,r) = C(t) + r n(t).

在负 r 的一小条内侧带采样，以多点一维延拓代替逐阶估计高阶法向导数。
窄带必须同时受曲率和非相邻边界间距限制，尤其是 C 形凹口；精确法向
不保证所有法线在任意延拓距离内不相交。三次样条的有限几何正则性仍需保留。

**尚未实现的 BSPF 适配建议**：预构造线性延拓矩阵 E，使矩形场 v=E q；
q 为域内未知量，用 PDE 与真实边界条件求 q，而非提前知道 q 后才能延拓。
示意系统为 [RΩ(−Δ)E; sqrt(tau) TΓE] q=[f; sqrt(tau)g]，以过采样和
正则化求解。RΩ 为域内采样，TΓ 为精确边界取值。对于从系数直接表示的
BSPF 场，法线内侧样本可直接由基函数求值，避免额外低阶插值。
需验证离散 E 与 BSPF 微分/矩形椭圆预条件器的一致性，不能默认现有对角
Poisson 逆就是整个嵌入系统的精确逆。

首先用已知随机波检验 E 的独立值、梯度、Laplacian 误差和网格收敛，再将
E 接入未知解系统。另一条更短的实验路线是继续改进已有 BSPF frame 式
强残差法，系统扫描过采样率、外盒大小、SVD 截断及系数/外域范数。
两条路线均不需要把未知 Neumann 数据作为额外已知条件。
