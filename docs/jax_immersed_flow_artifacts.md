# 固定网格的绕流伪影诊断与 AAA–lightning 参考解

结论：原方案的很小壁面残差没有转化成很小的内场误差。
固定 73×33 背景网格后，实际 Stokes 解几乎达到其约束空间的最佳 H¹ 速度逼近；
主要误差来自当前的近似空间与孔壁表示。提高求解容差不能消除这一误差。
解析壁面因子能改善结果，但目前仍不能声称伪影已消除或达到近谱精度。

代码：

- `packages/models/src/bspf_models/fluids/immersed_flow.py`：原 SVD 方法及可选解析壁面因子。
- `examples/pde/lightning_stokes_reference.py`：独立 Goursat/AAA–lightning 参考实现。
- `examples/pde/compare_immersed_stokes.py`：固定空间的实际解/最佳逼近对照。
- `examples/pde/render_immersed_stokes.py`、`render_immersed_artifacts.py`：误差图与 NS 对照。

## 参考问题与参考解

采用同一个矩形 `[-1,5]×[-1,1]`、同一个偏心椭圆
`center=(0.19,-0.13), axes=(0.31,0.23)`。两种方法均解无体力、稳态 Stokes：

- 左端 `u=1-y², v=0`；上下壁面与椭圆上无滑移。
- 右端是原有的 Laplacian 零牵引 `(nu*u_x-p, nu*v_x)=0`，没有改为指定出口速度。
- **两边均关闭缓冲力**，保留原计算盒子。空间变系数缓冲项不属于齐次 Stokes，
  不能直接用 Goursat 齐次解作为该带缓冲问题的参考。
- BSPF 仍是 73×33，积分因子 2.5；没有网格加密、额外人工黏性或结果滤波。

参考方法遵循 Xue、Waters、Trefethen 的
[Computation of Two-Dimensional Stokes Flows via Lightning and AAA Rational Approximation](https://doi.org/10.1137/23M1576876)
及[作者公开 LARS 代码](https://github.com/YidanXue/LARS)。本仓库是独立的基准实现，
不是官方 LARS 软件包。使用

\[
 \psi=\operatorname{Im}(\bar z f(z)+g(z)),\quad
 u+iv=z\overline{f'(z)}-f(z)+\overline{g'(z)},\quad
 p/\nu=4\operatorname{Re}f'(z),\quad
 \omega=-4\operatorname{Im}f'(z).
\]

复坐标以孔心为原点。基底包括多项式、孔心 Laurent 项、17 个由椭圆 Schwarz
函数 AAA 近似选取的孔内极点、矩形角点外侧的 lightning 极点，以及保证速度单值的
对数共轭项。每个基函数在流体内部满足齐次 Stokes；边界用超定最小二乘拟合。
使用二次重正交 Arnoldi 递推，并解析求一、二阶导数以施加出口牵引。

这里的混合速度/牵引条件采用未加权的边界拟合；训练点的角点聚集尺度随最小极点
距离调整。初试中固定 tanh 聚集尺度配合距离权重会掩盖角点附近误差，未作为最终参考。
每级均在另一组 997 点/边的独立点上检查边界。提高的是**参考解的有理逼近阶数**，
不是 BSPF 网格。

|多项式阶数|每角极点数|Laurent 阶数|实未知量|独立速度边界最大误差|独立牵引/ν最大误差|
|---:|---:|---:|---:|---:|---:|
|48|16|24|620|5.849e-7|8.787e-8|
|72|24|36|892|2.132e-9|3.528e-10|
|96|32|48|1164|8.065e-12|6.186e-12|
|120|40|60|1436|3.810e-12|3.791e-12|

96→120 阶在共同 401×161 验证网格上：速度最大差 **8.520e-12**，
涡量最大差 **2.276e-9**。因此使用 96 阶解评估下列百分量级 BSPF 误差已经足够。
四个精确角点不纳入点值比较：混合边界交点没有统一的牵引/涡量点值条件。
独立边界检查仍包含非常靠近角点的点，没有删掉一个有限宽度的角点邻域。

完整数据：`build/immersed_flow/lightning/convergence.json`。
程序的参考压力是 `p/nu`，不能不乘黏度就与物理压力比较。

## 同网格实测

下表 L² 误差在相同的独立 401×161 全流体验证网格上计算；H¹ 使用相同物理积分。
“最佳 H¹”是对参考速度做相同离散 H¹ 内积下的最小化投影，包含速度及全部一阶导数，
并使用与实际求解完全相同的提升和约束空间。它不是用离散算子反推的 MMS。

|方法|速度相对 L²|涡量相对 L²|实际速度相对 H¹|最佳速度相对 H¹|独立孔壁最大速度|
|---|---:|---:|---:|---:|---:|
|原 SVD，rcond=1e-10|0.4123%|5.6790%|5.067566%|5.067552%|1.828e-12|
|SVD，rcond=1e-5|0.3264%|4.8288%|4.301701%|4.301690%|6.415e-9|
|解析壁面因子|0.1781%|2.9113%|2.583948%|2.583943%|6.363e-16|

原 SVD 保留 127 个约束方向，放宽阈值后为 92 个。
过紧截断造成了额外误差，但仅放宽它不足以解决问题。
实际 Stokes 解与最佳 H¹ 投影的误差几乎相同，说明主要瓶颈在当前空间，
不能将其归因于时间推进，也不能期望只调线性求解容差或画图方式来修复。
这不构成“任何 73×33 BSPF 方法都只能达到这个精度”的结论。

数据与图：`build/immersed_flow/stokes_comparison/`。
误差图两种 BSPF 方法使用同一色标，展示 `-1<=x<=2` 的孔壁附近视图；
上表误差仍覆盖整个流体区域。

## 可选解析壁面因子

令椭圆水平集为 F，定义

\[
 F=\frac{(x-c_x)^2}{a^2}+\frac{(y-c_y)^2}{b^2}-1,\qquad
 G=\frac{(x-L)(R-x)(1-y^2/h^2)}{(c_x-L)(R-c_x)(1-c_y^2/h^2)},
\]
\[
 q=\frac{F^2}{F^2+\beta^2G^2},\qquad
 \psi=q(\psi_0+B c)+(1-q)c_h,\qquad \beta=2.
\]

孔壁上 `q=0, grad(q)=0`，所以 `psi=c_h, grad(psi)=0`；矩形边界上
`q=1, grad(q)=0`，保留原有外边界处理。`c_h` 是求解中的未知数，不能任意固定。
分母在整个实矩形内严格为正。代码使用解析一、二阶导数和完整乘积法则，
对修改后的速度基底重新组装质量、黏性、对流和缓冲项。
这不是求解后乘遮罩或对数值解做平滑滤波。

这是**改变近似空间**，不再是纯全盒 BSPF 展开：保留原矩形 BSPF 基底与张量求值，
再乘几何因子。背景节点相同，但有效自由度不同：原方案 1932，解析因子方案 2060
（原矩形 2059 个系数，另加一个孔壁常数）。`wall_width` 是近似空间参数，不是物理壁厚。

用 `wall_method="factor"` / `--wall-method factor` 启用；默认 `svd` 保留用于原方案复现。
它的改进已经验证，但表中 2.91% 涡量误差说明其仍不是最终的近谱精度方案。

## NS 回归与误差边界

同样的 Re=20、dt=0.02、t=20、缓冲长度 2、强度 3、73×33 上，解析壁面因子：

- 独立孔壁最大速度 `1.036e-15`。
- 八截面最大相对流量误差 `1.266e-14`。
- 固定独立网格上的稳态涡量输运残差估计 RMS 从 11.22 降到 3.324。
  此项是六阶中心差分的后验估计，在物理区固定的内点子集上计算，排除壁面邻域，
  不能当作精确 PDE 残差，也不是 NS 解相对 AAA 的误差。

NS 图条纹减轻，但不能把齐次 Stokes 参考精度直接转移到非线性、非定常 NS。
结果在 `build/immersed_flow/factor/`；三种方法同色标图在
`build/immersed_flow/artifact_diagnosis/ns_fixed_grid.png`。

新增连续 NS MMS 包含解析几何因子、非零孔壁常数、独立自动微分体力及出口牵引。
33×25 固定空间下，将积分因子 2.5→4，弱残差 9.664e-5→3.769e-7，
场表示误差约 1e-14，确认积分仍须独立验证。
这项积分检查未改变未知量网格，也未以额外积分点代替求解自由度。

14 项测试通过：原绕流/cavity、解析几何导数、连续 NS MMS、孔壁/通量，及
有理参考解的独立边界和有限差分 Stokes/散度/牵引导数检查。

## 对进一步算法设计的含义

AAA–lightning 用孔内极点、Laurent 和对数项表示边界引起的解结构，不强迫这些
结构成为孔内正则的全盒 BSPF 场。后者要先选取优质延拓，再在有限全局空间中近似；
很小的壁面拟合残差本身不保证这种延拓适合该空间。

下一步应以该参考解为验收标准，研究 BSPF 体内空间加几何适配的有理边界修正空间，
而不是继续调整一项 SVD 阈值。对非定常 NS 必须把新增基底一致地纳入质量、黏性和
对流算子；不能把稳态齐次 Stokes 修正直接当成时间离散后 Helmholtz–Stokes 的齐次解。
后续已实现固定几何的混合空间原型，并用连续非定常 NS MMS 验证完整时间项和
非线性耦合，见 [BSPF 与有理边界修正](jax_hybrid_rational_flow.md)。
原长时绕流的伪影消除仍需单独验收。

## 复现

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python examples/pde/lightning_stokes_reference.py
python examples/pde/compare_immersed_stokes.py
MPLCONFIGDIR=/tmp/bspf-mpl python examples/pde/render_immersed_stokes.py
python examples/pde/immersed_channel_flow.py --wall-method factor --out build/immersed_flow/factor
MPLCONFIGDIR=/tmp/bspf-mpl python examples/pde/render_immersed_flow.py --out build/immersed_flow/factor
python -m pytest -q packages/models/tests/test_immersed_flow.py packages/models/tests/test_cavity.py packages/models/tests/test_lightning_stokes_reference.py
```

依赖 SciPy 的 `scipy.interpolate.AAA`（本机 SciPy 1.17.1）。无需 MATLAB，
也不需要运行时下载作者代码。公开参考代码仅用于公式核对，下载副本放在 build 下。
