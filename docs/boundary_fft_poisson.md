# 从等弧长 FFT 构造边界谱 Poisson 算法

当前原型：`examples/pde/boundary_fft_core.py`。
结果：`build/boundary_fft_core/results.json`。

这里把原先只用于 H^(3/2) 边界范数的 FFT，进一步用于物理边界算子的奇异主部。
方法与经典周期谱 Nyström / Kress 对数核分裂相连，不声称是一种新的边界积分理论。
它不依赖辅助矩形或外域平滑延拓。

## 几何与算子

设 Gamma 为简单、光滑、闭合曲线，s 是弧长，周长为 L，r0=L/(2*pi)。
原有 `ArcLengthBoundary` 使用高阶 Gauss 积分和标量反解，在数值容差内产生等弧长点；
不是声称 B-spline 的弧长反函数具有闭式解析表达式。

令

```
G(x,y) = -log(|x-y|/r0)/(2*pi),       -Delta_x G = delta_y,
S mu(s) = integral_Gamma G(gamma(s),gamma(t)) mu(t) dt.
```

首先考虑调和问题 -Delta u=0，u|Gamma=h。采用表示

```
u(x) = c + integral_Gamma G(x,y) mu(y) ds_y,
S mu + c = h,       integral_Gamma mu ds = 0.
```

mu 是辅助单层密度，并非一般意义下的物理法向通量。额外常数 c 与零总密度条件
一起处理二维单层算子的常数/对数容量问题；没有直接除以零频特征值。

对边界上的核作精确分裂：

```
G(gamma(s),gamma(t)) = G0(s-t) + R(s,t),
G0(delta) = -log(2*|sin(pi*delta/L)|)/(2*pi),
R(s,t) = -log(|gamma(s)-gamma(t)| /
              (2*r0*|sin(pi*(s-t)/L)|))/(2*pi).
```

因为 |gamma'(s)|=1，R 的对角极限为 0。圆的半径恰为 r0 时，R 处处为零。
一般曲线的 R 不必幅值小，也不一定是低秩矩阵；它的光滑性依赖几何光滑性。

令 S0 是 G0 的卷积算子，Rop 是 R 的积分算子。对弧长 Fourier 模态，

```
S0 exp(i*2*pi*k*s/L) = L/(4*pi*|k|) exp(i*2*pi*k*s/L),  k != 0,
S0(1) = 0.
```

等弧长采样使 S0 的离散作用可由 FFT 对角乘法准确处理，这同时给出对数奇异核的
周期乘积积分权重，而不是对无限大的对角核直接使用普通梯形求积。
Rop 则用等弧长梯形求积离散。原型保留稠密 Rop，以便独立检查结构。

## FFT 预条件

记 P 去除弧长平均值，T 在非零模态上乘 4*pi*|k|/L，在零模态上取 0。
设 mu=T nu，可以求解

```
(I + P Rop T) nu = P h,
mu = T nu,
c = mean(h - S mu).
```

方程自动使 nu 的均值为零。原型使用 GMRES，不使用截断 SVD。
圆上 Rop=0，方程退化为 nu=P h；这是实现中可严格检查的退化情形。
一般几何的迭代数不能仅凭这个形式保证统一上界。

第一版每次矩阵作用为 O(M^2) 的几何修正加 O(M log M) 的 FFT，存储 O(M^2)，
其中 M 是边界点数。几何预处理可复用。没有声称整体实现已达到 FFT 复杂度；
更大规模需要对几何算子和势求值单独加速。

## 非零体源：完整 Poisson 的设计

对 -Delta u=f、u|Gamma=g，在物理区域内定义体积势

```
v_f(x) = integral_Omega G(x,y) f(y) dy.
```

它满足 -Delta v_f=f。取 h=g-v_f|Gamma，使用上述边界求解器，再恢复

```
u(x) = v_f(x) + c + integral_Gamma G(x,y) mu(y) ds_y.
```

该过程不使用 Omega 外的 f 或 u，也不要求矩形上的周期延拓。
全局线性系统的未知量位于边界，但非零 f 的体积积分仍需计算，不能从成本中省略。
体积势的奇异与近奇异积分、边界单层势的近边界求值，是完整实现必需的独立模块。
它们可采用分段奇异求积、局部展开等方法；普通加密梯形求积不能保证任意贴边点的精度。

## 原型验证范围

当前代码只实现 f=0 的边界核心。测试函数为

```
u(x,y) = exp(3x) cos(3y) + 0.1 Re((x+i*y)^7),
```

它严格调和，只用边界函数值作为求解输入。测试圆、解析椭圆和既有凸三次 B-spline
边界，M=32,64,128,256,512。独立边界验证使用 2M 个错开的等弧长点，并采用核分裂
重新求值，区别于训练点残差。

内部验证点位于半径 0、0.15、0.30、0.45 的同心圆上，共 192 个点，远离边界。
内部单层势以 4M 点普通求积求值。原型**没有**验证任意贴边内部点，也没有集成非零体源。

三次周期 B-spline 在简单节点通常只有 C2 连续性。分段解析法向不等于全局解析几何。
因此全局 Fourier 离散在这条曲线上不应预期无限阶/指数收敛；要追求极高精度，需要
针对样条节点的有限光滑性设计分段离散或修正，而不是只继续加密同一个全局 FFT。

运行：

```sh
PYTHONPATH=jax/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python \
  examples/pde/boundary_fft_core.py
```

## 相关来源

- [Barnett: Boundary integral equations and high-order Nyström quadratures](https://math.dartmouth.edu/~fastdirect/notes/quadr.pdf)
- [Hao et al.: High-order accurate Nyström discretization of weakly singular kernels](https://arxiv.org/abs/1112.6262)
- [Barnett: Evaluation of layer potentials close to the boundary](https://arxiv.org/abs/1310.5352)

文中的 FFT 符号与投影预条件公式由上述具体 G0 分裂直接推导；原型的数值结果来自本仓库运行。
