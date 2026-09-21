# 方腔中心磁能释放：二维不可压电阻 MHD

本例从 BSPF 不可压 NS 方腔扩展到双向耦合的 MHD：洛伦兹力反馈到流体，
流体同时输运磁通；黏性和电阻均为正。磁能来自给定的初始磁场，无持续
磁场源项，也不施加中心机械体力或热源。密度、磁导率归一化为 1。

## 方程与边界

\[
u=(\psi_y,-\psi_x),\quad B=(A_y,-A_x),\quad j=-\Delta A,
\]
\[
u_t+(u\cdot\nabla)u=-\nabla p+\nu\Delta u+j\nabla A,
\qquad A_t+u\cdot\nabla A=\eta\Delta A.
\]

两种场均通过旋度表示，物理散度为零。流函数检验消去压力。
速度边界沿用正则化顶盖方腔，顶盖峰值速度为 1；其余三壁静止无滑移。
磁通函数在四壁取相同常数 `A=0`，对应无穿壁磁通 `B_n=0` 及零切向电场。
磁通函数的法向导数不固定，允许切向磁场。
这是初始无穿壁磁通的理想导电壁模型，不是绝缘外部真空模型。
导电壁电磁条件参考 [Datta & Crews](https://faculty.washington.edu/shumlak/WARPX/html/mhd_bc.pdf)。

## 初始磁场

\[
A_0=C\,256x^2(1-x)^2y^2(1-y)^2\sum_{x_c\in\{0.37,0.63\}}
\exp[-((x-x_c)/0.10)^2-((y-0.5)/0.14)^2].
\]

将它投影到磁通 BSPF 空间，再归一化，使积分点上的最大磁场强度为指定值
（默认 6）。磁场以 Alfvén 速度为单位，因而 6 表示最大初始 Alfvén 速度
为顶盖峰值速度的六倍，不预先保证流体速度达到这个值。

这是两个同号磁通岛组成的局部非平衡初态，核心电流同向。它借鉴磁岛合并
激发电流层和流动的思路，但**不是周期 Fadeev 平衡或标准岛合并基准**。
参见 [Adler 等的不可压电阻 MHD 岛合并研究](https://jadler.math.tufts.edu/publications/files/2013_pFOSLSMHD_SISC.pdf)。
单个轴对称磁团的洛伦兹力可能完全被不可压压力抵消；采用双岛和椭圆结构，
是为了产生非零的无散度洛伦兹力分量，而不是仅增加磁压。

初始磁场的给定视为 `t=0` 前已完成的能量储存；程序不模拟实际注入线圈或
磁场建立过程。二维不可压模型能模拟磁张力驱动的喷流和涡旋，但不描述
声波、压缩激波或热爆炸。

## 离散与能量检查

速度使用原先值及法向导数齐次的 BSPF 流函数基；磁通使用只限制值的 BSPF 基。
磁通基的质量矩阵为单位矩阵，梯度刚度对角化，记为 `K`。
定义混合 Galerkin 电流 `j_h=M_A^{-1}K A`，洛伦兹力和感应输运采用同一组
过积分点及这个电流，从而逐项满足

\[
\langle u,j_h\nabla A\rangle+
\langle j_h,-u\cdot\nabla A\rangle=0.
\]

不能把速度和磁场各自的离散随意拼接，否则能量交换可能产生虚假增益。
诊断同时给出混合电流与强导数 `-Delta A` 的相对 L2 差异；该差异需要
随空间加密检查。零散度和能量交换正确并不意味着薄电流层已充分分辨。

时间推进复用二阶中点预测/校正，黏性与电阻采用隐式处理。对流及 Alfvén
传播显式耦合，因此仍然有时间步限制。动量隐式矩阵仍是中小网格用的稠密
Cholesky；磁通扩散为对角直接解。

理想的机械—磁能收支为

\[
\frac{d}{dt}(E_K+E_B)=P_{lid}
-\nu\int|\nabla u|^2\,d\Omega-\eta\int j_h^2\,d\Omega.
\]

`P_lid` 从顶壁速度和法向剪切导数独立计算，不由能量差倒算。保存时刻之间
采用梯形积分，因此累计收支残差还包含采样积分误差。静止壁面下的半离散
总能量恒等式在测试中单独检查。黏性和电阻损失不转存为热能，模型没有温度方程。

## 运行

```bash
# 独立运行：初始流体静止，顶盖随后平滑启动。
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python examples/pde/mhd_cavity_2d.py --n 49 --T 0.5

# 接续前一轮已计算的稳态方腔，瞬时设置初始磁场。
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python examples/pde/mhd_cavity_2d.py --n 49 \
  --background build/cavity_n49/state.npz --out build/mhd_cavity

MPLCONFIGDIR=/tmp/bspf-mpl python examples/pde/render_mhd_cavity.py build/mhd_cavity

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m pytest packages/models/tests/test_mhd_cavity.py -q
```

默认 `Re=100, Rm=200, dt=0.0005`，所以 `nu=0.01, eta=0.005`。
Re、Rm 都以单位长度和参考速度 1 定义；Rm 不是基于初始磁场的 Lundquist 数。
背景状态的网格、Re 和启动时间必须匹配。
无磁场 NS 对照组从完全相同的速度初值同步推进，使用完全相同的壁面驱动。

输出 `evolution.npz` 包含每帧实际计算的速度、磁场、磁通、电流和对照速度；
`summary.json` 包含收支与约束诊断。渲染器产生 `mhd_cavity.png` 和
`mhd_cavity.mp4`，全程固定颜色范围，箭头为速度、等值线为磁通。
需要 Matplotlib 和 imageio-ffmpeg；可用 `--no-movie` 只生成图片。

中心区域定义为 `[0.25,0.75]^2`，诊断分别报告该区域的最大速度、无磁场对照
最大速度，以及同一点速度差的最大模，不把这三者混为一个量。

## 本次结果与精度界限

从前一轮 `t=30` 的稳态方腔启动，49² 网格，磁场峰值 6，运行至释放后 `t=0.5`。
中心形成向上、向下的局部高速流动，周围出现回流；随后黏性和电阻耗散使运动衰减。

| 指标 | 本次测量 |
|---|---:|
| 中心区域最大速度峰值 | 3.2093（t=0.070） |
| 同时刻无磁场 NS 对照最大速度 | 约 0.2716 |
| 初始 / 峰值动能 | 0.0188249 / 0.146950 |
| 初始磁能 | 1.258115 |
| 全程最大速度散度 | 7.11e-14 |
| 全程最大磁场散度 | 1.28e-13 |
| 全程最大穿壁磁场 | 1.16e-15 |
| 最大累计能量收支残差 / 初始总能量 | 2.41e-4 |

动画截图选择“相对对照的速度变化”最大的 `t=0.075`，该帧中心最大速度
为 3.1390；它和中心最大速度本身的峰值时刻略有区别。

无磁场退化、导电壁、逐点无散度、闭壁磁能—动能交换、独立解析正弦磁扩散、
磁压梯度平衡、二阶时间加密等 MHD 测试全部通过。连同方腔及原流函数固定/
开放边界回归，本次共 **18 passed**。

额外比较：

- **时间**：49² 网格将 dt 减半为 0.00025，比较至 `t=0.15`，同网格、同保存时刻的
  最大速度向量差为 0.001194，最大磁场向量差为 0.001349。峰值中心速度为 3.20897。
  保存间隔也减半后，累计能量收支相对残差降至 5.99e-5；这里同时改善了时间推进
  和收支诊断积分，因此不能把全部下降归因于时间格式。
- **空间**：41² 与 49² 的峰值中心速度分别为 3.17791 和 3.20931，差约 1%。
  两者精确共有的 9×9 节点上，全时间最大速度向量差为 0.1797、磁场差为 0.5783。
  磁能最大差为 0.00464、动能最大差为 0.00596。
- 混合电流与强导数电流的最大相对 L2 差异由 41² 的 6.91% 降到 49² 的 3.80%。

**这是一组已验证基本耦合、并呈现明确中心加速的探索算例；薄电流层细节尚未达到
精细网格收敛。** 不应从当前 49² 动画提取精确重联率、最小电流层厚度或高 Rm 标度律。

复现对比（相同背景 Re、同为原方腔 `t=30` 状态）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python examples/pde/mhd_cavity_2d.py --n 41 --background build/cavity_2d/state.npz \
  --out build/mhd_cavity_n41
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python examples/pde/mhd_cavity_2d.py --background build/cavity_n49/state.npz \
  --T 0.15 --dt 0.00025 --sample 0.0025 --out build/mhd_cavity_half_dt
python examples/pde/mhd_cavity_compare.py build/mhd_cavity_n41 build/mhd_cavity \
  --out build/mhd_cavity/spatial_comparison.json
python examples/pde/mhd_cavity_compare.py build/mhd_cavity build/mhd_cavity_half_dt \
  --out build/mhd_cavity/time_comparison.json
```
