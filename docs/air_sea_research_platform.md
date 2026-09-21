# 二维海气研究平台 v1

本平台研究水平二维、固定层厚、垂直平均的理想化双环流、温湿度输运和海气交换反馈。它不是三维海洋大气模式，也不从层平均状态恢复真实近海面廓线。30 天是可靠性与调整过程试验，不是气候平衡证明。

## 安装与统一入口

在仓库根目录，使用支持 float64 的 JAX 环境：

```bash
python -m pip install -e './jax[air-sea,test]'
bspf-air-sea run --config experiments/air_sea/coare35_24h.json --out build/my_coare24h
bspf-air-sea report build/my_coare24h > build/my_coare24h_report.json
bspf-air-sea resume build/my_coare24h/checkpoints/step000000144.json --until 172800
bspf-air-sea validate --suite smoke --config experiments/air_sea/coare35_24h.json --out build/my_smoke
```

`until` 单位是从初始时刻起的秒数，必须与同步分组精确对齐。实验目录必须不存在。续跑读检查点中的完整原配置；更换参数、网格、时间方法或软件环境需要新建派生实验。输出 JSON 配置不能作为续跑时覆盖检查点参数的入口。

Python API 位于 `bspf_jax.air_sea_platform`：`run(config, out, until=None)`、`resume(checkpoint, until=None)`、`validate(suite, config, out)`、`report(run_directory)`。报告读取已有文件，不改变原实验。示例 `examples/pde/air_sea_double_gyre.py` 是参数兼容入口；原历史驱动保存在 `air_sea_double_gyre_legacy.py`。

## 方程、边界与单位

所有配置采用 SI 单位。计算坐标归一化到 [0,1]²，物理边长为 `length`；梯度乘以 1/length，扩散乘以 1/length²。温度系数存储相对 `reference_temperature` 的扰动，交换核和输出使用 K；比湿 q 为 kg/kg。速度为 m/s，流函数为 m²/s。

海洋无散水平速度由 BSPF 流函数表示，四边不可穿透、无滑移。大气为 Fourier x 周期通道，y 不可穿透、自由滑移，另有独立的 x 向平均风模态。标量使用相应的无通量边界空间；没有出入盆地的平流通量。速度满足投影后的二维动量方程：非线性平流、β 平面旋转、黏性、界面应力；海洋另有线性底摩擦，大气另有风恢复。

令 `M_o=rho_ocean*ocean_depth`、`M_a=rho_air*atmosphere_depth`、`C_o=rho_ocean*cp_ocean*mixed_layer_depth`、`C_a=M_a*cp_air`。动量所用海洋深度与温度所用混合层深度不同，均固定。界面应力 τ 指向海洋，H、E 分别是向上的显热和水质量通量。

- 海洋动量界面倾向：+τ/M_o；大气：−τ/M_a，随后投影到各自速度空间。
- 海洋温度：平流 + 扩散 + (R−H−Lv E)/C_o。
- 大气温度：平流 + 扩散 + H/C_a + 温度恢复。
- 大气比湿：平流 + 扩散 + E/M_a + 湿度恢复。
- 诊断海洋水库：dW_o/dt=−〈E〉，不反馈层厚或盐度。

外部恢复的目标场、初值以及所有参数由 `air_sea.py` 中的 `plan_air_sea`、`initial_air_sea_state` 与版本化 JSON 明确给出。风恢复是维持双环流驱动的外部动量/能量源；温湿恢复及辐射是外部热水源汇，不属于守恒的内部交换。

## 表面代理与 COARE 3.5

`surface.method=constant` 保留历史常系数公式。`coare35` 使用 NOAA 官方参考源码的动量、热量、水汽交换核心；参考提交 `bd1cac80d2dae454e699f9ef46204dfd88086a71`，源码、许可证、SHA256 清单位于 `jax/tests/reference/coare35/`，不会自动追踪上游。

原始参考：[NOAA-PSL COARE-algorithm](https://github.com/NOAA-PSL/COARE-algorithm/tree/bd1cac80d2dae454e699f9ef46204dfd88086a71)。参考核的官方常数与适配到模型 `rho_air/cp_air/Lv` 的通量分开验证。

本模型将层平均风、温、比湿直接作为指定参考高度的有效状态，混合层温度作为有效表皮温度；默认三个高度均为 10 m，边界层高度为固定大气层厚。关闭冷皮肤、暖层、雨与外部波浪输入。保留 COARE 海水蒸气压 0.98 修正，但没有海洋盐度状态。比湿到官方 RH 输入的转换仅用于独立参考对照；模型核直接使用比湿。

使用海气相对速度；矢量应力计算不除以相对风模长，因此零相对风应力为零而阵风仍可驱动标量交换。官方十次固定迭代和初始 zetu>50 的第一迭代分支原样保留（包括极弱风下触发该分支的行为）。`iteration_change` 是末次迭代变化诊断，不是额外收敛承诺。公式本身的分段和非整数幂保留。没有隐藏中性回退或结果截断。`stability=false` 只关闭稳定度修正函数，其他粗糙度/阵风机制保留，用于机制对照。

## 时间推进与完整账本

生产运行只接受 MRI-GARK ERK45a 四阶耦合，双方状态在每个快阶段共同进入交换计算，使用同一通量与同一阶段权重更新双方及账本。历史一阶耦合仍在原数值 API/legacy 驱动中，用于对照，不通过生产 runner 隐式启用。

`air_sea_audit.PROCESSES/QUANTITIES` 定义 18 个过程、22 项累计量。包括两侧显热、水汽潜热、空气水量和海洋水库、两侧水平动量及约束反力、两侧动能、三个标量半方差、累计绝对热/水/功、界面相对运动耗散。

总热残差用海洋显热+大气显热+水汽潜热的变化减去明确外部热输入；内部输运和界面不闭合单独显示为 `internal_heat_leak`。总水残差用空气水量+诊断海洋水库变化减去湿度恢复输入。动能预算保留有限步长缺陷，不能要求非线性动能机器精度守恒。约束反力是物理力与可容许速度投影后的动量变化之差，不宣称独立恢复了壁面压强分布。黏性、底摩擦及相对滑动耗散不回填为热。

## 有效性、存储与重启

每个 RK/MRI 阶段及微步组合终点检查非有限值、节点/积分点负比湿、q≥1、非正绝对温度和交换核有效性；保留首个失败阶段状态、物理时间、事件序号、字段索引和位置。JIT 同步分组内保留首次异常证据，主机在该组返回后终止，不把失败组写成有效结果。每个同步组检查预算和梯度能谱尾。RH>100% 报告但不移除，本模型没有大气内部凝结；界面水通量仍允许负值（逆向交换）。

- `config.json`：六组版本化配置；`manifest.json`：Git 提交、差异指纹、依赖、后端、精度、实际源码校验和及参考闭合版本。
- `diagnostics.jsonl`：逐时预算、方差残差、极值、梯度、通量和交换诊断、谱尾等。
- `fields/step*.nc`：逐帧 NetCDF，坐标/单位/正号/代理说明，温湿度和速度、应力/显热与潜热合计通量、交换系数和稳定度等空间分布。
- `checkpoints/step*.npz` + `.json`：完整状态、账本、初始预算、时间步、配置与来源；写临时文件后原子替换，JSON 为完整性提交标记，SHA256 校验。首版保留所有检查点，至少保留最近两个有效版本。
- `latest_checkpoint.json`：最新完整检查点的相对路径。
- `failure/`：首次异常现场及错误；`status.json`：运行状态与本次调用耗时。

正常 SIGINT/SIGTERM 请求在同步组边界保存检查点。回退到旧检查点时，晚于检查点的输出被移入 rollback 文件，避免混入重复时间记录。旧 `state.npz` 缺少账本，不能作为精确续跑输入。相同软件环境及步序要求连续/续跑结果一致；不要求跨硬件逐位一致。

## 自动验证与发布门槛

```bash
python -m pytest -q jax/tests/test_air_sea.py jax/tests/test_multirate.py jax/tests/test_surface_exchange.py jax/tests/test_air_sea_audit.py jax/tests/test_air_sea_platform.py jax/tests/test_air_sea_validation.py
bspf-air-sea validate --suite all --config experiments/air_sea/coare35_24h.json --out build/full_acceptance
```

`all` 包含常系数和 COARE 各自的空间/时间/积分矩阵、参考高度与弱风/稳定/中性/闭合源汇试验，以及两种闭合的 30 天试验（在第 1 天实际暂停/续跑）。单独的 `resolution/quadrature/sensitivity/reliability` 可分批运行。失败证据不会被覆盖；同配置、同源码的未完成有效运行可以继续。

空间 n=33/49/65/81，统一 H=150 s、快步30 s，6/12/24 h 检查；n33/n81 另做 H=600/300 s。积分采用固定 n81、H150 的32/40/48，并用 n65、48 点作为独立空间误差参照。不同积分方案的正交基需先做同一试验空间的系数变换，不能直接复用系数。共同过积分网格比较原始场和解析梯度，不用绘图插值估计误差。

门槛由配置和测试固定：空间最细相邻 RMS<1%、峰值差<5%且递减；时间/积分差各小于空间差10%；光滑制造解三个有效区间阶数3.6–4.4；参考核相对1e−6，应力绝对1e−8 Pa、热通量绝对1e−5 W/m²；热残差1e−3+1e−10累计绝对热通量，水1e−9+1e−10累计绝对水通量，动能1e−6+1e−5累计绝对功（均按单位面积）。浮点噪声区不拟合虚假阶数。分段转折附近不预设四阶。

JAX 专用 CI 为 `.github/workflows/air-sea.yml`；普通提交运行短测试，长实验手动触发。提供自动化入口不等于实际通过发布验收；查看相应实验目录的 `validation.json` 与失败记录。任何未完成或失败的长实验都阻止宣称全部验收通过。

## 当前限制

无垂直动力、凝结/降水、盐度、自由面、球面、真实边界层廓线或机械耗散热回填。有限解析度、表面代理高度、外部恢复强度和闭合适用条件均可影响结果；默认 n81 不是所有参数组合都已解析的保证。没有保正限制器、额外滤波或谱尾失稳后的自动修补。默认30天两组的执行成本较高，需要保留完整输出和计算时间记录。物理真实性、SOTA 或长期统计平衡需要额外独立证据。

## 报告与动画

```bash
python examples/pde/report_air_sea_platform.py build/my_coare24h --out build/my_coare24h_report
python examples/pde/render_air_sea_mp4.py --run build/my_coare24h --frames 48 --out build/my_coare24h.mp4
```

报告写到独立目录，不修改原实验；动画读取已保存的实际时刻，COARE 热通量直接取 NetCDF 中的模型输出，不重套常系数公式。固定色标，无时间插值。

分段专项可独立执行：

```bash
python examples/pde/validate_air_sea_surface_branches.py --out build/coare_branch_validation
```

它积分规定的弱风反转及稳定/不稳定转换通量历史，与自适应独立积分比较；仅报告阶数，不将不光滑分支强行判定为四阶。完整耦合的光滑分支四阶性质由短测试另行验证。

本机依赖版本锁定在 `experiments/air_sea/environment.cpu-macos-py313.lock.txt`；其他硬件应建立独立环境并重新验证，不要求跨硬件逐位一致。

误差矩阵图可从已完成的验收 JSON 只读生成：

```bash
python examples/pde/report_air_sea_validation.py build/full_acceptance --out build/full_acceptance_report
```

图中分别显示相对空间差、时间/积分差与空间差之比；浮点噪声候选在原始 JSON 中标记，不拟合阶数。
