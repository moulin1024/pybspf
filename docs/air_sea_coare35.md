# COARE 3.5 海气交换方案

本平台已有独立 JAX COARE 3.5 交换核和 MRI 阶段接入。本次为当前两周期射流建立明确启用稳定度反馈的48小时配置，并增加温湿度反馈到应力的回归测试。没有重新运行验收矩阵，也不将历史常系数结果改标为 COARE。

## 当前算例的启用方式

```bash
PYTHONPATH=jax/src python -m bspf_jax.air_sea_platform run \
  --config experiments/air_sea/shear_instability_coare35_48h.json \
  --out build/air_sea_shear_coare35_48h
```

新配置与之前常系数两周期射流采用同样的初始状态、空间参数和动力参数；指定48h总时长、每半小时输出。交换选择为 `surface.method="coare35"`，`surface.stability=true`，风温湿参考高度均为10m。常系数配置中的三个固定交换系数不会用于 COARE 计算。

必须从相同初值新建实验，不能把常系数48h终态当成同初值的COARE对照，也不能修改旧检查点物理参数后声称精确续跑。

短程检查配置 `shear_instability_coare35_smoke.json` 使用 n33、积分20点、600s总时长；其他物理参数与48h配置相同。它用于阶段接入/有效性/预算/输出冒烟检查，不用于证明48h空间解析度。

## 已实现的交换路径

每个 MRI 快阶段从双方当前状态重建海气相对速度、海温、大气温度与比湿。COARE 核计算稳定度、动量与标量粗糙度、摩擦速度及弱风阵风，返回矢量应力、显热和水质量通量。

- 应力向海洋为正，投影后以相反号作用于大气。
- 显热和水通量向大气为正；海洋热损失为 H+LvE，大气显热获得 H，水汽能获得 LvE。
- 潜热常数只由模型统一应用一次。
- 双方交换倾向与预算账本在同一阶段采用同一通量和权重。
- 零相对风不通过除以该风速求应力方向；阵风仍可参与标量交换。
- 非有限值、负比湿或闭合无效会留下现场并终止，不切回常系数或中性公式。

NetCDF 同时保存 `exchange_drag/heat/moisture/stability/friction_velocity/gustiness/valid/thin_stable_branch/iteration_change`，便于确认运行中实际使用的交换方案与空间变化。

## 适用范围与验证

以 NOAA 官方固定提交 `bd1cac80d2dae454e699f9ef46204dfd88086a71` 为参考；源码、MIT许可证与SHA256清单位于 `jax/tests/reference/coare35/`。保留官方十次迭代和分段弱风处理；关闭冷皮肤、暖层、降雨及外部波浪输入。混合层温度和大气层平均状态仍是有效表面代理，不是解析的垂直廓线。

专项测试覆盖官方通量对照、5/10/20m参考高度、零/弱风、稳定与不稳定状态、单位与符号、热水界面守恒，以及光滑分支的MRI四阶时间精度。新增测试分别改变海温、气温和比湿，确认 COARE 会改变动量倾向，而常系数方案在相同速度下不产生这条动力学反馈。

```bash
PYTHONPATH=jax/src pytest -q jax/tests/test_surface_exchange.py \
  jax/tests/test_air_sea_audit.py::test_coare_coupled_interface_is_conservative \
  jax/tests/test_air_sea_audit.py::test_thermodynamic_feedback_changes_stress_only_with_coare \
  jax/tests/test_air_sea_validation.py::test_smooth_coare_multirate_fourth_order
```

本次只执行专项短测试和600s冒烟运行。48h配置已准备，不自动启动长运行；现有48h视频仍属于常系数方案。
