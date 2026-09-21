# 耦合多射流剪切不稳定探索算例

目的：在现有水平二维海气平台中，观察剪切不稳定、涡旋卷起、温湿度场拉伸，以及交换通量对海洋的反馈。不是三维边界层湍流或充分发展的湍流标度验证；本次不执行验收矩阵。

动力学探索所用配置：`experiments/air_sea/shear_instability_constant_24h.json`。
COARE配置另存为`experiments/air_sea/shear_instability_24h.json`。COARE探索因实际计算成本较高，已在1.5h安全暂停并保存现场；常系数版本保持相同初值和动力参数，保留双向交换，但不含稳定度反馈。两者是独立实验，不混接检查点。

- 方盒1000 km；海洋封闭无滑移，大气x周期/y不可穿透自由滑移。
- 目标风 `u=-12 cos(4πy/L)` m/s，两个空间周期；因此这是一种多射流扩展，不是原始双环流风型。
- 只在初值加入确定性的无散扰动，速度尺度1 m/s，含x方向两个波数；恢复目标不含扰动。
- 大气黏性5000 m²/s、标量扩散10000 m²/s，风恢复时间3天。按L/(4π)定义剪切尺度，Re≈191；不是仅凭这个数值判定湍流。
- 海洋动量层深200m，热力混合层仍50m；海洋黏性2000 m²/s。
- 当前快跑采用常系数双方温湿度交换；COARE配置可单独用于后续稳定度反馈研究。
- n81、积分32点；MRI宏步150s、实际快步30s、同步600s；24h，每半小时输出场和检查点。
- 保留原有预算、阶段有效性和谱尾阈值，不为了出图放宽保护条件。

新增的`wind_jet_count`默认1、`initial_air_perturbation`默认0，因此历史默认初值不变。新版本修改了源码指纹，旧实验若需精确续跑，仍需使用原冻结软件版本。

```bash
PYTHONPATH=jax/src python -m bspf_jax.air_sea_platform run \
  --config experiments/air_sea/shear_instability_constant_24h.json --out build/air_sea_shear_constant_24h
PYTHONPATH=jax/src python examples/pde/render_air_sea_shear.py \
  build/air_sea_shear_constant_24h --out build/air_sea_shear_movie
```

动画显示解析基函数导数计算的大气涡量、温湿度、热通量和海洋流函数。非纬向动能定义为大气速度减去每条纬线的x向平均速度后的能量，采用求解器的积分点与权重；可以与初始扰动能直接比较。首个24h动画有48个正时刻帧；续跑到72h后有144帧，均取自实际半小时输出，不做时间插值。

24h观察到射流弯曲，扰动能由202增加至754 J/m²，但不足以称为湍流。随后保持同一配置和初值历史，从完整检查点续推到72h，命令如下：

```bash
PYTHONPATH=jax/src python -m bspf_jax.air_sea_platform resume \
  build/air_sea_shear_constant_24h/checkpoints/step000000144.json --until 259200
```

多射流剪切不稳定的设计借鉴Kolmogorov流研究；本文通道边界、旋转、恢复及海气交换不同于双周期标准问题，因此不套用其精确临界Re：
[Lucas & Kerswell, Spatiotemporal dynamics in 2D Kolmogorov flow over large domains](https://arxiv.org/abs/1308.3356)。

## 48小时动画

按用户选择，采用两周期射流的弯曲演化，停止继续寻找更强湍流。已停止四周期探索。两周期原始运行安全暂停于53小时20分；动画严格截取前48小时，共96个半小时正时刻帧，8fps，时长12秒。

```bash
PYTHONPATH=jax/src python examples/pde/render_air_sea_shear.py \
  build/air_sea_shear_constant_24h --until-hours 48 --out build/air_sea_shear_movie_48h
```

此动画使用常系数双向交换，展示射流弯曲和温湿度输运，不标称充分发展湍流。
