# 内孤立波沿斜坡传播

从 `BSPF_ISW_Slope_Run01_Rebuilt` 迁入的二维无滑移 Boussinesq 算例。
物理域长 1800 m，水深从 100 m 过渡到 45 m；黏性 0.01 m²/s、
浮力扩散率 0.003 m²/s，无旋转、无外力。所有壁面无滑移、不可穿透，
总浮力采用自然无通量边界。默认网格 321×161，非线性积分使用 2 倍 Gauss 点。

## 运行

从仓库根目录安装两个包，JAX 的 GPU 后端需按设备单独安装：

```bash
python -m pip install -e . -e './jax[slope]'
python examples/pde/isw_slope/run_slope.py \
  --case R2_321x161 --dt 0.5 --tfinal 50 --save 50 --out build/isw-slope-50s
```

未安装时可在命令前加 `PYTHONPATH=src:jax/src`。
单 CPU 计算线程运行：

```bash
PYTHONPATH=src:jax/src PJRT_NPROC=1 JAX_PLATFORMS=cpu \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python examples/pde/isw_slope/run_slope.py \
  --dt 0.5 --tfinal 50 --save 50 --out build/isw-slope-singlecore
```

`run_slope.py` 本身不强制 CPU；安装兼容的 CUDA JAX 后，可设置
`JAX_PLATFORMS=cuda` 使用 GPU。当前正确性验证来自 CPU，未宣称 GPU 性能。

省略时间选项时仍按原物理算例跑到 1800 s：前 1200 s 步长 2 s，之后 1 s。
可选 `--case R0_193x97`、`R1_257x129`；`--nowave --test` 是无波短测试。
`--resume PATH/state_*.npz` 支持续跑，网格、基函数和数学源码指纹必须兼容。
迁移前检查点的指纹不同，不支持隐式续跑。

## 代码划分

- `pybspf.ClosedBSPFLine`：端点节点化的 QR BSPF 标量空间及导数闭合的
  无滑移流函数空间。与压力求解器复用 QR 样条拟合，不导入旧三维求解器。
- `bspf_jax.mapped_boussinesq.MappedBoussinesq`：给定几何和背景分层的
  二维映射模型，复用库内张量算子、预条件 CG 和 RK4。
- `source/common.py`：本算例的地形、坐标映射、分层和冻结初场。
- `source/slope_solver.py`：将这些物理定义传给库模型的薄适配器。
- `source/execution.py`：时间调度、检查点、配置与诊断。

构建使用 NumPy/SciPy；推进采用 JAX float64。每步有限值检查在设备端完成，
仅回传 104 字节的 CG/状态报告，失败立即拒绝该步，保存最后有效状态。
旧的 NumPy 时间推进、三维速度/压力构造、打包验哈希和历史结果对照脚本不再
属于运行依赖。

## 初场和输出

`reference/shared_initial.npz` 保留原冻结初场，SHA256：
`92b87c89fc5f0efaf485b59083abc1c235774e69889bb78a522d3c0f5fed0603`。
它由历史导出数据重建，不是原始 DJL 连续解；本例不附带 DJL 求解器。

输出包括 `config.json`、`basis.npz`、每个保存时刻的 `state_*.npz`、
`diagnostics.json` 和完成后的 `summary.json`。导出物理场可用：

```bash
PYTHONPATH=src:jax/src python examples/pde/isw_slope/export_fields.py \
  --run build/isw-slope-50s --out build/isw-slope-50s.nc
```

NetCDF 导出复用已安装的 SciPy，无额外依赖；导出网格是后处理采样，不是更高的积分分辨率。

## 回归验证

```bash
PYTHONPATH=src:jax/src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  tests/test_galerkin.py tests/test_pressure_poisson2d.py \
  jax/tests/test_isw_slope.py jax/tests/test_flow_kernels.py -q
```

测试覆盖闭合基函数的壁面迹、导数关系、质量正交性，迁移前 NumPy 数值记录，
初值投影、RK4、散度、能量关系、失败拒绝、小状态包回传和检查点续跑。
测试参考数组在 `jax/tests/reference/isw_slope`，不携带第二份求解器。

迁移验证：上述 34 项测试通过；321×161、dt=0.5 s 完成 0–50 s 的 100 步。
在共同 97×65 后处理网格上，50 s 的 u/w 与迁移前结果最大绝对差约
3.2e-14 m/s，质量方程最大残差约 2.34e-12。NetCDF 导出维度、单位和
有限值检查通过。这是迁移回归，不是网格收敛或 GPU 性能验证。
本机证据保存在 `build/slope-migration-50s-final/migration_validation.json`；
迁移前完整目录存档于 `build/slope-migration-reference/original`，运行不依赖该存档。
