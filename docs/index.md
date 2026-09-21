# pybspf documentation

- [Getting started](../README.md)
- [Public API and array shapes](api.md)
- [Architecture](design.md)
- [Compatibility and behavior changes](compatibility_strategy.md)
- [Overall assessment and next steps](assessment.md)
- [Historical migration backlog](refactor_backlog.md)
- [JAX focusing NLSE example and validation](jax_nlse_example.md)
- [JAX nonperiodic KdV example and validation](jax_kdv_example.md)
- [Corrected JAX Euler–Bernoulli beam](jax_beam_correction.md)

- [JAX Sine–Gordon example and validation](jax_sine_gordon_example.md)

- [Boundary-driven Alfvén waves: JAX validation](jax_alfven_example.md)

- [Open parallel kinetic dynamics: JAX validation](jax_parallel_kinetic_example.md)
- [BSPF 1z2v 磁镜运动与粒子、能量收支](jax_drift_kinetic_mirror.md)
- [轴级 FFT / 低秩动理学实现](jax_fast_drift_kinetic.md)
- [保留 FLR 的静电 slab 回旋动理学](jax_gyrokinetic_slab.md)
- [五维回旋动理学解析 MMS 与收敛验证](jax_gyrokinetic_mms.md)

- [Self-consistent open-domain Landau damping study](jax_landau_open_example.md)
- [非周期开放 GK 波包、BSPF 收敛与温和端点结点加密](jax_open_slab_packet.md)
- [标准 BSPF 线性 ITG：增长率、FLR 和自由能验证](jax_linear_itg.md)
- [无驱动多模态非线性 ITG：守恒与带状模态转移](jax_nonlinear_itg_conservation.md)
- [ITG 驱动与带状流：增长转入非线性阶段](jax_driven_nonlinear_itg.md)
- [BGK 碰撞、平均热通量与统计饱和诊断](jax_bgk_itg_saturation.md)
- [固定 Nx=33 的 BGK 速度分辨率与长窗口扫描](jax_bgk_velocity_convergence.md)
