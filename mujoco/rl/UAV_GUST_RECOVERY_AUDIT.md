# UAV Gust Recovery Audit — 只读审计

日期：2026-10-10。Branch：`feat/uav-gust-recovery-audit`。Base：`e273c24eda37a3f06c21d845a4bc5e94be065b98`。

## 结论

最强证据是三个控制器共有的**撤风后往复运动与停稳失败**，而非残余外力、动作/allocator饱和或仅仅未通过“五步保持”。600条Gust轨迹均完整经历2秒外力，4秒后每个已记录physics tick的外力严格为零；直到15秒终止，没有一次完整25Hz观测同时满足距离<0.10m与速度<0.15m/s。599/600曾单独进入距离阈值，592/600曾单独低于速度阈值，但两者发生在不同时间。全部是timeout，非physical failure。

可以确认有限时间内没有有效停稳，不能据此声称永久无法恢复、动力学发散或PI windup已被证明。PI内部过程未记录，单一模块根因仍未确定。

## 数据与身份

复用上一实验所有原始文件；未生成任何新MuJoCo episode、未运行policy inference、未训练。完整核对1800条源记录唯一task ID、record hash、原始NPZ SHA256；详细分析Constant-Medium、Gust-Medium、Gust-High各3controller×100目标，共900条。没有挑选成功片段或删除失败。源manifest、原报告、acquisition cache、控制/物理源码及checkpoint均在分析前后核对hash。

全量指标、逐episode派生统计和1800个原始身份hash见[JSON报告](../reports/uav_gust_recovery_audit.json)。源数据位于本地`mujoco/reports/uav_bc_external_disturbance_seed0_parts/`，不重复提交。固定代表episode indices **0、33、66、99**在重算指标前指定；这些目标上的Constant成功/Gust失败和三个controller严格按target ID配对，不按图好看与否选样。

Original BC checkpoint SHA256：`7d7cf248c1fc672ec859bb906f4e8e0f442fbee9c6daea8fec54d4eee64c772c`。

Yaw-Augmented checkpoint SHA256：`d29b93555ef46f231228c23345761f76b5758f6451648375fcc2b7e4501f03ce`。

两文件字节在本审计前后完全一致。上阶段parameter hash attestation为Original `771382758078def13665abef25ec5a9b3702b7938f265f55294605bcc5b60cfa`，Yaw-Augmented `b1cdac0dd564baaee225b2c13e8e19a0540a842b5764da47212ebba8efffa5f5`；本轮未加载/执行模型，因此不把继承的attestation写成新运行时检查。

## 时间轴与判定

| 项目 | 源码与轨迹核对结果 |
|---|---|
| Physics | 500Hz，dt=0.002s |
| PI/control | 100Hz，dt=0.01s，5个physics ticks/control update |
| Policy | 25Hz，dt=0.04s，4个control updates/policy step |
| Episode上限 | 15s，375 policy steps |
| Gust | `[2,4)`秒，整数physics ticks1000–1999，完整1000个受力ticks |
| 撤风后窗口 | 所有600条均为11s；没有提前success或physical termination造成暴露不足 |
| Task success | distance<0.10m且speed<0.15m/s，连续5个policy结果；没有yaw阈值 |
| Timeout | 到达原始step上限且没有success/physical termination |
| Gust recovery | 相同几何/速度阈值，但只使用完成整个gust之后、t≥4s的5个连续完整有效policy观测 |

源码位置：[环境step](mine_uav_env.py#L242)、[gust hook](uav_bc_external_disturbance.py#L60)、[原始recovery verifier](uav_bc_external_disturbance_eval.py)。外力在每个`mj_step`前清零buffer并写入正确body的world-XY COM合力；没有直接yaw torque。本轮只读源码与保存的force buffer，不再次积分。

5个25Hz边界观测首尾跨度为0.16s，而5个完整控制区间覆盖0.20s。这不是判定被修改：原环境success和原独立recovery的索引规则分别保留。当前600条的joint qualifying count本身就是零，故任何5步dwell细节都不是本次共同失败的充分解释。

11s是真实非空窗口；数据仍在15s右删失，无法判断延长仿真后是否最终恢复，也不能证明11s一定足以恢复。没有发现gust结束时间错误或“尚在受力却被算作撤风后”的证据。

## 实际PI实现

只读检查[velocity controller](../control/velocity_command_controller.py)、[PI配置](audit_velocity_pi.py#L32)、[thrust/attitude转换](../control/position_controller.py)、[attitude PD](../control/hover_controller.py)。

- World-frame速度误差`e_v=v_cmd-v`；P项`diag(1.5,1.5,2.0)*e_v`。
- 每100Hz更新候选积分`I_candidate=I_old+e_v*0.01`。Ki为`[0.5,0.5,0.8]`；积分误差单位m，Ki贡献单位m/s²。
- clamp的是**Ki贡献**：XY向量norm≤1.5m/s²，Z绝对值≤1.5m/s²。等价XY积分norm≤3m，Z绝对积分≤1.875m；不是每个XY分量各自1.5。
- Conditional anti-windup：若`P+candidateI`超过3m/s²且新增Ki积分沿饱和方向继续增大，则冻结该XY或Z更新。没有持续外力估计器、显式leak/back-calculation或gust撤除时积分重置。
- 最终desired acceleration为P+I，再限制XY norm≤3、Z绝对值≤3m/s²。经`m(a-g)`世界推力向量→desired quaternion/thrust→attitude PD wrench→allocator→rotor dynamics。速度不能瞬间等于policy command。
- 每次episode reset清零积分、desired acceleration、anti-windup计数，恢复yaw target。episode内部保留积分和积分式yaw target；**保存的初始snapshot证实积分零，但后续积分时序not recorded**。
- Scripted command为`clip(target_error/3,velocity_limits)`，再映射为4D normalized action；其xyz指令没有显式利用观测速度，yaw为wrapped yaw error的限幅rate。三个controller共享同一低层cascade。

三类饱和不能混同：policy action到接口边界、allocator对wrench/rotor的饱和、PI积分/desired acceleration限幅。900条记录中前两者都是0；第三类没有可信运行时信号，**不能从前两者为0推断PI未饱和**。

## 全量结果

下表是每组100条的episode等权均值。Constant成功即停止，因此final和completion与Gust固定15s终止不处于同一物理时刻；不能把Constant成功后外推至15s。

| Condition | Controller | Success | Final distance m | Final speed m/s | Peak speed m/s |
|---|---|---:|---:|---:|---:|
| Constant-Medium | Scripted | 100/100 | 0.055920 | 0.109575 | 1.103025 |
| Constant-Medium | Original BC | 97/100 | 0.061910 | 0.113132 | 1.096313 |
| Constant-Medium | Yaw-Augmented | 94/100 | 0.067928 | 0.110935 | 1.088197 |
| Gust-Medium | Scripted | 0/100 | 0.235642 | 0.160165 | 1.506488 |
| Gust-Medium | Original BC | 0/100 | 0.226688 | 0.171175 | 1.496542 |
| Gust-Medium | Yaw-Augmented | 0/100 | 0.233598 | 0.145088 | 1.478409 |
| Gust-High | Scripted | 0/100 | 0.114990 | 0.344430 | 2.100862 |
| Gust-High | Original BC | 0/100 | 0.097200 | 0.357982 | 2.089564 |
| Gust-High | Yaw-Augmented | 0/100 | 0.113910 | 0.313078 | 2.065649 |

Constant-Medium Scripted平均成功时间12.0664s。与相同100targets的Gust-Medium配对：100个Constant-only successes、0个Gust-only successes；Gust final distance增加平均0.179722m，peak speed增加平均0.403463m/s。BC对应差异同方向。Gust各组全100timeout，无physical/numerical failures。

### 分阶段观察

统计使用每个动作开始前的真实观测、按实际控制区间时长加权，然后episode等权聚合；对每阶段报告eligible episodes。这里9组在4个阶段起点均有100条，但Constant晚期逐时点样本逐渐减少。图中的成功终止后区域为NaN，不继续携带terminal值。

Scripted的phase mean distance / mean speed如下（各cell单位m / m/s）：

| Condition | 0–2s | 2–4s | 4–6s | 6s–结束 |
|---|---|---|---|---|
| Constant-Medium | 1.45289 / 0.68742 | 1.07870 / 0.67328 | 0.44676 / 0.32725 | 0.31004 / 0.23654 |
| Gust-Medium | 1.35933 / 0.33439 | 0.69779 / 0.79032 | 0.81725 / 1.00709 | 0.35613 / 0.44730 |
| Gust-High | 1.35933 / 0.33439 | 1.17272 / 1.38121 | 2.08337 / 1.32971 | 0.40027 / 0.59557 |

BC也表现相近：Gust-Medium Original/Yaw-Augmented的4–6s速度为1.00016/0.98863m/s，6s–结束为0.44491/0.43555m/s；Gust-High对应1.32316/1.30993及0.58541/0.56963m/s。最后1秒平均速度仍为Medium S/A/C：0.29075/0.29046/0.27637；High：0.45575/0.43769/0.42363m/s。Yaw error很小，JSON保存各phase及final yaw误差，不能把yaw_error日志当作完整attitude日志。

### 阵风撤除后的误差是否继续扩大？

**Confirmed：会短暂继续扩大，但不是持续发散。** 每组100/100在4秒后达到比4秒至少高1e-6m的distance peak。Scripted Medium/High平均超出4秒distance为0.10197/0.29625m；High post-distance peak时间median4.44s。Medium最大的post-distance peak可能来自稍后反向往复，时间median4.32s、max6.52s，不能都称为一次同方向overshoot。

Scripted Medium/High distance在4s为1.18295/2.49616m、6s为0.72706/0.49134m、15s为0.23564/0.11499m；15s相比4s下降平均0.94731/2.38117m。速度撤风后再次增大：post-speed peak平均1.50649/2.08428m/s，peak time median5.40/5.60s。最终误差下降与继续往复、速度未与位置一起达标同时存在。

### 控制指令与速度方向

对XY command norm≥0.02m/s且actual XY speed≥0.15m/s的观测定义`dot(v_cmd_xy,v_actual_xy)<0`。Scripted Medium的6s–结束反向比例episode均值47.78%，每episode该阶段最长连续反向段平均1.1012s；High46.32%/1.2388s。BC相近，详见JSON。不是整段11秒持续反向，更不是证明PI方向写反：反向指令本来可以是正常制动；source error、P/I及attitude未记录，不能知道转向/制动延迟具体来自哪个环节。

## 逐问题证据等级

| 问题 | 等级 | 回答 |
|---|---|---|
| Constant-Medium100成功、Gust-Medium0成功为何不同？ | Confirmed（现象）；Supported hypothesis（机制） | 前者从0s受力，后者2s才开启、4s撤除；其运动/内部状态历史不同。Gust撤风后存在共同大幅瞬态与往复而未停稳，Constant完成任务。不能将差异唯一归因于撤风/I项。 |
| 4s后误差继续扩大？ | Confirmed | 全600条短暂增大，随后明显缩小，不是单调发散。 |
| 长期速度/指令方向不一致？ | Confirmed（观测） | 存在多次约秒级反向段，非持续同一方向错误；反向可为制动。 |
| PI积分限幅、滞留、反向补偿？ | Unresolved | 代码允许限幅/持有积分，但没有运行时PI数据证明事件发生或量化其作用。 |
| 三controller机制相似？ | Confirmed（过程）；Supported hypothesis（共因） | 撤风后速度/距离曲线和timeout相近，支持共同cascade/vehicle transient限制，不能证明唯一共享模块原因。 |
| 不能恢复，还是未及时停稳？ | Confirmed（窗口内）；Unresolved（永久性） | 很多重新进入目标区域，但在15s前从未同时达位置和速度阈值；超过15s会否成功未知。 |
| 能确定根因？ | Unresolved | 可以定位可见失败表型，不足以识别PI windup、braking环节、attitude lag等单一根因。 |

刹车不足的直接证据仅限于**有效停稳结果不足**：near-target仍有速度，joint sample为零；不等于已证明PI制动输出不足。低阻尼/共有transient tracking问题为Supported hypothesis。Integrator windup、clamping实际触发、反向补偿和“evaluation timing bug”均未被确认为根因。严格success确实决定timeout，但本组不是仅多等4个采样就能成功；保留原标准且不制造宽松成功率。

## 缺失信号与唯一后续建议

未记录：100Hz actual velocity / velocity error、PI integral state/Ki contribution、P项、raw/limited desired acceleration、clamp与anti-windup events、实际quaternion/angular velocity、desired attitude、rotor/thrust/wrench时序。记录的是25Hz实际position/world velocity、target error、wrapped yaw error和policy commands，500Hz force buffer、allocator saturation counts与terminal flags。初始PI zero和yaw target来自snapshot，不是全episode内部日志。

不生成`pi_integral_diagnostics.png`，因为没有可信PI时序。

唯一建议是**一个固定target的Scripted Constant-Medium/Gust-Medium最小telemetry验证实验**：保持gains/criteria/force设置不变，记录上述100Hz P/I/限幅/anti-windup及attitude/allocator输出，验证撤风后哪个内部状态先于速度异常持续。这里只提出方案，未运行。即使拿到日志，因果归因仍可能需要后续单因素干预，不能直接凭相关性定因。

## 复现、验证与资源

从repo根目录顺序执行（仅读取已有轨迹）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python mujoco/rl/uav_bc_safety.py --directory mujoco/reports/uav_gust_recovery_audit_parts --phase analysis -- .venv/bin/python -u mujoco/rl/uav_gust_recovery_audit.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python mujoco/rl/uav_bc_safety.py --directory mujoco/reports/uav_gust_recovery_audit_parts --phase tests -- .venv/bin/python -m unittest discover -s mujoco/rl -p test_uav_gust_recovery_audit.py -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python mujoco/rl/uav_bc_safety.py --directory mujoco/reports/uav_gust_recovery_audit_parts --phase publication -- .venv/bin/python -u mujoco/rl/uav_gust_recovery_plots.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python mujoco/rl/uav_bc_safety.py --directory mujoco/reports/uav_gust_recovery_audit_parts --phase finalization -- .venv/bin/python -u mujoco/rl/uav_gust_recovery_plots.py --finalize-resources
```

18个新增纯数据fixture测试覆盖时间/动作对齐、共同判定、partial boundary、方向解释、缺失PI、hash/read-only、配对ID、图表不外推和代表索引。Final review指出success独立verifier需要与源环境一致的失败终止门控：已在一轮RED→GREEN中补全完整/有效/无failure步判定，并验证全部saved streak，而不仅最后一项；当前900条无physical failure，原有指标不变。资源发布另外纳入完成后的supervisor峰值，避免只报告JSON写入前的早期HWM。全项目测试本轮不运行，因为其中部分会创建MuJoCo episode或训练fixture，违反本轮边界。独立重算terminal distance/speed/peaks、success streak和gust recovery与原指标一致；全部1800源hash前后一致。连续第二次分析的derived metrics必须逐项相同，证明重算可复现。

单进程顺序分析，**0 MuJoCo workers/0新episode/0optimizer steps**，atomic JSON/PNG输出，保留原始data与历史untracked。guard保留约1.5GiB系统余量，RSS上限4GiB/available低于2GiB或OOM计数增加即停止；不改swap。最终受保护执行的进程树峰值RSS约155.19MiB，全部phase OOM计数增量0；18/18纯测试通过。实际资源测量和测试结果见JSON。图表为5个所需有效文件，所有internal-missing说明保留，非凭空PI图。

图表：[force](../reports/uav_gust_recovery_audit_figures/force_vs_time.png)、[velocity](../reports/uav_gust_recovery_audit_figures/velocity_vs_time.png)、[position error](../reports/uav_gust_recovery_audit_figures/position_error_vs_time.png)、[command/actual](../reports/uav_gust_recovery_audit_figures/command_vs_actual_velocity.png)、[termination](../reports/uav_gust_recovery_audit_figures/termination_timeline.png)。
