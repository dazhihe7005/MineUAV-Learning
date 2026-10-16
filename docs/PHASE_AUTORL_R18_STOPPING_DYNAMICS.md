# AUTORL-R18：真实停稳动态诊断——本例final是未到达，不是到达后高速

## 1. 阶段目标

补足R16/R17旧Validation汇总缺少的逐策略步速度/动作证据，区分未到达与到达后未停稳。不更新任何网络，不重新训练，不覆盖旧Val原始数据。R17完整发布07f2113b44e25f473dada365d1487b5b4df0f984，永久科学FAIL不改。本阶段工程采样与复现核验PASS；算法性能提升**NOT TESTED**。最终阶段完成仍以独立结果复审/Commit/正常OPS sync远端核验为准。

## 2. 技术背景

R17增加empty重放后，原Validation final仍0成功/3碰撞/3超时；两个empty在32k成功、final超时。旧bounded_traces仅最初四个Train episode每5step记录，不能借末距离编造final运动过程。本轮预先固定同一开发Validation empty及三个不同冻结策略，得到新的受控诊断轨迹，不当作新的独立泛化评价。

## 3. 系统架构

既有CheckpointManager只读加载→verify_centered_policy→原build_ppo_scene/HeadingObservation/LidarNavigationEnv→当前真实state与LiDAR→deterministic推理→原PI/四电机/MuJoCo推进→25Hz逐步JSONL→独立因果分析。每次只有一个plant，关闭后才下一个模型。没有PX4、SLAM、真实设备或Test。预测使用合法当前观测，不向网络注入障碍真值。

## 4. 代码实现

新增`research_regression/authorized_rl/stopping_dynamics.py`（固定引用/协议/准入）、`stopping_trace.py`（真实轨迹采样）、`stopping_dynamics_run.py`（独占生命周期）及对应`stopping_dynamics_tests.py`和`tests/research_ops_v2/test_stopping_dynamics.py`。未改任何历史环境、Reward、控制、Actor/CNN及训练入口。

每步先复制pre time/qpos/qvel/state7/heading/age，再推理；env.step后记录实际info的速度/yaw命令、post position/velocity、PI积分、期望加速度、接触/Reward和终止。pre与post不混时钟。原环境freejoint位置/线速度遵循world ENU协议，四元数wxyz；不是新的机体COM标定、SLAM或PX4估计。回调每条完整transition立即flush；异常保留已完成prefix与failure信息，未知字段不补零。

独立进程组+原lifetime guard+flock；每策略步检查资源和停止。learn/train/Adam.step硬拒绝；模型/Adam/num_timesteps前后核恒等。全部299授权文件和七份额外历史raw只读保护，无新预算事件。已有OUT永久禁止重试。

## 5. 分析原理与阈值

沿用原成功条件：post距离<.10m、真实速度<.15m/s、连续5策略样本。独立重算near、near-fast、joint和最长连续joint。命令与pre速度点积<0只表示声明速度方向相反，不等价于真实减速度或稳定性。逐轴pre目标误差sign变化跳过abs<=1e-6m死区并保留上个非零sign；是25Hz采样现象，不据此宣称系统不稳定。图中离散声明command用阶梯图、真实运动用实际post时间。

## 6. 固定协议与身份

唯一AUTORL_R18_STOPPING_DYNAMICS_ONLY，三个新诊断episode：R17@32768（与R16@32768同权重）、R16final、R17final。固定Val身份fe85ac443a0dc1bd2c676b6455f28ef0a4d15ea046287b692cc7eb3f6eefe06f，每个最多375评价step/15s，合计最多1125；训练/optimizer/Test0，CPU1worker、wall180s。没有加载旧模型继续训练，模型选择不变。

- REG634fa6175039f3a4aca3e4252d0da7bdd548c3f05805b8bef31338e670283164。
- SOURCE65386a82f54146cbac41afa918feab4780e13990ca5198bafd5079302274c480（121文件）。
- EXEC1091396345397801e99dd5bfec549763753e5099692766e8e7a69296fa957778。
- checks52c50e58411748660c7c80604511d3ad39c379fda02d36f0c2db6b1a3a38f2ad。
- result bd56b076e5f2241ce94a6843c6760709eaa70dc81ba64ee74f8edc4b8703384b。
- 冻结分析4c5c32487e643c5c8eb6a66a0e8801202314cb7b36c9580168b6c0522c6a93ed。

三个策略的终止、步数、末距离及累计Reward均以1e-8容差复现旧同场景摘要，证明本例插入诊断没有改写行为；不是把本次trace回填旧实验。共同32k只加载一次。

## 7. 真实结果、测试与资源

| 冻结策略 | 新评价step | outcome | 最小采样误差m | 首次进入<.1m | near/fast/joint | 最长joint |
|---|---:|---|---:|---|---|---:|
| R17 32k = R16 32k | 283 | success | .035374 | 10.60s | 19/14/5 | 5 |
| R16 final | 375 | time_limit | .101451 | 未发生 | 0/0/0 | 0 |
| R17 final | 375 | time_limit | .123842 | 未发生 | 0/0/0 | 0 |

32k连续5 joint结束于11.32s，成功末距离.050981m。两个final最小误差都在15s终点，末真实速度分别.131937和.065627m/s，均小于.15，但位置尚未进目标区；本例不能归因为“已到达却速度过快”。32k峰速.779626m/s，R16final .560065、R17final .399145；较低峰速不等于更好的导航。pre误差XYZ sign变化分别[3,5,0]/[0,6,0]/[0,2,1]，不据此证明稳定性或忽略未采样的运动。

实际1033个新评价step、20660物理tick=41.32s仿真时间，3个新Val诊断/Test0。硬拒绝optimizer、实际模型/Adam/训练步不变，旧grant全部299文件字节保持，预算仍1213568step/43200s、stop=false；无预留/退款/账本重置。6份OUT raw与外部console/exit都有SHA，NPZ/模型未上传或改写。三图`goal_stop.png`、`commands.png`、`trajectory_pi.png`位于`mujoco/reports/autonomous_rl_stopping_dynamics_r18/figures/`，来自实际射线/动力学链，主执行者已目视核验。

1126项当前选择PASS=178OPS+63Tensor+885历史；15明确排除、0测试optimizer，不称全仓。另4独立分析契约GREEN，包含时钟因果、tick差、非法值和逐轴sign死区。最初缺模块/未接准入函数/缺on_record签名的失败收据均保留。独立预审C0/I0/新Minor0；预注册sign统计缺项Minor启动前RED→GREEN补齐并复审关闭。继承R7失败上下文Minor未改，本次不走该scheduler路径。

内层16.749s/CPU16.103s，launcher24.011s；whole reaped-child及worker HWM1444.871MiB，逐策略guard采样minMem7025.723MiB。该采样不覆盖imports/model loading内部的连续最低，完整连续最低**NOT AVAILABLE**。三个策略依次加载，无多worker、无OOM观察或超时，主管exit0且owned group gone。可选cv2/NumPyABI及Matplotlib3D警告不隐瞒，CPU/2D实际可用，无全局依赖修改。

## 8. 问题分析

**CONFIRMED**：本场景两个final在15s内未进入目标半径，不能称到达后停稳失败；成功32k兼有更快接近和末段五样本停稳。新轨迹与历史摘要严格一致，非插入诊断导致性能改变。**SUSPECTED**：后续学习造成声明运动更保守、位置反馈改变或奖励/时间信用折中，可能使15s任务慢速渐近。**UNVERIFIED**：唯一根因、若延长任务时限会否成功、对其余五Val及障碍任务的解释、修改哪项算法必然有效。不会由单个case或曲线外观直接改Reward/PID。

## 9. 局限性

仅一个开发Validation场景/三个冻结权重，无新训练成绩或泛化认证；同场景新采样是诊断复现，不增加独立评估样本数。25Hz边界轨迹不提供500Hz连续状态证明；误差sign变化不等同稳定性结论。动力学理想传感器假设不变，freejoint协议不作真实COM/外参标定。没有实机、PX4、FAST-LIO或Test，没有精确/边界训练恢复能力。

## 10. 下一安全任务

先实际结果独立复审、历史audit、Commit/正常OPS sync远端。随后基于“final未进入目标而非末段高速”的新事实冻结下一项单变量研究：优先验证奖励时间信用/声明位置反馈或任务时限的诊断假设；不修改本轮结果、不延长已结束run、不把更宽时限结果冒充原15s PASS。任何训练需新profile、准入/GO及实际剩余原grant，不盲重跑百万步或旧ZIP续训。

## 11. 给项目负责人的总结

这次没有训练，而是把三个固定策略真实运行中的每一步动作和速度记录下来。发现我们先前怀疑的“到了以后停不住”并不适用于这个失败空场景：最终模型其实在15秒内还没走进合格半径，虽然最后速度已经很低。中间模型能在11.32秒成功。这个结果没有提升算法，但纠正了下一步排查方向，避免盲目加强制动或修改PID。接下来会围绕到达速度和时间信用做有对照的独立研究，仍在原授权内，不动Test、不接PX4。

### 发布流程补充（不改实验结果或运行源码）

首次结果提交1ee2d6f7568b0e6043386763410bee32c27884e7已通过正常OPS sync并远端核验。此前seal已追加真实结果复审并显式stage，但`git diff --cached --check`因`stopping_dynamics.py:85`多余文件末空行返回2，顺序工具编排没有在该失败后中止，仍生成了提交。这项空白检查是**FAIL（非功能格式警告）**，不能写成PASS。独立统计、1126测试和实际运行结论不因此改写。

补核HEAD实际19项提交恰等已审注册白名单，所有blob/秘密数据限制、121文件SOURCE、真实复审引用、history与readiness均通过；没有误提交历史文件或额外源码。现场收据`.research_local/autorl_r18_publication_warning_01.json`保留。不为消除此警告而改写已验证运行源码，不重跑实验、不amend或force push；本补充仅修改已登记文档和报告，另经独立只读处置审查及正常提交同步。后续提交必须先取得检查的完整成功结果，再单独执行commit，不将失败或未知状态继续串接。补充交付结论以独立复审/远端收据为准。
