# AUTORL-R20：gamma时间信用单变量对照

日期：2026-10-16。训练执行 **INTERRUPTED / FAIL**，科学验收 **NOT TESTED**。已跨boot，原attempt永久封存，不重试、不续训、不退款。65,536步目标没有完整完成证据，不能把最后日志40,448冒充精确最终步数。正常发布与远端核验另以实际收据为准。

## 1. 阶段目标

相对R17只改变gamma=.99→.995，检验更长期的回报权重能否改善并保留原距离到达与停稳能力。不是重跑R17，不从任何旧ZIP续训。

## 2. 技术背景

已完整发布R19（545c6c9a321956e1b89314f1fa8a63583e475cf1）发现：三条实际诊断轨迹上成功回报仍高于失败，不能认定Reward鼓励这些失败；R17原域empty已经占83.47%的实际采样步。成功bonus在第283步、gamma=.99下只保留5.88%，提示时间信用是值得单独反驳的假设，不是已确认根因。

## 3. 系统架构

12个冻结short Train均匀→32768后自然reset切原距离、empty概率.5→原真实MuJoCo/LiDAR/编码/PI→中心化Actor PPO→原预算主管→冻结checkpoint→关闭Train后36个新Validation。Final Checkpoint，Test0。

## 4. 代码实现

新增temporal_credit_profile.py固定schema20与对照身份，temporal_credit_training.py在fresh模型首次采样前同步model.gamma和rollout_buffer.gamma，temporal_credit_cli.py只接受固定prepare/start、显式期望SHA。原budget/runtime/worker仅增加该严格profile的分支，复用既有主管和训练循环。旧schema17前缀检查保持；新gamma从首rollout生效，不错误要求32k权重等于R17。

R13 state_centered_policy.py、原CNN、Reward、原物理模型、PI/分配、动作、LiDAR及历史checkpoint不修改。每次优化前后核验真实gamma/buffer/lambda与原超参数；每份实际重载checkpoint再次核验gamma、原Adam与预测一致性。

## 5. 算法原理

SB3的model.gamma参与TimeLimit终端价值bootstrap，RolloutBuffer.gamma参与TD/GAE递推，两者必须一起变。lambda=.95保持，gamma*lambda=.94525。Reward分项记录的是环境真实Reward，不把PPO内部bootstrap或advantage混进去；实际逐transition advantage没有新增持久化，不可由episode return直接推断梯度。

Reward仍为NAV_V2_COLLISION_PREFLIGHT_V1：10Δd−.005||a||²−.5w(d)||v||²，成功+10、碰撞/物理失败−30、传感器失败−10、每执行step−.01、超时另−2；终止step的progress与优先级沿用原代码。成功仍距离<.10m、速度<.15m/s、连续5policy样本；不延长15s。

## 6. 冻结实验协议

ARL_R20_TEMPORAL_CREDIT_SEED0，seed0、65536实际step/3600s保守墙钟预留、CPU1worker。中心化Actor170857参数、LiDAR[3,12,24]+state7+heading2+age1、手写先验fresh独立初始化、std.1可训练；学习率3e-5/n_steps128/batch64/epoch2/clip.2/entropy0/value.5/gradclip.5、lambda.95全部保留。动作归一化[-1,1]^4经原接口转[1.5,1.5,1]m/s及1rad/s；500/100/25/10Hz、24000rays、7kg原模型不变。

REG572e0645ff4ef5d33b39ece9bf6aeb9860d274b0232e163747cc986a24b43092；129文件SOURCE7e674406492935f2ca0953d3e6e64ecd5860851b9f16746a6944a076210142c1；EXEC5f4c657947014e512725a7425d4174921974a39ee97d9ec425577e8214d63f23。0/32768/65536各6short+6original全新Val，历史episode复用0、Test0；训练完成关闭MuJoCo后顺序评价。

冻结五判据：32k short成功>=4、碰撞<=1且无物理/传感器失败；final short不劣于ownstep0；original final成功多于ownstep0且碰撞不增；相对R17 original final成功>0、碰撞<=3；原域empty final成功>=own32k且own32k>=1。五者合取，Final Checkpoint，不挑最好的32k。

预审披露文案Minor：继承evaluation.rehearsal_benefit字符串仍有旧R16字样，数值>0/<=3与R17当前控制值相同；本轮对照SHA、设计及正式分析判据temporal_credit_benefit_over_R17明确只核R17。保留Minor，不误称新Minor0、不修改旧实验。R7与R18历史Minor分别保留。

## 7. 实际结果

最新可核验完整优化日志为40,448步、316 rollout更新记录、1,264次Adam、632个PPO epoch；精确最终环境步数/停止信号/主管退出码 **NOT VERIFIED / UNKNOWN**。旧formal_progress停在39,936，只是更早快照。65,536步final ZIP、summary、complete和所有Validation文件均不存在，不能补写。新Validation实际0/计划36，科学五判据均 **NOT TESTED**，不是成功率0/6，也不是gamma科学FAIL。

| 已结束Train部分 | episode | success | collision | timeout | 平均Return | 已结束policy steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SHORT | 251 | 246 | 0 | 5 | 8.317552 | 32839 |
| ORIGINAL | 40 | 19 | 15 | 6 | 3.513537 | 7478 |

共291个结束episode，最后结束位于40,317步。没有为后续未结束状态补Reward、动作、姿态或终止原因。32839是实际自然reset切换点，不强行要求等于R17的32850。Train部分成功与手写先验有关，不能拿它替代独立Validation。

两份实际checkpoint（0/32768）的ZIP CRC/文件SHA/sidecar payload和静态JSON身份通过；32k ZIP SHA `dc0d55168cba43d4ea82c686966c87801b6fd4c79c1b307a0cb81322db68e3d4`，local-only。本轮没有在中断后加载R20训练checkpoint、核验其训练后Adam/预测或对它运行新episode，因此这些能力 **NOT TESTED**。另行postboot接口测试会构建fresh模型、加载测试自己的临时checkpoint和执行受控MuJoCo reset/step，不是R20模型评估或训练。容器内WARM_START能力标签不授予续训权限；本实验仍 **RESTART_REQUIRED**，禁止从32k ZIP恢复或用新ID掩盖重试。

训练前1144选择回归已完整PASS（193OPS/66Tensor/885历史）、15排除、0测试optimizer，不是全仓测试，也没有把被中断的训练当作测试PASS。新增6纯/3真实Tensor RED→GREEN，5分析方法契约+1finalizer来源契约。独立预审C0/I0/新文案Minor1；finalizer错误分析引用Important已启动前真实RED→GREEN关闭，失败收据保留。重启后另做193OPS/66Tensor的无优化器定向复核，最终结果见JSON实际收据，不把它称为重跑885项。

最后资源样本为wall538.279462s/CPU529.889144s、采样RSS1235.3828MiB，已记录训练HWM1657.7461MiB、累积采样minMem6423.9180MiB。完整wall time、whole-run峰值、最终物理tick和含退出的连续最低内存 **NOT AVAILABLE**，不从旧快照推算。原console的cv2/NumPy ABI导入stderr之后仍有316次完整更新，不能把这段启动警告误认为最终故障。

实际部分数据图：[episode](../mujoco/reports/autonomous_rl_temporal_credit_r20/figures/episodes.png)、[Validation缺测说明](../mujoco/reports/autonomous_rl_temporal_credit_r20/figures/validation.png)、[PPO记录](../mujoco/reports/autonomous_rl_temporal_credit_r20/figures/ppo_updates.png)、[目标误差与停稳](../mujoco/reports/autonomous_rl_temporal_credit_r20/figures/goals.png)、[资源样本](../mujoco/reports/autonomous_rl_temporal_credit_r20/figures/resources.png)。全部仅使用真实完整前缀；Validation图明确无数据，不伪造0成功柱或外推曲线。

## 8. 问题分析

**CONFIRMED**：permit/child原boot为b5fd7f5c-6278-4129-ae22-6a50bed7c8f9，恢复时当前boot为0b6aec35-b19b-4fd1-b0f0-c41ff4b7e148。限定原boot的kernel journal于2026-10-16 12:51:29.577043 UTC记录“Syncing filesystems and block devices.”，12:51:29.585268 UTC记录“Sending SIGTERM to remaining processes...”。最后PPO日志mtime为12:51:27.509480 UTC，时间紧邻。可以确认原boot发生系统关机；不能据此证明具体worker的最终信号、精确步数、关机发起者或原因。

原supervisor/worker PID及worker PGID已不存在、无活跃授权训练模块、flock空闲。16个原R20文件保持原样；新只读posthoc审计SHA `a3a83177901705d5f25525313f9a8dbe3d19bb91ab91a05175469af007586afd`，不是运行时异常捕获。初审未绑定boot、读取新boot oom_kill=0且笼统写journal无条目，不能用于排除旧boot OOM；原收据不改，新增boot补充SHA `cbdaf53bc6b8b31d761d01b19114fb7f9b224a2bf41e6bb6089acb70cd7ca066`透明更正并保存原boot实际关机消息。旧boot OOM是否参与原因仍 **UNVERIFIED**。

**SUSPECTED**：系统关机与训练中断强时间相关；没有证据支持模型NaN、资源阈值或本轮gamma为最终中断原因。**UNVERIFIED**：gamma实际收益、稳定避障与跨seed泛化。已有部分Train不能支持本轮科学验收。

独立boot-bound处置审查GO后，只通过现有BudgetStore.fail追加第26个FAIL事件、正常更新head；不改前25事件，不动原raw，不补造supervisor_failure或complete。处置收据SHA `f7441f2a5713031ec1a795415f299ef08b8f553c17017f1edd1f29a20a7dd359`。完整65,536step/3600s保守预留仍永久计费，余额1148032/39600、stop=false；这不是退款、重新授权或实验成功。旧298不可变授权文件、16额外raw、原模型及87基线untracked保持。

## 9. 局限性

单seed和小冻结开发Validation，手写初始化本就能完成部分任务；高Train成功率不等于学会LiDAR避障。无Test、无PX4/FASTLIO/实机。RESTART_REQUIRED，不支持exact/boundary resume，不从旧ZIP重启。cv2/NumPy可选ABI警告与Matplotlib3D警告保持披露，未改系统依赖。

## 10. 下一阶段

先完整发布本次INTERRUPTED报告与跨boot更正，独立实际复审、完整检查、单独commit和正常OPS sync远端；V2不得因资料已发布就把本次训练标记完成。下一安全任务仅离线研究关机/SIGTERM/不可捕获中断的证据与账本语义，测试使用独立模拟进程，优化器0，不注册后台守护服务。

在停机生命周期问题处理并经审查前不启动新训练；即使之后恢复研究，也不得重试本ID、从32k ZIP继续或以新ID冒充同一实验。下一算法实验须独立假设、协议和准入，不能把R20当作完整对照；余额仍1148032step/39600s，不扩大累计授权。

## 11. 给项目负责人的总结

这轮本来检验“更重视稍远未来”，但电脑在训练期间发生了关机。已经保留了真实的部分训练数据和32k中间模型；不能说算法已成功，也不能说这个改进无效。我们没有续训、重试或退回已用预算，也没有改奖励、模型或控制器。现在先把每日停机时的证据与恢复流程做清楚，再决定新的安全研究；电脑关机期间不能继续训练，中间ZIP也不等于能无损接着跑。当前没有额外权限申请，禁止PX4和真实设备。
