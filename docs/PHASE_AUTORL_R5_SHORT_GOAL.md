# AUTORL-R5：短距离到点停稳的真实物理可完成性

## 【1】阶段目标

验证更短任务能否在原动力学/PI/真实LiDAR/原Reward/原成功条件下完成，为独立课程PPO对照提供准入，不把手写控制器成果当作学习成果。

## 【2】技术背景

R4已完成并远端核验ac10e5cccb2e84f9b42b39d4924f4c71bfdd19bd，不重放。有限反事实探针显示训练后目标反馈偏置，不证明唯一原因。近距离课程旨在缩短获得成功反馈的难度，原R1仅切换empty/混合且未改变距离，不能作为相同研究重复。

## 【3】系统架构

原L6冻结元数据→距离任务派生与独立hash→原MultiScenarioTrainingEnv/HeadingObservation/LidarNavigationEnv→state-only Scripted→原PI→实际MuJoCo mj_step与24000 raycast→原终止分类→有界真实轨迹证据。500/100/25/10Hz及原7kg模型不变，不使用KINEMATIC_POSE_REPLAY。

## 【4】实现

新增short_goal.py负责固定派生、阶段和证据准入；short_goal_env.py仅在reset选择目标，保留原场景/起点/Yaw和抽样；short_goal_feasibility.py唯一受限执行；short_goal_tests.py及OPS测试。base identity指原几何任务，task identity指新短目标，二者分别记录，不冒用原场景身份。

完整证据校验要求固定R5注册/源码/协议、PASS、raw success、12双身份、起点/目标、逐帧时间/20物理tick/位置距离/速度范数和总tick。首次预审_01为NO_GO（I1证据不足），先RED7拒绝失败→GREEN8项，再冻结_02完整复审。原候选/原始失败保留，未先运行后修复。PPO.learn/train/Adam.step tripwire禁止更新；原导航环境、Reward和训练器均未改。

## 【5】算法与课程变量

仅派生target=start+0.20*(target_original-start)/原距离。12个Train短任务、6个Validation短任务（本阶段不运行Validation），Test不派生/不运行。场景障碍、起点/Yaw/seed不变。静态可达性检查只服务任务生成，不给网络地图或最优路径。

原0.35m设计在Train41004/41009和Validation51001的0.20m几何裕量检查中被拒绝，原始geometry_red_01保存。实际实验前改统一0.20m并重新核验18行目标；没有放宽几何裕量、原成功半径0.10m、速度0.15m/s及连续5step标准。起始0.20m大于成功半径，不会reset即成功。

## 【6】执行协议

ID AUTORL_R5_SHORT_FEASIBILITY_ONLY，唯一.research_local/autorl_r5_short_goal/run_01。12Train各最多375策略步，总最多4500，单CPU worker/单场景常驻；guard180s，内部170s。MemAvailable≥1536MiB、磁盘≥2GiB、预算stop请求/源码变化立即停止；已有目录不覆盖、不重试。预算grant/head/events前后逐字节SHA核验，训练/optimizer/Validation/Test均0。

登记608a2c4c24a23343338d48832ab78ddf574527aa919c5e4d5072e2beb86c5a63；源码2d576a2e81db76b0c6f3252fa97aa0da1b2ab074bebc42e32f45b33a00653289；协议f52ea7b9e84ee44335c256521a086432894390aedba4b6745da93857a83fc1b7；任务清单f93b6dce91abeadbde8ebaf619a3c5839cbb9312cca2402f2868472345d5faf3。_02终证据、独立GO、登记/doctor全部通过后才唯一运行。

## 【7】实际结果

PASS：唯一12个Train短任务全部达到原成功标准，0碰撞/超时/物理/传感器失败；每个58–59步、2.32–2.36s，合计701策略step/14020物理tick/28.04s累计仿真时间。初始误差统一.20m，最终误差.078590–.083638m、速度.097661–.109534m/s，全部末5step连续满足距离<.10m且速度<.15m/s。生成288个真实LiDAR帧（含每episode reset初帧），没有未来观测；这是原宽容成功定义的停稳，不等价于零速度或精确停在目标中心。

result SHA395fe3891efc2b7917d087a939922d945bd9d3c1aace4e632efa95b366fd0a94，运行源码与_02冻结一致。程序退出后用完整准入验证器复核PASS；后续R6必须pin这个已发布proof SHA，不接受任意自洽的新JSON。物理原始episode逐个原子写出，local-only，未覆盖旧实验。

1003项唯一相关测试PASS=99 OPS+19无更新Tensor/环境+885历史，15逐项排除、历史回归125.16s/2既有warning，非全仓。_01旧源码测试不当作新版本覆盖，也不重复计数。_02独立预审GO、实际终证据及准入af22e782…均先于唯一执行；运行后没有改运行时代码。

producer内wall11.418s、进程CPU11.531s（非整条启动/分析链耗时），worker HWM1612.86MiB、最低采样MemAvailable7894.04MiB；未观察OOM。环境step P50/P95/P99/Max=1.168/12.099/12.499/14.941ms。按每个policy step记录的最近一次扫描耗时P50/P95/P99/Max=6.524/6.769/7.490/7.523ms，编码4.146/4.394/6.895/7.196ms；这些701次“最近样本”会重复同一10Hz帧，不冒充288次独立扫描的分位数。历史测试进程HWM1921.86MiB不是本次物理基线内存。

grant/head/events所有SHA前后不变：累计保守预留327680步/18000s，剩余1,672,320步/68,400s。Test/Validation/训练/optimizer均0；旧模型、L9失败及87历史untracked保留。图在mujoco/reports/autonomous_rl_short_goal_r5/figures：trajectories、goal_stop、timing，全部从实际物理轨迹绘制。

## 【8】问题分析

CONFIRMED：0.35m直接派生存在几何拒绝，统一0.20m静态及真实动态均通过；原物理/Reward保持。手写state-only反馈能在约2.3s内获得原成功事件。SUSPECTED：这能提高PPO成功反馈密度，相比原约8–12s手写基线有更短信用路径。UNVERIFIED：PPO是否真正学会停稳、课程能否转移到原距离/避障；非学习可完成性PASS不自动证明这些假设，也不能从两个任务耗时直接证明GAE是唯一原因。

## 【9】局限

理想传感器/仿真自身状态，state-only Scripted不是避障专家；新短目标可能位于障碍前，不是绕障成功。6个派生Validation仍很小，不验证矿井泛化。真实MID360S/逐点时序/FAST-LIO2/PX4/硬件均NOT TESTED。旧cv2/NumPy ABI可选导入stderr、Axes3D warning保留，不改系统依赖。不能精确续训。

## 【10】下一阶段

R5实际证据/报告/独立结果复审/Commit/Push远端核验后，不在普通阶段结束。重新核验grant/stop/资源，自动进入独立R6：固定发布的R5 proof SHA、扩展固定新课程执行协议与入口，先无更新回归和独立预审，随后在原预算执行新65536步课程。前32768短目标、下一自然reset恢复原12目标；std.3/随机均值其余算法同R3历史对照，不重跑旧控制臂。Final选择，Test0，失败保留证据不自动重试同attempt。

## 【11】给项目负责人的总结

先确认“更近的目标”确实能靠原物理控制器到达和停稳，再让PPO尝试学习。目标不能随意放到障碍里，我们已在正式运行前发现并纠正这个设计问题。即使手写基线成功，也不能说神经网络已学会导航。后续会在原已授权预算内继续冻结新课程和真实训练，不需要每完成一个普通阶段再问你；不会启动PX4或实机。
