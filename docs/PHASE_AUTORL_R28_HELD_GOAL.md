# AUTORL-R28：保持初始目标航向的新任务学习

## 【1】阶段目标

检验PPO能否在新“保持起始航向”的任务上学习、保持停稳。工程PASS、五冻结科学判据FAIL；不等于原目标Yaw0任务成功。

## 【2】技术背景

R26既有原距离empty先验两例均超时；R27只改目标航向后相同先验两例均193步成功。此证据支持研究任务定义，不证明控制器bug、唯一失败根因或已学会导航。R24跨boot中断永久FAIL，绝不续训/退款或补complete。

## 【3】系统架构

固定R28 CLI→新失败闭环/实际授权账本→原worker+held-goal task wrapper→原MuJoCo/PI/LiDAR→关闭Train后独立新任务Validation。单CPU/单worker/最多一个resident场景；无Test/PX4/设备。

## 【4】代码实现

新增closed_yaw_failure及held_goal profile/env/validation/cli/tests。budget/runtime/worker仅固定28分支，旧schema17前缀检查仍只17；旧L6环境、控制器、Reward及历史验证器不改。新manifest显式24Train+12Validation任务身份，仅目标Yaw由0改initialYaw。原12base几何/起点/终点位置/初始姿态不改；原采样RNG不增加消耗。checkpoint记录新任务列表SHA与新contract，旧distance accounting另列。visit_counts是12base，task_visit_counts才是24新任务。

## 【5】算法原理

CNN/中心化Actor/手写先验650a与std.1保持R17；170857参数，γ.99/λ.95/LR3e-5，不是旧ZIP初始化。Reward仍NAV_V2_COLLISION_PREFLIGHT_V1：`10Δd−.005||a||²−.5w(d)||v||²+10success−30collision/physical−10sensor−.01step−2timeout`。成功仍距离<.1m、速度<.15m/s连续5个25Hz样本，不随结果放宽。网络只接LiDAR3×12×24、state7、heading2、age1，无地图/障碍坐标/未来信息。

## 【6】冻结协议

ARL_R28_HOLD_GOAL_YAW_SEED0，fresh seed0/CPU1/65536实际步/3600秒全额预留。REG `cc5a35abd5d6a965945a42c7fe0584946bc1df37a98941c4c61e82ded9a604dc`，SOURCE `94daaf6b68f5db9977ede2c5db14660a067549f1e28d41b8ee48fb29aada3fc3`，EXEC `bc2f81ffa7e7fb67ec46a6afe31255300f6ff86ce782168edc3f648e043db3f7`，manifest `3a670bbbc1b841fc81efbc2b6f81f3a58e4d84583a5f5cebaa61820f0e8007b3`。前32768步0.2m课程，之后自然reset切原距离、empty概率.5；500/100/25/10Hz、24000rays与动作尺度不变。三个checkpoint各6short+6original共36新held任务Val，Final Checkpoint，不与R17不同任务分数声称公平同条件改进。原Yaw0迁移NOT TESTED。

## 【7】实际结果

65536步/1310232实际物理tick，512rollout/2048Adam/1024epoch；Actor/Critic更新、梯度有限，三个ZIP实际重载optimizer及预测一致。Train终止{'collision': 45, 'physical_failure': 0, 'sensor_failure': 0, 'success': 490, 'time_limit': 58}；short {'episode_return': {'available_count': 461, 'mean': 10.188552546164958}, 'episodes': 461, 'final_target_distance_m': {'available_count': 461, 'mean': 0.08002711327956805}, 'mean_action_norm': {'available_count': 461, 'mean': 0.19195935330868338}, 'minimum_goal_error_m': {'available_count': 461, 'mean': 0.07590459584531095}, 'motionless': 0, 'terminal_counts': {'success': 460, 'time_limit': 1}, 'trajectory_length_m': {'available_count': 461, 'mean': 0.20657309789187017}}，original {'episode_return': {'available_count': 132, 'mean': -4.985694089226504}, 'episodes': 132, 'final_target_distance_m': {'available_count': 132, 'mean': 0.5324533810863807}, 'mean_action_norm': {'available_count': 132, 'mean': 0.3407506592734886}, 'minimum_goal_error_m': {'available_count': 132, 'mean': 0.4731994292888644}, 'motionless': 0, 'terminal_counts': {'collision': 45, 'success': 30, 'time_limit': 57}, 'trajectory_length_m': {'available_count': 132, 'mean': 3.8028138112350693}}。最后未结束尾部259步不伪造terminal；物理失败可在策略step内部结束，ticks从真实仿真时间而非20×步数还原。

|实际checkpoint步|held short成功/碰撞/超时|held original成功/碰撞/超时|
|---|---|---|
|0|6/0/0|4/2/0|
|32768|6/0/0|0/3/3|
|65536|6/0/0|0/2/4|

冻结五判据{'held_empty_retention': 'FAIL', 'held_original_candidate': 'FAIL', 'held_original_learning_over_prior': 'FAIL', 'held_short_candidate': 'PASS', 'held_short_prior_retention': 'PASS'}，总体科学FAIL。empty子集{'0': {'success': 2}, '32768': {'time_limit': 2}, '65536': {'time_limit': 2}}。1221选择PASS=253OPS+83无更新Tensor+885历史，15排除，非全仓；22新增测试，分析4/发布2纯函数测试另列。真实RED→GREEN覆盖旧worker随机初始化、旧checkpoint列表SHA，以及完整24任务初始化/单步物理与Reward一致、12Val各1步probe隔离。独立最终预审发现分析器只检查final而遗漏早/中Val的物理/传感器失败：新增all36负例真实RED→GREEN后，原零容忍要求统一覆盖36episode，未改阈值/协议/运行SOURCE。开发中旧GREEN名收据曾有message断言FAIL、分析fixture空场景索引/缺省Counter0错误与距离stage识别错误；全部保留、最终完整回归及helper GREEN为准，不假称首次全绿。强制计数器单元测试不冒充已训练32768步。

CLI墙钟1725.025s，worker含Val1704.583s；训练末65536步采样的累计HWM为1704.703MiB，整个worker含延后Val生命周期HWM为1743.793MiB，CLI已回收子树最大单进程HWM也为1743.793MiB。训练采样累计最低MemAvailable8716.840MiB。资源图mem_available_mib是截至各采样点的累计最小可用内存，不是瞬时值。初稿误把summary的生命周期HWM标为Train HWM，独立复审指出后经真实RED→GREEN更正标签和来源；旧草稿、变更收据保留，raw/模型/图/运行SOURCE不改。并发合计峰值与全程CPU%未知，未观察OOM。419旧文件身份不变，实际事件31RESERVE/32COMPLETE，余额951424步/28800预留秒stopfalse。CPU/2D路径保留已知cv2/NumPy、Axes3D警告，不修改依赖。

## 【8】问题分析

CONFIRMED：上述实际结束任务和五判据，不用Reward单曲线代替任务结果。SUSPECTED：航向任务与平移/停稳学习存在耦合，需要独立研究；不是凭两个先验例子断言唯一根因。UNVERIFIED：原Yaw0任务迁移、LiDAR因果依赖、多seed、真实矿井泛化；没有运行Test。

CONFIRMED：同一组新held任务、同一确定性评价协议中，原距离成功数在step0为4/6，在32768和65536均为0/6；两个empty子集从2成功变成2超时。短距离三个checkpoint都为6/6。原距离能力下降在仅完成短距离训练的32768模型上已经出现，不能全部归因于后来混合原距离训练。初始4成功来自手写先验，不能归为PPO成果。全部36次评价均无物理/传感器失败，因此这不是用异常评价缺失补成的零成功。

SUSPECTED：短距离训练的数据范围可能未约束长距离状态下的反馈，或学习改变了接近目标时的速度/制动映射。现有episode聚合不能区分这些机制，不能据此断定Reward、学习率或PI控制器是唯一根因。下一项先做独立、无optimizer的同输入策略函数诊断，不重复训练或复用已完成的36次评价。

## 【9】局限性

公开JSON仅投影三个checkpoint的身份、步数与ZIP/完整sidecar SHA；完整reset/RNG/scheduler清单、权重和23个原始文件均local-only。机械压缩前草稿及非checkpoint字段相等校验保留在本地，不改变统计、模型、图或实际实验。旧sidecar的通用WARM_START标签不是本实验的恢复授权；本实验始终RESTART_REQUIRED。

目标航向变化改变了任务定义与Validation，不是相同旧任务上的严格增益对照。单seed/小开发Val；手写先验能力不能归于训练。仍RESTART_REQUIRED，无exact/boundary resume，SIGTERM保护不覆盖断电/SIGKILL。旧L9/R20/R24永久失败不改。

## 【10】下一阶段

实际结果独立复审、正常Commit/OPSsync并核远端后，再核预算/stop/资源，选择新可反驳假设。不旧ID重试、不延长本轮、不改冻结门槛；不得普通阶段结束就停止长期研究，也不无依据消耗预算。

## 【11】给项目负责人的总结

这一轮把“必须转到世界零航向”改为“保持起始航向”，其他算法和物理保持不变。科学结果FAIL：短距离仍全成功，但原距离从未训练先验4/6成功降到训练后0/6，不能说算法提升了。下一步先查短课程后策略的反馈为何没有保留长距离能力，而不是再盲跑相同训练。真实预算与历史失败都保留，发布后在剩余授权内继续；不会擅自接PX4或实机。

最终独立结果复审GO，C0/I0/新增Minor0；认证本轮65536新held-goal任务训练与36新Validation的真实计数和冻结科学判据，不认证原Yaw0迁移、LiDAR因果依赖或真实飞行泛化。旧R24永久FAIL、不中间ZIP续训。JSON仅追加review/recovery/doctor与audit引用，统计/图不变。28文件显式白名单Commit及正常OPSsync/远端状态以实际收据为准，不预先声称推送。

提交前第二轮同一1221项选择回归也全部PASS（253OPS/83无更新Tensor/885历史、15排除），预算与历史身份未变；两次运行不是2442个不同测试。收据见JSON postrun_regression。
