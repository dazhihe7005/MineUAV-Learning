# AUTORL-R5 短距离到点课程设计与实施计划

> 使用writing-plans与executing-plans，Native实施，TDD和独立只读审查。用户已明确授权在累计预算内连续推进普通研究，不重复询问阶段执行；独立安全准入不省略。

## 研究假设与边界

R4发现有限固定背景下训练后目标反馈有偏置，不能唯一归因Reward或学习率。候选假设：原约2m到点/停稳的信用分配对当前PPO较难，先练0.20m目标能提高成功反馈密度，再恢复原任务。对照为已完成R3 RANDOM/std.3，不重复旧训练。仅改变目标距离课程；原12个Train的起点、初始Yaw、障碍、场景抽样、原6Validation均保留。既不额外简化Yaw，也不修改Reward、CNN、PI、PPO或动作/频率。

替代方案：直接减学习率可能缓解解析Actor的大KL，但同时改变探索/优化问题不便解释；BC训练目前预算折算未实现。不先实施这些方案。此阶段先做独立物理可完成性，再自动进入训练执行阶段，不停在规划文档。

## 固定任务与未来对照

从原L6冻结清单逐行派生：target_short=start+0.20*(target_original-start)/norm(target_original-start)。同高目标，保留所有其他字段。新row另记录base identity与DISTANCE_ONLY_SIM_TASK来源并重新hash，不改原清单或旧报告。12 Train短目标、6 Validation短目标；Test不生成派生任务，也不运行。几何可达性仅供审计，绝不作为网络输入。

执行前设计修订：最初0.35m候选在静态几何测试被拒绝，Train41004/41009与Validation51001不满足0.20m独立几何裕量；原始静态证据geometry_red_01保留。改为统一0.20m，障碍／方向／裕量／原成功标准不改，18个派生目标全部重新检查；没有实际实验先运行再降低阈值。0.20m起始距离仍是成功半径0.10m的两倍，不能reset即成功。

未来候选训练：新的随机seed0、初始std.3、65536步/3600s/CPU单worker；0–32767步只抽样12短目标，下个自然reset≥32768恢复原12目标。旧episode目标不在中途改变。每128步完整rollout更新，其他PPO不变。不是R1的empty/混合排序实验。Final选择，不挑最好Validation。

在训练前冻结新协议及新身份，原grant不变。建议事前指标：短目标Validation final成功至少4/6、碰撞至多1/6且物理/传感器失败0；original Validation成功>历史0且碰撞≤历史6；学习优于自身step0短目标成功数。三个条件分开评价，失败仍继续有证据的下一假设，不自动重试本attempt。

## 架构与资源

`short_goal.py`纯标准库派生与完整身份验证→`short_goal_env.py`复用MultiScenarioTrainingEnv/LidarNavigationEnv和HeadingObservation、只在reset选择冻结目标→`short_goal_feasibility.py`独立一次12 Train短任务Scripted物理测试。共享flock、原资源门槛、唯一目录、父进程/独立lifetime guard、失败现场。PPO.learn/train与Adam.step禁止；0 optimizer/Validation/Test，预算字节不变。每个episode≤375策略步，总≤4500步，guard180s；实际mj_step/raycast，不设置qpos运动回放。

## Task1：固定目标清单与来源

文件：新增`research_regression/authorized_rl/short_goal.py`、`tests/research_ops_v2/test_short_goal.py`。

- [ ] 先写7项真实行为测试并RED：手算[0,0,1]→[2,0,1]派生[.20,0,1]；12/6/无Test且原清单不变；非法/NaN/零距离拒绝；旧身份/字段篡改拒绝；0/32767/32768阶段规则；缺准入与错误可完成性拒绝。
- [ ] 最小实现`short_manifest(prereg)`、`verify_short_manifest(manifest,prereg)`、`stage_for_steps(steps)`、`verify_short_checks(registration)`，GREEN。完整原prereg SHA仍由现有load_l6_prereg核验。

## Task2：真实环境reset隔离

文件：新增`short_goal_env.py`、`short_goal_tests.py`。消费Task1清单；产生`make_short_scheduler(prereg,upstream,seed)`。

- [ ] 先RED：实际reset目标范数.20、起点/机体Yaw/障碍不变、state7目标误差正确、雷达真实24k/时间0；实际step20个物理tick/无未来帧；重复reset积分/缓存清空；强制测试计数跨32768仅在reset换目标、不消耗额外RNG；Validation/Test强制reset拒绝。使用FORCED_TEST_COUNTER，不谎称运行32768步。
- [ ] 最小实现只reset目标覆盖、双base/task身份、visit计数、阶段切换和scheduler_state，GREEN。绝不修改历史导航环境/控制器/Reward/终止。

## Task3：可完成性执行与准入

文件：新增`short_goal_feasibility.py`，固定唯一`.research_local/autorl_r5_short_goal/run_01`。

- [ ] 为不完整或不满足原成功条件结果写RED判定测试，再实现`verify_feasibility` GREEN。
- [ ] 冻结源码、清单、非训练协议及未来训练建议；完整无更新测试，独立预审GO、OPS登记、doctor通过才执行唯一12episode。
- [ ] 逐episode保存实际轨迹、目标、终止class/raw、reward、物理ticks/扫描时序及资源；不调用模型优化器。12/12按距离<.1m、速度<.15m/s连续5step成功才允许后续训练准入。

## Task4：交付并继续

- [ ] 11节中文MD/JSON/真实轨迹与停稳图、更新六份长期文档，独立结果复审、显式Commit/Push/远端核验。
- [ ] 重新审计grant/停止/资源；无阻塞即进入独立PPO执行阶段，实现固定协议入口、预审后在原预算执行新课程对照，而非结束整个任务。

## Review Focus

目标距离变更必须产生新任务身份，不能冒用原row；task/base几何身份不能混淆；阶段不能在旧episode中途切换；reset不多抽Train RNG；派生Validation不进入训练；未知终止和异常仍fail-closed；近目标不能在reset就满足原成功阈值。
