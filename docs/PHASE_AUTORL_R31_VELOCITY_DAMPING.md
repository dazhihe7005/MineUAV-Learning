# AUTORL-R31：显式平移速度反馈诊断

## 【1】阶段目标

验证R30两个Yaw清零仍超时的empty病例，增加固定速度负反馈后是否满足原停稳标准；不是PPO训练。

## 【2】技术背景

R30已发布64fe3ed22281f3929f8bb3efa36a47e864b85dc9。两个控制实际进入.10m球却速度约.345/.350m/s、联合停稳样本全0；仅清Yaw不足。R29函数probe未支持远距离输出塌缩，不能据此认定唯一根因。本轮复用R30既有逐步真实控制数据，不重跑旧实验。

## 【3】系统架构

R28最终可信网络只读推理→原clip动作→Yaw清零→当前ENU速度负反馈→原PI→真实MuJoCo物理与LiDAR→逐步记录。500/100/25/10Hz、24000射线、7kg历史模型不变；不是运动学qpos轨迹回放。

## 【4】代码实现

velocity_damping_diagnostic.py冻结病例/旧控制/协议；velocity_damping_trace.py显式DampedTranslationPolicy与float32动作函数；velocity_damping_run.py核源码/OPS/资源/唯一attempt与进程生命周期，禁止PPO.learn/train/Adam.step。6新增测试真实RED→GREEN。独立本地分析器复用固定R26/R27算术并独立IEEEfloat32逐运算复算，验证完整状态链/动作/每个冻结Reward分项与停稳计数；另2纯分析契约测试使用明确unit-adapted旧数值，非新物理实测。

## 【5】算法原理

先clip原网络动作，保持R30的Yaw动作0，再施加 a_xyz=clip(a_original_xyz−0.3·v_ENU/[1.5,1.5,1],−1,1)，float32实现；对应物理指令修正−0.3·v(m/s)，不是不分尺度的normalized扣减。反馈读取当前合法SIM_GROUND_TRUTH_STATE7速度，不读取未来状态或障碍地图。0.3在观察本轮结果前固定：R29局部位置增益约.43/s与.10m/.15m/s边界给出约.287，四舍五入，仅启发式诊断，不是全局系统辨识/稳定性认证。不改PI、网络权重或Reward；成功也只是hybrid动作效果。

## 【6】冻结协议

REG `328721f314b7a3c8b560c24735826c0852ccc4779e98300948c03c166a1e08b6`；178SOURCE `58a0c94fe2839482952c22a21316c154df99c1aadfd88161f0d415ca1cae04db`；EXEC `35e551a8409baae94aed70c0d41f902c35cdcee39fdbc0e8fd891cf3efc068e8`。固定final模型 `f7349c40158b1f064542bdf8180f22cf9b68f11aee9ab563f732e678bfc85475`，两held原距离empty，CPU1/内部180秒/每例≤375策略step。单变量相对R30为D=0→.3，其余Yaw清零/模型/任务/初始条件不变。距离<.10m、速度<.15m/s连续5step且无物理/传感器失败，两例均成功才PASS；不事后调阈值。训练/optimizer/Test=0。

## 【7】实际结果

工程PASS；冻结双成功对照 **FAIL**。两条新轨迹合计750策略step/15000实际tick，新成功0/2；R30旧控制0/2。原R28科学FAIL不改。

|新例终止|策略步|实际tick|终点误差m|终点速度m/s|最长联合样本|路径m|
|---|---|---|---|---|---|---|
|time_limit|375|7500|0.07593670983093022|0.4336434424708809|0|6.308775972331777|
|time_limit|375|7500|0.07638633071613764|0.4481670923785633|0|6.526651585704636|

每条初始qpos/qvel、模型身份及首个原始mean与既有控制相符，干预后轨迹/观测可以不同。全部新行独立复算目标误差、实际速度、state/heading、当前反馈速度、float32clip次序、ENU物理动作映射、LiDAR因果性、每个Reward分项及连续停稳条件，case与result总量一致；图表均来自新旧真实射线物理轨迹，不合成理想轨迹。

worker墙钟17.593s、CPU17.748s、生命周期HWM1409.988MiB，逐策略guard最低MemAvailable8909.191MiB；outer墙钟25.163s单列，不等同长期内存证明。1240选择PASS=266OPS+89无更新Tensor+885历史，15排除、非全仓；2本地分析测试另计，所有测试优化器0。507历史文件及原173SOURCE不变；账本head32/余额951424步/28800预留秒stopfalse。

## 【8】问题分析

CONFIRMED：真实施加显式负反馈及双成功判据FAIL。若PASS，只支持反馈对该两病例停稳的贡献，不证明训练得到该能力；若FAIL，保留反证，不根据失败自动调大gain重试。SUSPECTED：原网络缺少适当速度反馈与停稳失败有关。UNVERIFIED：唯一根因、其他任务/随机种子、LiDAR依赖与避障能力。

## 【9】局限性

单模型/两个empty/显式hybrid；不是学习成功、泛化或实机安全验证。当前速度为仿真自身真值，迁移需要独立估计状态验证。没有新增Test/设备/PX4/FASTLIO，不改变历史L9/R24永久FAIL。caught cv2/NumPyABI与Axes3D警告仍记录，不升级系统、不改swap；RESTART_REQUIRED不是无损续训。

## 【10】下一阶段

真实独立结果复审、18文件Git检查/Commit/正常OPSsync/远端核验之后，查budget/stop/resource，依据结果冻结下一可反驳研究。若显式反馈支持停稳，可研究独立新初始化先验中的速度反馈，而非把wrapper成功当网络学习成功；新训练另协议/审查/实际预算预留。不得重做本次attempt或读取Test。

## 【11】给项目负责人的总结

本轮只在原网络命令后加入当前速度的反向修正，让真实MuJoCo检验两次能否停稳，结论FAIL。没有花新的训练步数，也没有改旧模型或控制器；即使有效，也还需证明网络经过训练能保持/学到停稳，不能直接称无人机已学会避障。接下来在原剩余授权内按证据推进，先完成同步与独立审查。

最终独立只读结果复审GO：C0/I0/newMinor0，限两例显式速度反馈诊断与结果边界，不认证原网络学会导航或新训练。18文件实际Git检查/Commit/正常OPSsync/远端以真实收据为准。

发布补记：第一次metadata seal后，原OPS readiness拒绝报告遗漏的review.source_identity_sha256及tests.evidence，尚未stage/Commit/Push，不能称阶段完成。原失败报告/MD字节及结果GO收据另存不覆盖。仅补齐实际已通过的266OPS证据引用和同一178SOURCE身份，科学FAIL、1240测试/图/raw/运行代码不变；重新等待独立发布复审。额外1个本地readiness契约真实RED02复现，RED01仅是直接脚本导入路径错误，均如实保留；不混入1240测试数。

发布元数据独立复审GO C0/I0/newMinor0；原OPS实际readiness RED→GREEN，仅补证据绑定，运行身份/统计/科学FAIL不变。最终发布仍以真实Commit/OPSsync/远端收据为准。
