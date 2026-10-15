

## 当前阶段：AUTORL-R1 课程对照

两臂各65536真实步已完整结束：A398ep=377碰撞21超时；B278ep=221碰撞57超时；Train及最终6Validation均0成功。课程假设未获支持（FAIL），不是用更高Reward报告改善。独立原Train empty基线3／3成功。真实累计131072步，余1,868,928步／79,200秒；后续先冻结教师初始化候选并测试，不修改本次协议或旧L9。

## 历史阶段：AUTORL-01 授权执行机制

新用户授权累计200万训练步／24小时，CPU单worker，不改历史零预算。新机制完成测试／独立审查及远端同步后，先empty实际基线，再执行两臂课程对照；源码、协议和余额见新阶段报告／CLI。本阶段不产生算法性能提升结论，不自动重启L9。

## 历史阶段：AUDIT-01

分支 `research/lidar-ppo-learning-diagnosis`；基线 `6258dd4ecae09e2614d9c9b47f9eebd52b6b6bee`。完成已有LiDAR-PPO只读学习失败诊断，不是新的训练实验或导航性能提升。

确认：空场景也0成功；有实际运动，不能认定学会悬停；严格成功反馈未出现；原记账Bug确认导致中断但不解释全部学习失败。Reward、有效探索、部分可观测性、更新强度及任务难度须区分假设，不能仅根据一个PPO快照归因。

下一项建议为独立预注册的课程调度对照：两臂各65536、合计131072步，当前**未授权、预算0**；不直接重跑原百万步，不从失败checkpoint无损续训。报告：[AUDIT-01](PHASE_AUDIT_01_PPO_DIAGNOSIS.md)。测试与审查结果见JSON；最终Commit/Push完成仍以V2收据为准。以下保留旧阶段记录。

## 历史阶段：OPS-02

分支 `feat/research-operations-v2`，基线 `37b6d90e84ba0554db3574c9913413ee93c58e41`。实现便携回归及独立V2登记；885项相关历史回归、49项V1管理测试、34项新增V2测试通过；15项历史排除保留明细。最终发布状态以 `researchctl-v2 status` 的 `phase_complete` 和远端SHA核验收据为准。

训练预算0、自动训练NO，L9保持FAIL，L9R/L9S及模型原始数据不变。此阶段不提升避障成功率、不证明实机部署能力。下一项仅允许已登记Level0/1离线审计；无权自动训练。详见[阶段文档](PHASE_OPS_02_PORTABILITY.md)。以下OPS-01记录保留。

<!-- RESEARCH_OPS_V1_BEGIN -->
# 项目进度

阶段：OPS-01；状态：SAFE_STOPPED。

分支：`feat/research-operations-v1`。快照基于Commit：`37b6d90e84ba0554db3574c9913413ee93c58e41`。

当前Commit以 `git rev-parse HEAD` 为准；快照不能包含它自身尚未创建的Commit SHA。最近成功推送及实时完成状态见 `./researchctl status` 的核验收据。

训练授权：NO；剩余timesteps/wall-time：0/0；optimizer更新：禁止。

历史L9保持FAIL；L9R修复、L9S4096步链路Pilot已记录。L9S无成功导航，不能当作避障能力。

安全下一步：`./researchctl doctor` 后运行 `./researchctl start --task ops-audit`，或在Codex中只读分析已有实验。 新的训练需要用户提供独立预算与协议。
<!-- RESEARCH_OPS_V1_END -->

## OPS-01验收记录

49项OPS测试通过，885项历史相关选定测试通过；15项历史排除单列，不能称整个仓库全测通过。实际工作区的初始化、依赖恢复、维护、恢复及核验通过。初次混路径旧回归失败保留并解释；后续测试路径迁移作为独立L1工作。

独立初审6项Important已通过RED→GREEN修复，最终闭环为SELF_REVIEW，不宣称独立最终认证。详细证据：[阶段报告](PHASE_OPS_01_PERSISTENCE.md)、[机器报告](../mujoco/reports/research_operations_v1.json)。最终Git交付以实际HEAD及已核验远端SHA收据为准。
