

## 当前阶段：OPS-02

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
