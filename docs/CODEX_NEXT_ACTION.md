

## OPS-02交接：优先读取本节

项目 `/home/keyousuke/MineUAV-Learning-px4-sitl`，当前分支 `feat/research-operations-v2`。先读 `CODEX_MASTER_SPEC.md`、`RESEARCH_OPERATIONS_V2.md`、本阶段JSON及 `./researchctl-v2 status`，实际HEAD以 `git rev-parse HEAD` 核对；提交自身SHA不写回静态报告以免循环。

下一次第一步：`./researchctl-v2 recover`，然后 `doctor`。若 `SYNC_PENDING`，先运行 `sync`，不改待推文件；任何身份或历史冲突BLOCKED，禁止reset或重新init。V1本地文件是只读历史，不要继续运行会改变它的旧recover/start/sync。

权限Level0/1，训练剩余timesteps/wall-time=0/0；不允许PPO optimizer、Pilot、正式训练、PX4。可运行 `./researchctl-v2 start --task audit` 维护核验。新增阶段用prepare→只读审查→register，不能直接改protocol或预算。

最新已知有效模型为历史L9S `L9S_REPAIRED_PILOT_ONLY_SEED0.zip`（身份由V1配置只读校验）；不是正式Seed0最终模型，不允许自动Warm Start。当前训练恢复仍RESTART_REQUIRED，不具备精确续训。最近代码问题为本阶段审查发现的启动窗口，最终关闭状态看JSON与审查证据。以下V1块仅为保留历史快照。

<!-- RESEARCH_OPS_V1_BEGIN -->
# 次日恢复入口

先读 `docs/CODEX_MASTER_SPEC.md`。

阶段：OPS-01；状态：SAFE_STOPPED。

分支：`feat/research-operations-v1`。快照基于Commit：`37b6d90e84ba0554db3574c9913413ee93c58e41`。

当前Commit以 `git rev-parse HEAD` 为准；快照不能包含它自身尚未创建的Commit SHA。最近成功推送及实时完成状态见 `./researchctl status` 的核验收据。

训练授权：NO；剩余timesteps/wall-time：0/0；optimizer更新：禁止。

历史L9保持FAIL；L9R修复、L9S4096步链路Pilot已记录。L9S无成功导航，不能当作避障能力。

安全下一步：`./researchctl doctor` 后运行 `./researchctl start --task ops-audit`，或在Codex中只读分析已有实验。 新的训练需要用户提供独立预算与协议。

项目目录：`/home/keyousuke/MineUAV-Learning-px4-sitl`。协议：`59964b4d2bba385308e62ff0b9f9d8d0818e53ba767aaf398ec20fa95007a5a8`。

最新checkpoint：本地冻结L9S Pilot；其文件身份由doctor核验，不授权初始化正式模型。训练中断恢复策略RESTART_REQUIRED；未实现EXACT_RESUME或BOUNDARY_RESUME。

早上：`./researchctl status` → `./researchctl recover` → `./researchctl doctor`。

晚上：`./researchctl request-stop` → `./researchctl verify`；若SYNC_PENDING，运行 `./researchctl sync`。

CLI只运行登记的维护任务；AI开发需用户启动Codex。无开机服务或无人值守训练。
<!-- RESEARCH_OPS_V1_END -->

## 下一项研究工作

优先只读审计旧回归中的固定 `/tmp/mineuav-mujoco-lidar-audit` 路径，设计可配置、可复现的测试入口迁移。当前OPS入口已使用持久化依赖；不要为让旧测试通过而放宽扫描器身份检查。若需修改源码或登记新阶段，保留V1协议/attempt并另行版本化，当前V1不提供静默rebaseline。

L9S已在前一阶段完成，不能再次执行其4096步Pilot。当前训练授权NO、预算0、optimizer禁止。原L9保持FAIL。下一次训练预算和任何PX4/设备任务须由用户另行授权。
