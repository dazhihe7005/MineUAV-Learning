## AUTORL-R1：当前执行优先于以下历史

**已结束**：两臂各65536步均完整，最终Validation均0成功6碰撞，课程改善FAIL。不得再次start两个R1 ID。预算真实余额1,868,928步／79,200秒；两次预留完成记录均在独立账本。先完成本阶段结果独立复审、Commit/Push远端核验，再进行新注册研发阶段。

下一候选：解析state-only教师初始化的单变量诊断。仅改变Actor初始化，原结构／Reward／PPO／std不变；先检验两层Tanh能否在预注册容差内逼近已成功Scripted反馈（不是已证明的精确等价），不调用optimizer；再冻结新实验预算和研究评价。现有CLI尚只支持固定R1，必须先开发／测试／新版本登记／独立GO，不能改R1协议或复用其attempt。普通研发及明确预算内PPO已获授权，不用再次索取普通阶段指令。

当前分支`research/lidar-curriculum-r1`，基线e9b90198732bec00c64173c154bd63b7cbc8c537已推送。R1登记SHA de9a5517c7806fce459411acb9204e010124284d1dd763718e99797684cb7714，冻结协议见`autonomous_rl_curriculum_r1_prereg.json`。同注册版本baseline三条empty Train已真实成功（.research_local/authorized_rl_v1/baseline.json），不重复运行或initialize账本。

上述两个ID已按顺序各执行65536步、各预留wall3600s；这不是待执行指令。先读新预算`python3 -m research_regression.authorized_rl status`；锁被持有或存在未知／失败attempt时不能重放。完整训练结果在`.research_local/authorized_rl_v1/runs/<ID>/`，无complete不能宣称完成。训练器已使用过的ID永久不能自动重试；不是旧L9的续训。

真实剩余额度以新append-only账本为准；公开文字和旧OPS的NO都不是恢复授权来源。训练结束后独立结果复审、报告／图／长期文档Commit/Push和远端核验；若结果未达目标，则先分析并冻结下一新协议，不盲目增步。

## AUTONOMOUS-RL-RESEARCH-V1：授权历史与操作说明

用户已于AUDIT-01之后明确授权：累计2,000,000训练environment steps／86400s wall，单次<=1,000,064、CPU单worker；不授权PX4、FAST-LIO2、实机、Test调参。读`CODEX_AUTONOMOUS_RL_V1.md`、本阶段机器JSON及本地新预算，不能从以下历史NO推断新授权丢失，也不能用本段恢复账本。

工作目录不变，当前分支`feat/autonomous-rl-research-v1`。先`git status`及`python3 -m research_regression.authorized_rl status/doctor/recover`；OPS-V2仍负责维护／历史核验和显式文件发布，其NO/零预算只是旧范围。不得改旧L9R denial或V1/V2账本。

新入口和独立预审完成后，下一安全操作为在新阶段登记下运行三个empty Train的state-only Scripted基线，全部原标准成功再开展两臂65536步课程对照。无SMOKE/PILOT/L9权重初始化，B在32768后下个自然reset切换；两个checkpoint延后至Train关闭后分别Validation，Test=0。完整细则在设计文件。预算先预留全额、失败不退不重试；未知attempt必须recover审计。每阶段先Commit/Push/远端SHA核验再推进下一阶段。

停止用`python3 -m research_regression.authorized_rl request-stop`；此版本停止锁存，不自动清除或重跑。最新Commit和实际阶段完成读V2收据；不存在必须续训的旧checkpoint。以下为历史快照，不构成当前执行权限。


## AUDIT-01交接：优先读取本节

位置 `/home/keyousuke/MineUAV-Learning-px4-sitl`，分支 `research/lidar-ppo-learning-diagnosis`，实际最新Commit读 `git rev-parse HEAD` 及V2同步收据。先读总协议、V2协议、[本轮中文诊断](PHASE_AUDIT_01_PPO_DIAGNOSIS.md)与JSON，再运行status→recover→doctor；若SYNC_PENDING优先sync，不改已封存待推文件。

当前授权Level0/1、训练预算0；**不允许任何PPO/监督optimizer更新、Pilot、正式训练、Test episode、PX4或设备通信**。L9永远FAIL，L9R denial不变，L9S只是4096步链路Pilot；中断策略RESTART_REQUIRED，不能加载L9中间checkpoint继续原实验。

最推荐的下一次实验是“仅改变场景调度”的课程对照：新初始化seed0，两臂各65536步，总131072；单worker顺序执行、wall总上限7200s建议，原Reward/CNN/PPO不变。**这只是提案，不是可执行授权**。需项目负责人独立批准预算，随后新协议和安全预检；不自动运行。

仍可执行已登记的 `./researchctl-v2 start --task audit` 做历史核验。新分析原始结果在 `.research_local/audit_01/run_01/`，已使用目录拒绝覆盖；不把原始数据粒度不足用新训练或伪造轨迹补齐。没有必须自动追加的研究阶段，本轮交付后停止训练相关工作。以下为历史交接，发生冲突以本节和真实本地状态为准。

## OPS-02交接：历史记录

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
