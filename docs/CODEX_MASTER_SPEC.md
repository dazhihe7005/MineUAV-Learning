# MineUAV-Learning 长期自主研发总协议

Version: **MASTER_PROTOCOL_V1.0**. Source: 项目负责人于2026-10-15给出的长期总指令。此文件是整理后的持久化执行规范；后续会话应先阅读它，再运行 `./researchctl status` 和 `./researchctl recover`，不得依赖聊天记忆恢复权限。

## 1. 目标与工作方式

长期目标是基于LiDAR感知、运动状态及目标信息的无人机自主避障、抵达与停稳。流程为问题→证据→可反驳假设→预注册→实现→测试→获授权实验→分析→Markdown/JSON→Commit→Push→核验远端→判断下一项权限。普通开发无需反复询问；权限、预算或风险等级扩大必须由用户明确批准。不得为了复杂性添加算法，也不得承诺100%成功或真实部署安全。

## 2. 仓库身份与历史

- Repository: `https://github.com/dazhihe7005/MineUAV-Learning`。
- Worktree: `/home/keyousuke/MineUAV-Learning-px4-sitl`。
- 用户总协议给出的历史基线：`fix/lidar-ppo-terminal-accounting` / `77e921f15a3e275df448641672f348991d27a3ce`。
- OPS实际开始时：`test/lidar-ppo-repaired-integration` / `984a1fd46609dca510eee650ac46bef57fa5e1cf`。这是上项基线的直接后继，已在此前会话完成L9S并推送，经Git及本地证据核实；不回退或重跑。
- L9正式Seed0永久保持FAIL；119552是最后资源日志进度，不是已证实的最终完成步数。保留attempt、中间checkpoint及全部失败证据，禁止补造complete或无损续训。
- L9R修复SHA：`77e921f15a3e275df448641672f348991d27a3ce`；协议SHA256：`e267406ec4203353b4ee0c7f8349b57759ef64b31bd076c3db57b99b3682678f`；`formal_retrain_authorized=false`不可由助手更改。
- L9S为已经消耗的一次4096步独立Pilot；它不赋予未来训练预算，不将其工程PASS解释为学会避障。

## 3. 权限与预算（本版本的硬约束）

| Level | 内容 | 当前权限 |
|---|---|---|
| 0 | 源码、静态分析、历史只读审计 | 允许 |
| 1 | 普通开发、受限离线MuJoCo、测试、报告、Commit/Push | 允许 |
| 2 | 有界RL Pilot | 未授权 |
| 3 | 正式RL训练 | 未授权 |
| 4 | PX4-MuJoCo飞行仿真 | 未授权，另需安全审查 |
| 5 | 真实无人机 | 本协议不授权 |

`authorized_training_timesteps=0`，`authorized_training_wall_time=0`，自动训练DISABLED。不得执行调用PPO/其他optimizer的测试。预算账本跨重启持久化；新实验ID不能刷新累计预算。未来训练必须明确单次及累计timesteps/wall time、seed、worker、算法、有效范围、内存/磁盘门槛，并冻结新协议；用户明确授权后才可使用。公开摘要、checkpoint或CLI状态文件不能自行授予权限。

## 4. 本轮最高优先级OPS-01

在 `feat/research-operations-v1` 建设Research Operations and Persistence。实现并测试 `researchctl status/doctor/start/request-stop/recover/verify/sync`、状态机、原子状态、flock独占、异常恢复、停止请求、零预算、持久化依赖、文档及Git同步。只允许离线软件与模拟进程测试；禁止RL更新/Pilot/正式训练、PX4、FAST-LIO2、真实设备。OPS-02若已在本轮实现并验收，不重复建立无意义阶段。

## 5. 恢复、锁及停止

机器状态存于Git忽略的 `.research_local/research_state.json`。至少保存project/phase/experiment/run、branch/base/latest commit、协议/场景哈希、阶段状态、权限及剩余预算、seed/名义预算/已核验实际步数及证据、checkpoint及恢复类别、有效报告、最近成功push、下一动作、更新时间、关机状态。未知项用UNKNOWN/NOT_VERIFIED，不以0伪造。状态只原子发布，不包含凭证。

阶段状态仅为PENDING/RUNNING/STOP_REQUESTED/SAFE_STOPPED/INTERRUPTED/BLOCKED/PASS/FAIL/SYNC_PENDING。使用实际内核flock，不能只靠PID。检查进程及启动身份、锁、attempt、目录及完整性；不确定则BLOCKED，不杀用户进程，不删除attempt。

request-stop应停止新调度，在安全边界保存真实证据、状态、日报及下一步，提交允许文件并Push核验。正常停止不等于实验完成。SIGKILL/断电后recover审计，禁止自动认定安全停机。Push失败为SYNC_PENDING，保存本地Commit，下次优先同步。管理工具不能让Codex在关机时运行，也不注册开机服务或后台训练。

## 6. 恢复类别

- EXACT_RESUME：权重、optimizer、所有RNG、MuJoCo、PI、LiDAR、episode、rollout、场景调度全部可恢复且通过等价性测试。
- BOUNDARY_RESUME：经实际验证的完整rollout边界恢复。
- WARM_START：旧权重初始化新实验，必须新ID，不冒充续训。
- RESTART_REQUIRED：另获新授权后按冻结协议从指定seed重启新attempt。

当前未验证EXACT_RESUME或BOUNDARY_RESUME。L9/L9S checkpoint的存在不授权继续训练。优先单seed当天完整结束，不优先大改SB3内部恢复。

## 7. 固定系统及依赖

MuJoCo3.14.0、约7kg、500Hz物理、100Hz PI、25Hz策略、10Hz LiDAR、24000射线、3×12×24编码及异步缓存；原CNN-PPO、12Train/6Validation/6Test。MuJoCo-LiDAR固定提交 `e6b879df4b96daaed30ef10fe02650d29048fa9b`、模式SHA256 `9fa0165576f0060488254f04bd647a891004d4f4faa6e9b03b69d830c117c9a3`。不依赖易失/tmp；记录真实解释器、包路径、版本、环境变量及源码身份。不得系统升级Python/Torch/SB3、sudo pip或改swap。现有cv2/NumPy ABI问题继续记录，不无依据升级。

## 8. 每阶段与每日交付

每阶段无论PASS/FAIL/BLOCKED均有独立 `docs/PHASE_<ID>_<NAME>.md`、JSON，更新README、EXPERIMENT_LOG、PROJECT_PROGRESS、CODEX_NEXT_ACTION、DAILY_PROGRESS。阶段文档包含目标、背景、架构、代码、实际相关算法、实验协议及身份、真实结果、问题分级、局限、下一阶段和10分钟内可读的中文负责人总结。

完整完成条件：实现、相关测试、真实证据、全部文档、Commit、Push以及远端SHA核验全部成立。否则INCOMPLETE；Push失败SYNC_PENDING。不要为了日报将未完成阶段标PASS。日报包含日期、工作、阶段、Commit、状态、测试、未完成项、阻塞及次日第一步；不是完整训练快照。

## 9. Git与数据保护

保护历史MJCF、BC/PPO/WorldModel、L9/L9R/L9S、raw NPZ、日志及用户untracked。建立含路径/大小/SHA256的本地基线；比较实际清单，数量不等于损坏。当前历史主工作树有87项untracked，禁止删除/覆盖/移动。每阶段独立分支，不改main，不force push，不 `git add -A`。显式核对待提交文件及敏感信息。

提交源码、配置、Markdown、JSON汇总、小图及哈希；模型、大NPZ/rosbag/轨迹/日志默认local-only，登记路径/大小/SHA/实验ID。本地保存不等于远程备份；私有备份需另行授权。

## 10. 后续研究路线与纪律

OPS→已授权有界集成→独立正式Seed0预注册→训练分析→单变量改进→多seed→泛化鲁棒性→PX4迁移准备。现有L9S可作为历史有界链路证据，不能因此再运行4096步。后续可候选Curriculum/BC初始化/Teacher-Student/观测历史等，必须先定位环境、观测、Reward、探索和Critic证据，不同时无依据更改多变量。

每次训练登记experiment/protocol/seed/manifest/network/reward/timesteps/wall/memory/disk/checkpoint/validation/test及初始化来源；冻结期间不改Reward、CNN、PPO、动作、分布或频率。失败不自动重试/增预算，不从PILOT_ONLY冒充正式随机初始化。主要指标是success/collision/timeout/physical/sensor failure、路径、距离、动作及PPO诊断，不以Reward代替成功。单seed、少场景不外推真实矿井泛化。

Train可训练，Validation可诊断，封存Test只在预注册最终评估使用。网络不获得地图、障碍坐标、未来帧或最优路径；Privileged Teacher必须明确标识。Test使用后继续开发须保留旧结果并预注册新最终评估集。

## 11. 资源、失败与审查

默认单CPU worker；MemAvailable<1.5GiB、磁盘低于协议阈值、NaN/Inf、未知终止、身份改变、checkpoint损坏、Test越权或OOM风险立即停止。保存attempt、可靠日志、checkpoint、异常及精确证据边界；Confirmed/Suspected/Unknown分开。资源日志不冒充最终step。记录器失败须披露。

阶段末核验源码/协议/统计/历史身份、运行适用回归及尽可能独立只读审查；无独立执行器时标SELF_REVIEW。Critical/Important必须解决。SIGTERM测试为INCOMPLETE；排除项逐项登记，不称全仓PASS。

## 12. 人工日常操作及长期边界

早上status→recover→doctor→start已授权任务；晚上request-stop→verify→sync确认后关机。用户不用手改JSON。CLI不自动启动跨会话AI；由用户开启Codex阅读本文件与NEXT_ACTION。普通L0/1可自主继续，遇到需要训练/飞行的新预算则记录具体BLOCKED及所需授权，不能自批。

本版本变更须新版本、说明原因并保留Git历史。涉及训练权限、预算和资源上限必须用户批准。当前最终固定：`AUTO_TRAINING_AUTHORIZED=NO`、`TRAINING_BUDGET_AVAILABLE=0`。
