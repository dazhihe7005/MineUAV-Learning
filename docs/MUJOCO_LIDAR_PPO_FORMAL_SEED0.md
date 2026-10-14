# Phase 3E-L9 — LiDAR-CNN-PPO 正式训练（Seed 0）

本阶段唯一一次 seed 0、单 CPU worker 的正式 LiDAR-CNN-PPO 训练 **FAIL：因终止原因契约不一致而中断**。最后资源记录到 119,552 环境步，未完成授权的 1,000,064 步，未产生最终模型。没有自动重启、warm start、seed1/2、Test episode、PX4 或实机运行。逐项证据见 [JSON 报告](../mujoco/reports/mujoco_lidar_ppo_formal_seed0.json)。原始 SB3 checkpoint、episode/resource JSONL、异常堆栈、里程碑事件与 Validation 原始记录保留在 ignored `mujoco/reports/mujoco_lidar_ppo_formal_seed0_parts/run_01/`，不提交到 Git。

## 冻结协议与准入

基础提交 `53bb653a428f28467863f1e625f493fb643f4edf`，分支 `feat/mujoco-lidar-ppo-formal-seed0`。L8 执行协议 SHA256 为 `fe07816f48bd4183165db482d8c93de50c637a32d13d86eccb47667d85e8cef6`，L8 验收报告 SHA256 为 `a08df1131aaac650482f64e302c5717fad3cf599062fa5356376e02ac6f69c78`。每个里程碑的名义步数为 100,000 的倍数；实际处理在完整 128-step rollout 及对应 optimizer 更新之后。最后名义 1,000,000 步对应实际 1,000,064 步。最终模型选择规则固定为 **Final Checkpoint**，Validation 仅用于诊断，Test episode 必须为 0。

训练直接复用 L8 `FormalPPOTrainingDriver`，新 CLI 只开放 `--formal-seed0` 与显式授权标记，不提供 checkpoint 初始化参数。新随机网络初始化、12 个冻结 Train 场景均匀调度、6 个独立 Validation 场景、6 个封存 Test 场景。每个训练 episode 重置场景、MuJoCo 状态、PI 积分与 LiDAR 缓冲；不把障碍物真值坐标输入策略。Reward 为 `NAV_V2_COLLISION_PREFLIGHT_V1`，原 7 kg MJCF SHA256 `cde862f424a86bda3676249859268c69dede9dde2ebadeef819813351684b520`，上游 MID360 近似扫描模式来自固定 MuJoCo-LiDAR commit `e6b879df4b96daaed30ef10fe02650d29048fa9b`。七个历史 BC/PPO checkpoint 只读核验，不修改。

观测为 `lidar float32[3,12,24]`、`state float32[7]`、`heading float32[2]=[sin(yaw),cos(yaw)]`、`lidar_age float32[1]`；动作是归一化 `[-1,1]^4`，经原环境适配为 ENU 世界速度 `[1.5,1.5,1.0] m/s` 与偏航角速度 `1.0 rad/s`。MuJoCo 物理 500 Hz、PI 100 Hz、策略 25 Hz、LiDAR 10 Hz、每帧 24,000 射线。PPO 为 SB3 2.9.0 `MultiInputPolicy` + 固定 `LidarStateExtractor`；`n_steps=128`、batch 64、2 epochs、lr `3e-4`、γ `.99`、GAE λ `.95`、clip `.2`、entropy 0、value coefficient `.5`、max grad norm `.5`。

正式训练入口在启动前拒绝错误授权、seed、实际预算、协议/模型/历史 checkpoint/扫描模式/依赖身份、已有非空输出目录、内存及磁盘不足。训练中每步执行原有 MemAvailable 1.5 GiB 防线，每 128 步核验 L9 运行源码及磁盘 2 GiB 防线，逐 episode 记录终止类别、回报、动作范数、轨迹长度、净位移与最后目标距离。每个里程碑在保存 checkpoint **之前**拒绝非有限模型参数、梯度或 PPO 训练指标。ZIP 与 sidecar 分别原子写入并即时重载核验；跨文件组合不是事务。中断处理严格为 `RESTART_REQUIRED`，不能把 warm start 称为同一次精确续训。

## 验证边界

训练前独立只读预审在源码漂移和无效参数/诊断的 RED→GREEN 修复后给出 GO；完整相关回归为 **714 PASS、0 FAIL、2 个继承警告**，其中新增 16 项准入及健康监控测试。系统 OpenCV 从旧 NumPy ABI 编译，在 SB3 的可选 Atari wrapper 导入路径产生警告；当前 CPU MuJoCo raycast/PPO 热路径不调用该 OpenCV 功能，未修改系统依赖。长跑的持续内存稳定性仍须由本次实际监控验证，预审 GO 并非性能保证。

## 训练结果与后续审计

唯一 `.attempt` 保留，进程以 `RuntimeError: invalid or nonexclusive training episode outcome`、exit code 1 退出。不存在 `formal_summary.json`、`formal.complete.json` 或最终 checkpoint。每 1024 步写入的 `formal_progress.json` 最后为118,784步；每128步资源日志更晚，最后为 **119,552步**，共934条。最后完整 Train episode 在119,370步结束。确切异常步及 `model.num_timesteps` 未在异常路径持久化，不能把119,552误称为精确训练总步数。由128步记录周期和异常执行位置可界定：成功返回的环境步数在 `[119552,119679]`；失败的环境调用编号在 `[119553,119680]`。这是源码推导范围，不是额外实测。

仅完成名义100,000→实际100,096的首个里程碑。checkpoint ZIP SHA256为 `d71927d83c3f844f035a3059c1329b0df22773a58ad6b258fa1093c2377abff9`，sidecar/event/ZIP身份一致，重载后权重和optimizer均有限；782个rollout、1564个PPO epoch updates、3128次Adam steps得到核验。该文件是**中间checkpoint**，不能替代预注册的Final Checkpoint。对应6个独立Validation场景均碰撞，0成功；Train场景RNG、训练权重及optimizer未被Validation修改。Test episode=0。

落盘的 **706条完整Train episode** 为0成功、656碰撞、50超时、0已记录物理失败、0传感器失败；这些统计**不包含抛异常的终止transition**。已结束episode平均return `−29.389149`，平均轨迹长度 `3.222638m`，平均最终目标距离 `1.785219m`，平均动作范数 `1.437426`。按预先定义的“净位移<0.10m且路径<0.20m”规则，已记录episode没有被归为无明显运动。当前证据没有训练成功提升，也不足以评价百万步结果或算法优劣。12个Train身份都出现在落盘episode中；完整reset访问序列仅在100,096步checkpoint中保存，之后只有已结束episode的身份记录，最后失败episode身份缺失。

## 中断原因及证据等级

**Confirmed — 终止原因接口不一致。** `navigation_env.py:284-300` 将 `z_below_zero`、`outside_flight_area`、`excessive_tilt`、`nonfinite_state` 等原始原因映为 `physical_failure`，用于Reward和互斥状态标志，却保留原始 `info['termination_reason']`。冻结的 `ppo_training_orchestration.py:223-226` 要求这个字段本身属于 `success/collision/time_limit/physical_failure/sensor_failure`，遇到原始物理原因便抛异常。相邻的Validation消费者会把非标准reason映为physical_failure，因此不会触发同一记账异常。

受控离线单元测试在Train环境reset后设置明确越界状态：实际返回的reason为 `outside_flight_area`，恰好一个 `physical_failure=True` 标志；原TrainingMetricsEnv仍抛出本次同名异常。测试没有optimizer更新，也没有重启正式训练。这证明错误信息中的“nonexclusive”不能直接解释为多个失败标志同时为真。

**Unresolved — 本次具体物理触发原因。** 失败transition的原始info、位姿、动作和episode身份没有持久化，不能确认到底是越界、倾角、负高度还是非有限状态。进程的已捕获Python异常说明直接中断原因是记账契约；现有证据不支持把它认定为OOM或特定动力学失稳。

冻结的L8训练/环境源码未被本轮改写。修复统一终止类别、保留原始原因并增强异常持久化需要另行审计和更新身份协议；本轮遵守中断即停止，没有在旧协议下修复后重新运行。`FORMAL_TRAINING_READY=NO`。

## 资源、数据完整性与图表

最后资源样本墙钟 `950.485628s`、CPU `945.539722s`，截至该记录平均约99.48%的一个CPU核心；它们是已记录下界，准确退出时刻未持久化。已记录进程峰值RSS `1793.050781MiB`（约1.75GiB），最低MemAvailable `8583.656250MiB`，磁盘仍约66.6GiB。没有观察到OOM，终止异常与内存守卫无关；准确退出前的RSS峰值没有单独捕获。每步环境/扫描/编码及PPO更新耗时只聚合在内存并计划写入最终summary，异常后这些聚合未保存，故标记 **NOT RECORDED**，不能用L8 Pilot时延替代本次长跑数据。

九张图使用706条真实episode、934条资源记录和唯一Validation/训练诊断样本，均注明 **INTERRUPTED**。训练成功/碰撞/超时曲线为最近100条已结束episode的比例；它们不是独立训练seed。PPO loss/KL只有一个里程碑采样点，不表示完整学习曲线。访问次数图仅展示100,096步checkpoint中的抽样账目。图目录为 `mujoco/reports/mujoco_lidar_ppo_formal_seed0_figures/`。

原7kg动力学模型、七个历史checkpoint和L8/L9运行源码哈希仍匹配训练前身份。原始文件保留完整；失败分析单独读取幸存证据，没有补造summary/complete，没有截断或修改训练数据。分析脚本的完成路径仍强制要求10个事件和1,000,064步，失败报告走独立路径且始终标为FAIL。

## 测试与独立复审

训练前714项相关测试通过，独立预审GO。运行后新增失败契约的零更新复现与中断报告负例：缺少完成标记不能报完成，已有completion标记不能进入中断报告路径。已完成的完整相关回归 **719 PASS、0 FAIL、0 SKIP、2个继承警告**（698历史+21新增）。之后补充第22项PNG成品标题检查，最终L9定向套件 **22 PASS**（28.69s）；联合覆盖720个不同测试，但**没有一次完整720项PASS运行**。附加全套复核会话22540在约10%进度被SIGTERM终止（exit143，原因未确认），该次复核为 **BLOCKED/INCOMPLETE**，没有计入PASS或归因于OOM。

独立只读代码/结果/成品复审 **GO，仅接受如实FAIL交付**；九张原始PNG中的INTERRUPTED标题由原始像素检查确认，先前缩放预览漏顶的判断已撤回。通过的测试不能消除本次已观察到的冻结接口缺陷。正式运行后没有修改训练运行时代码，后续改动限于失败分析、图表标签、测试与文档。

本次使用系统Python3.10解释器，并在 `PYTHONPATH` 显式指向历史虚拟环境中固定版本的Python包；没有安装或升级系统依赖。本轮实际训练命令如下（仅供审计；已有attempt会拒绝重跑）：

```bash
MINEUAV_LIDAR_UPSTREAM=/tmp/mineuav-mujoco-lidar-audit \
PYTHONPATH=/home/keyousuke/MineUAV-Learning-px4-sitl/mujoco:/home/keyousuke/MineUAV-Learning-px4-sitl/mujoco/rl:/tmp/mineuav-mujoco-lidar-audit/src:/home/keyousuke/MineUAV-Learning/.venv/lib/python3.10/site-packages \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python3 -u -m lidar_integration.ppo_formal_seed0_entry \
  --formal-seed0 --authorize-formal-training --seed 0 \
  --actual-timesteps 1000064 \
  --expected-protocol-sha256 fe07816f48bd4183165db482d8c93de50c637a32d13d86eccb47667d85e8cef6 \
  --upstream /tmp/mineuav-mujoco-lidar-audit
```

下一步唯一建议：先独立修复统一终止类别与异常证据持久化，完成RED→GREEN回归并重新冻结协议，再申请新的seed0正式训练；本轮到此停止。
