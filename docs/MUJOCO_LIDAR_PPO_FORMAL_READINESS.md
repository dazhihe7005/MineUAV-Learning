# Phase 3E-L7 — LiDAR PPO 多场景训练与评价准入

日期：2026-10-14。基础提交 `3658b9f230fd3b461976ae0f1abad60bf204bd72`；分支 `feat/mujoco-lidar-ppo-formal-readiness`。这是**不训练**的准入核验：本轮 PPO optimizer 更新数为 **0**，Test episode 数为 **0**，PX4、FAST-LIO2 与真实设备运行数均为 **0**。正式 1M-step×3 seed 训练没有启动。机器可核对的完整场景身份、时序、checkpoint 和资源数据见 [报告 JSON](../mujoco/reports/mujoco_lidar_ppo_formal_readiness.json)；原始探测与未训练 checkpoint 位于 ignored `mujoco/reports/mujoco_lidar_ppo_formal_readiness_parts/run_01/`。

## 冻结协议与场景调度

L6 预注册 JSON 文件 SHA256 `c4684901…`，场景清单 SHA256 `f6ee34a1…`，L6 正式训练提案 SHA256 `f12cee99…`；L7 在加载时逐项复核 12 个 L6 运行时源码 SHA256，包括原有 Reward、CNN、动力学、PI 与雷达接口。L7 **不改** Reward `NAV_V2_COLLISION_PREFLIGHT_V1`、`[-1,1]^4` 动作尺度、CNN、PPO 超参数、500/100/25/10 Hz 时序、24k 射线或历史模型。L7 候选协议只把 L6 的 `train_scene_scheduler=NOT IMPLEMENTED` 显式改为单实例、episode-reset 均匀 Train 抽样，并增加“中断的正式训练从原 seed 重启”语义；新协议 SHA256 `7c0c0320…`。正式预算仍是 seed `[0,1,2]`、每 seed 1,000,000 timesteps、`n_steps=128`、batch64、2 epochs、lr `3e-4`、γ `.99`、GAE λ `.95`、clip `.2`、entropy0、value coefficient `.5`、梯度裁剪 `.5`、100k 步 checkpoint/Validation、**final checkpoint** 选择。

`MultiScenarioTrainingEnv` 仅持有一个当前 MuJoCo 场景，按独立 NumPy RNG 从冻结的 12 个 Train 身份均匀抽取；换场景先关闭旧实例再编译新模型，同场景则复用并 reset。编译后的障碍几何 SHA 必须与清单一致。诊断模式可按注册 Train 身份逐项强制 reset，不能指定 Validation/Test。两轮逐项核验共 24 次 Train reset 和 24 个真实物理单步；构建未训练模型前另有 1 次 Train reset，因此实际调度访问总计 25 次。12/12 场景均通过：起点、目标、Yaw、qvel、act/ctrl、PI 积分、成功计数、碰撞记录和 LiDAR t=0/缓存清零正确；每步 20 个 500Hz 物理 tick。独立 seed29 的 1200 次**仅抽签、不编译场景**检查中，每场景计数 79–120，描述性 χ²(11)=19.5；这是抽样实现的有限检查，不是复杂环境分布的充分证明。

## 独立 Validation 与指标

`LidarPPOValidationEvaluator` 只接受冻结的 6 个 Validation 身份，顺序创建并关闭独立环境，以 `model.predict(..., deterministic=True)` 和 `torch.inference_mode()` 推理；模型权重、优化器状态、训练 timesteps 及 Train 调度器 RNG 不变。每个 episode 记录真实终止类别、累计 Reward、步数/完成时间、最终目标误差、轨迹长度、最小**有效 LiDAR 射线测距**、观测年龄、推理耗时和真实碰撞接触对。测距不是机体净空。成功、碰撞、超时、物理失败、传感器失败互斥。限制步数的接口探测标记 `probe_cutoff`，不计算成功率；本次正式准入证据则对全部 6 个 Validation episode 执行到既有终止条件。

所用策略是**随机初始化、未训练**模型，结果 0/6 成功、0/6 碰撞、6/6 在 15s 超时；均值回报 `−4.4635`，均值物理路径长度 `0.2120m`。这只是评价接口与终止记账的证据，不能描述训练后模型性能，也不能用于调整 Reward 或选择 checkpoint。Test 的 6 个封存场景仅核对预注册身份，未运行 episode。将来应按场景分别报告，再报告总体和至少 3 个独立训练 seed 的均值/离散程度；小型 6/6 集合不能证明真实矿井泛化。

## Checkpoint 与恢复边界

`CheckpointManager` 保存单独的 SB3 ZIP 与 JSON 身份清单，核对模型/优化器条目、timesteps、训练 seed、场景调度 RNG/访问计数、L6/L7 协议及场景清单 hash、Reward 版本、CNN 实现 hash 和七个历史 checkpoint hash。侧文件记录 PyTorch、NumPy 全局、Python random 的 **fingerprint**，不是可直接恢复的 RNG 快照。ZIP 和侧文件分别原子写入；若中断只留下一个，加载必须 fail-closed，二者并非跨文件事务。未训练契约模型保存/重载后 deterministic 动作一致；该文件不是正式训练初始化 checkpoint。

SB3 ZIP 包含 `policy.optimizer.pth`，但**没有**当前 MuJoCo `MjData`、PI 积分、LiDAR 缓存或未完成 PPO rollout。因此 `EXACT_RESUME` 与 `BOUNDARY_RESUME` 都不能声称已经实现；允许的技术类别仅为 `WARM_START`，若用于继续优化必须成为新的实验 ID。冻结的正式协议对中断运行采用 `RESTART_REQUIRED`：从原 seed 完整重启，不把 warm start 冒充无损续训。checkpoint 的保存/核验接口已具备，但自动每 100k 步保存及独立 Validation 触发尚未与未来正式训练 driver 接线。

## 资源、依赖和准入结论

本机实测版本为 MuJoCo `3.14.0`、SB3 `2.9.0`、PyTorch `2.14.0+cpu`、Gymnasium `1.3.0`、NumPy `2.2.6`。唯一受限探测 wall `23.13s`，24 次场景 reset/编译 P50/P95/P99/max `310.45/378.33/384.56/386.23ms`；24 次单 policy step `1.24/1.28/1.29/1.29ms`（该单步未必触发新扫描），每个 reset 的 24k-ray 初始扫描 P50/P99 `6.52/6.98ms`，编码 `3.94/4.29ms`。Validation 六场景完整评估 wall `14.08s`。进程峰值 RSS `1756.4MiB`，最低 MemAvailable `9556.1MiB`，未观察到 OOM；有限两轮换场景的当前 RSS 从首轮末约 `1130.8MiB` 到次轮末约 `1178.7MiB`，**不能外推为 1M-step 无泄漏**。低于 1.5GiB 可用内存会 fail-closed。未更改 swap，单 worker、CPU。

SB3 导入链仍输出系统 OpenCV/NumPy 1.x→2.x ABI 警告；测试与受限探测进程以退出码 0 完成，但正式长跑前应在独立环境中消除依赖混用，不修改系统 Python。历史 7kg MJCF、Reward、BC/PPO/World Model checkpoint 与原始 L6 轨迹均未修改。

**FORMAL_TRAINING_READY = NO。** 本轮多场景调度、完整六场景 Validation 接口、checkpoint 身份和安全重启语义已有实测证据；但 100k-step 周期保存/诊断尚未接入正式训练 driver，1M-step 的长时内存稳定性和训练中断流程未验证。下一次只建议单独授权周期编排与故障恢复的离线验收；本轮不开始正式训练。

## 验证与审查

相关历史测试基线 **667 PASS**；L7 新增 12 项定向测试，最终相关测试数量及独立只读复审结论以同提交的 JSON 报告为准。新增测试覆盖冻结文件/源码篡改拒绝、场景隔离与完整 reset、确定性/均匀抽签、Validation 不修改权重和优化器、互斥终止、checkpoint 保存/加载及篡改拒绝、恢复分级、限定资源与禁止重复探测。历史两条非算法 warning（Matplotlib 3D、Gymnasium render）如实保留。独立复审发现的优化器状态仅比对外层键和冻结源码未比对实际字节两个 Important 问题，均在探测前完成离线修复与测试；最终复审状态见报告。
