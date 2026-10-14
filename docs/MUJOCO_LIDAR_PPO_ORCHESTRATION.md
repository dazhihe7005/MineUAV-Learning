# Phase 3E-L8 — LiDAR PPO 多场景训练编排与单次有界 Pilot

日期：2026-10-14。基础提交 `c4e1166ac2e01f4747201442e07d4fe079261ce3`；分支 `feat/mujoco-lidar-ppo-training-orchestration`。本阶段仅运行一次独立 `PILOT_ONLY`、seed 0、单 worker、8192 环境步的真实 MuJoCo PPO 更新。正式百万步训练、Test episode、PX4、FAST-LIO2 和实机运行均为 **0**。逐项可核对数据见 [报告 JSON](../mujoco/reports/mujoco_lidar_ppo_orchestration.json)；原始 checkpoint、事件和完整本地证据保留在 ignored `mujoco/reports/mujoco_lidar_ppo_orchestration_parts/run_01/`，不随 Git 提交。

## 冻结输入与正式协议修订

Driver 在训练前重新核验 L6 预注册、12/6/6 Train/Validation/Test 场景清单、四份关键运行时源码、Reward 版本和七个历史 BC/PPO checkpoint 的 SHA256。只有 Train 的 12 个冻结身份能被 `MultiScenarioTrainingEnv` 抽取；Validation 有独立环境和确定性推理；Test 仅核对封存清单身份，不运行 episode。原始 7kg MJCF、24k-ray `mid360` CPU 模式、500Hz 物理/100Hz PI/25Hz 策略/10Hz LiDAR、`NAV_V2_COLLISION_PREFLIGHT_V1`、CNN、动作尺度和 PPO 核心超参数不变。Pilot 不从 L6 `SMOKE_ONLY` 或历史 PPO checkpoint 初始化。

L6 正式提案 hash 为 `f12cee995d003cbd502b3bd5477d142d658f1362dae368e5286ee1ddafc49bf4`；L7 候选协议 hash 为 `7c0c032002a5ea04b3400a5d939215cb10996a40fc5ff2f008c076ec51f55e33`。L8 明确修订了 **计步语义**，协议 hash 为 `fe07816f48bd4183165db482d8c93de50c637a32d13d86eccb47667d85e8cef6`。正式预算仍为 seed `[0,1,2]`、每 seed 名义 1,000,000 步、单 worker；`n_steps=128`、batch 64、2 epochs、lr `3e-4`、γ `.99`、GAE λ `.95`、clip `.2`、entropy 0、value coefficient `.5`、gradient clip `.5`。此正式协议**未执行**，且仍需用户单独授权。

SB3 2.9.0 的 `OnPolicyAlgorithm.learn()` 在完整 rollout 后调用 `train()`；`on_rollout_end` 则在该 optimizer 更新**之前**。因此 Driver 按完整128步 rollout 分段调用 `learn(delta, reset_num_timesteps=False)`，返回且确认预期 optimizer 更新后，才保存 checkpoint 并执行 Validation。`total_timesteps` 是收集目标下界，不是自动精确截断；每100,000名义步的事件顺延至首个完整 rollout 边界：100,000→100,096，200,000→200,064，…，1,000,000→1,000,064。每个事件同时保存名义和实际步数，不能把100,096记成100,000。模拟计数器测试覆盖99,968、100,000、100,096、200,000和1,000,000周边；这些**不是**实际执行的物理训练步。

Checkpoint 为 SB3 ZIP 加绑定执行协议 hash 的 JSON sidecar，事件仅在保存、重载校验及完整 Validation 后写入；已存在的事件或 Pilot attempt 会拒绝覆盖。ZIP 与 sidecar 各自原子写入，跨文件不是原子事务，残缺组合必须拒绝。Validation 必须保持策略、优化器、训练步数和 Train 场景 RNG 不变，且不接触 Test。检查点保存 optimizer 状态，但不含完整 MuJoCo、PI、LiDAR 或部分 rollout 状态；因此正式运行中断仍为 `RESTART_REQUIRED`，不能宣称精确恢复，warm start 只能属于新的实验 ID。正式终点选择规则仍为 **final checkpoint**，不是根据 Validation 或 Test 挑选最优。

## 唯一 Pilot 的实测结果

独立实验 ID `LIDAR_PPO_PILOT_ONLY_L8_SEED0`，新随机网络初始化，实际 8192 环境步、64 个128步 rollout、128 个 PPO epoch updates、256 个 Adam optimizer steps，Actor 和 Critic 参数 hash 都发生变化；策略参数与梯度有限。实际物理推进 163,440 tick，少于8192×20是因为部分碰撞在策略步内部提前终止，不是漏采样。47 次 Train reset 覆盖 **12/12** 冻结场景，逐场景访问计数按 identity SHA256 字典序为 `[10,7,4,4,4,2,2,2,2,5,2,3]`（完整场景→计数映射见 JSON），总和47，且与 seed0 调度器抽样序列一致。46 个已结束 Train episode：42 碰撞、4 超时、0 成功、0 物理失败、0 传感器失败；最后还有1个进行中的 episode。已结束 episode 平均回报 `−25.9688`。这说明训练编排确实运行，**不构成避障学习成功证据**。

完成优化更新后保存了一份 step8192 Pilot checkpoint，ZIP SHA256 `c0a11c85e038a312968bf806b040dd64d1007a3e8b8204e2a8d78b237de7aa20`；它仅供本地审计，不是正式模型。随后执行唯一一次完整独立 Validation：6个冻结 Validation 场景均有真实终止记录，0成功、6碰撞；策略、优化器、训练 timesteps 和 Train 抽样 RNG 不变。此结果只验证接口，并未用于改 Reward、调参或选择模型。Test episode 数为0。

## 性能、资源与身份

单 CPU worker。整体 Pilot wall `67.672s`，**包括** checkpoint 与约5.10秒 Validation，不能当作纯训练时间。环境 step P50/P95/P99/max `1.150/11.806/12.157/13.602ms`；3247次24k-ray 扫描 `6.213/6.646/6.940/7.844ms`，对应编码 `4.019/4.233/4.502/5.800ms`；47次场景 reset/编译 `331.126/393.848/452.264/456.422ms`；64次 PPO train 更新 `16.491/18.714/21.888/22.390ms`。峰值 RSS `1728.770MiB`，最低系统 MemAvailable `9245.742MiB`，未观察到 OOM，1.5GiB 保护阈值没有触发。短 Pilot 不能证明百万步的内存稳定性，故 `LONG_RUN_MEMORY_PROVEN=NO`。

MuJoCo `3.14.0`、SB3 `2.9.0`、PyTorch `2.14.0+cpu`、Gymnasium `1.3.0`、NumPy `2.2.6`。系统 OpenCV 与 NumPy ABI 混用仍产生导入警告，但本次受限进程完成；正式长跑前仍应隔离依赖来源，不修改系统 Python。训练运行时源码 hash 在前置身份、Pilot 原始 summary 与报告生成时一致；七个历史 checkpoint hash 未变。独立只读审查又逐项核对 complete→summary、ZIP→sidecar、场景 split、事件、权重变更和资源记录。

## 测试、结论与边界

先运行历史相关 **679 PASS**。本阶段新增19项测试，包括真实512步多场景 PPO rollout、128步真实保存/评价事件、后更新顺序、名义/实际里程碑、一次性 Pilot 保护、checkpoint 写入故障、Validation RNG 污染、源码身份、内存保护和报告 readiness 负例。最终完整相关套件 **698 PASS、0 FAIL、0 SKIP、2项继承警告**。独立只读复审 **GO，Critical 0、Important 0**；其提出的“测试失败仍可能写 READY” Important 项已通过 RED→GREEN 修复，未修改 Pilot 训练运行时代码。审查独立核对原始 evidence/checkpoint、Adam step、47次抽样及12/6场景划分；由于未另存逐 Train episode 轨迹，平均回报与逐次碰撞不能从第二份轨迹独立重算，只能依据已锁定源码及运行 summary。测试模拟的100k/1M计数不是真实训练。

在最终测试和独立审查均通过、报告身份验证仍完整的前提下，`TRAINING_DRIVER_READY=YES`、`FORMAL_SEED0_READY=YES` 的含义仅是**单 seed 正式训练的编排准入**；不是训练授权，也不是长期稳定性或导航性能认证。Pilot 的 42/46 碰撞和 Validation 的6/6碰撞如实保留；学成的避障成功率 `NOT TESTED`。真实 MID360S 时序/噪声、FAST-LIO2 与 PX4 控制同样 `NOT TESTED`。下一步仅建议在独立授权后运行固定协议的 seed0 正式训练，同时持续监控内存；本阶段到此停止。
