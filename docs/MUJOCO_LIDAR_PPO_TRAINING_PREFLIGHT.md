# Phase 3E-L6 — LiDAR-CNN-PPO 训练协议与有限冒烟

日期：2026-10-14。基础提交 `5205cf568dd2fab107436096d1c6b1a424260111`；分支 `feat/mujoco-lidar-ppo-training-preflight`。本轮只进行**一次**、单 worker、512 环境步的 PPO 程序冒烟，不是正式训练，也没有避障成功率验收。机器上完整数值、身份与状态见 [分析 JSON](../mujoco/reports/mujoco_lidar_ppo_training_preflight.json)；运行前冻结的协议、软件版本、源码/模型哈希与 12/6/6 场景清单见 [预注册 JSON](../mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json)。唯一冒烟 checkpoint 与原始 summary 位于本地 ignored `mujoco/reports/mujoco_lidar_ppo_training_preflight_parts/smoke_run_01/`，模型名带 `SMOKE_ONLY`，绝非正式模型。

## 观测、网络和动作

继承 `LidarNavigationEnv` 的真实 MuJoCo 500Hz 积分、100Hz 原 PI、25Hz 策略及 10Hz/24k 射线雷达。LiDAR 编码仍是传感器来源的 `float32[3,12,24]`：归一化距离、已扫描 mask、有效回波 mask。原 7D 状态保持不变：世界 ENU 的目标位置误差 3D、世界速度 3D、wrap 后航向误差 1D。单独的 `HeadingObservation` 从**当前** MuJoCo `qpos` 四元数读取 `heading=[sin(yaw_ENU),cos(yaw_ENU)]`，另有 1D LiDAR age；两者来源都明确为 `SIM_GROUND_TRUTH_STATE`，不是 PX4 估计。机体局部点云与世界坐标状态/世界速度动作之间因而有显式航向参考；±180° 连续表示经测试。旧 BC/PPO 的 7D checkpoint 未修改。原 7D 输入作为未来消融候选保留，不在本次训练比较。

自定义 `LidarStateExtractor` 使用 Conv2d `3→16, k3,s1,p1`，接 ReLU、Conv2d `16→32,k3,s2,p1`，接 ReLU、Flatten、FC `2304→64`，再与 7D state、2D heading、1D age 拼接为 74D。SB3 `MultiInputPolicy` 后接独立 actor/critic 两层 64D MLP，最终为 4D Gaussian 动作分布及标量 value head；全部 policy 参数共 **170,857**。`normalize_images=False`，**不会**把已归一化 float32 雷达再除以 255。单样本与 batch=4 前向、actor/value/log-prob/entropy、有限梯度及保存重载测试通过。PPO 内部 Gaussian 样本不等于物理速度；动作经 SB3/Gymnasium 边界及原 `MineUAVEnv.step` 的 `map_normalized_action` 才映射至世界 ENU `[1.5,1.5,1.0]m/s`、`1rad/s`。这不是 PX4 Offboard 接口。

## 场景和 Reward 审计

预注册 12 Train、6 Validation、6 封存 Test 场景，各有不交叉 scene seed、起点/目标/初始 yaw、障碍位置和尺寸及 SHA256 身份。场景只在内存编译，历史 MJCF SHA256 `cde862f4…` 不变；起点、目标都避开障碍和边界，并以独立 2D AABB 网格检查名义高度几何通路。可达路径只用于生成/审计，**不输入策略**。这些只是小型合成布局，目标分布的 X 范围受原环境 `±2m` 约束，不能证明矿井泛化。唯一冒烟仅用 Train 清单 `front_box` 第 2 项；Test 没有做 episode、模型选择或 Reward 调整。正式跨 12 Train 场景的 episode-reset 调度器**尚未实现**，所以正式训练协议只是提案，不能直接启动。

Reward 严格沿用 L5 已冻结的 `NAV_V2_COLLISION_PREFLIGHT_V1`：

`R = 10Δd·1[非失败] − 0.005||a||² − 0.5 clip((0.5−d)/0.4,0,1)||v||² + 10·success − 30·collision − 30·physical_failure − 10·sensor_failure − 0.01·executed_step − 2·timeout`。

先基于哈希核验 L5 原始真实物理轨迹：正前箱体障碍盲碰撞累计 `−25.3092`，通道无碰撞到达并停稳累计 `+25.6107`。理想静止、零动作、零速度的 375-step 悬停至超时为数学值 `−5.75`。这些历史证据**并非同场景**。因此在唯一 PPO 冒烟**完成之后**，另外预先固定 `front_box` Train 场景、相同目标/初态及三种审计动作，各只运行一次真实 MuJoCo episode：A 零速度命令 375 步超时、回报 `−5.7500`；B 原障碍盲 Scripted 43 步碰撞、回报 `−25.3092`；C 显式使用箱体几何预定 waypoint `(0.45,0.85,1)` 与 `(1.45,0.85,1)` 的 **privileged scripted** 绕行，344 步到达停稳、回报 `+22.8347`。三条轨迹的初始物理状态 SHA256 都是 `c7ea8770…`，场景 SHA256 都是 `54c2e1bc…`；三份原始 NPZ 哈希与逐步奖励求和已核对，保留 local-only。C 使用几何先验，不是 LiDAR 学习策略；此 `C>A>B` 仅表明固定场景下候选 Reward 对这三条真实轨迹的排序合理，**不能证明 Reward 全局对齐**。审计发生在训练之后，仅补充解释证据；没有依其结果改 Reward、协议或 checkpoint。

这里的“初始物理状态 SHA256”仅覆盖 `qpos/qvel/target`；PI、act、ctrl 由相同的确定性环境 reset 重置，但不包含在该哈希内，因此它不是完整控制器状态快照。旧 L5 的 `same_scene_three_policy_comparison=NOT TESTED` 仅指旧版跨场景轨迹；本轮另有冒烟后的同场景 A/B/C 审计。

## 冻结协议与唯一训练冒烟

实际软件：MuJoCo `3.14.0`、SB3 `2.9.0`、PyTorch `2.14.0+cpu`、Gymnasium `1.3.0`、NumPy `2.2.6`。固定上游 `MuJoCo-LiDAR` commit `e6b879df…` 与 `mid360.npy` SHA256 `9fa01655…`。运行前 JSON 同时冻结源码、Reward、网络、动力学、七个历史 BC/PPO 模型及场景清单 SHA。当前环境 import SB3 时会打印系统 `cv2`/NumPy ABI 警告，但相关测试及唯一训练命令退出码为 0；该依赖告警应在后续正式训练前隔离解决，未擅自升级系统包。

冒烟：seed 0；1 worker；`n_steps=128`、`batch_size=64`、`n_epochs=2`、Adam（SB3 PPO 默认）、learning rate `3e−4`、`gamma=.99`、GAE `λ=.95`、clip `.2`、entropy coefficient `0`、value coefficient `.5`、global gradient clip `.5`；实际总计 **512** policy/environment timesteps、**10,240** 真实 physics ticks，SB3 记录 `train/n_updates=8`（4 次 rollout×每次 2 epoch）。Actor 与 critic head SHA256 各自改变；末次记录梯度最大范数 `0.4594`，全部已记录权重、梯度和训练诊断有限；保存并重载后 deterministic 动作精确一致。唯一 Train episode 在 15s `time_limit`，这不代表学到避障。

测得训练 wall `2.913s`、CPU `2.901s`，约 `175.77` 环境步/s、`7.03` 仿真秒/墙钟秒。完整环境 step P50/P95/P99/max `1.096/11.510/11.833/13.190ms`；发生扫描的 204 个 step 中，24k-ray CPU 扫描 P50/P95/P99/max `6.039/6.420/6.659/7.754ms`，编码 `3.857/4.153/4.357/4.906ms`。扫描与编码已包含在环境 step 内。`wall−Σenv_step` 约 `0.326s`，包含 PPO 更新、重置、回调和框架开销，**不能直接称纯 PPO 计算时间**。峰值 RSS `1325.7MiB`；系统最低 MemAvailable `9982.6MiB`；一个进程，未观察到 OOM。另一个**未训练**的独立初始化进程中，`ru_maxrss`（截至该阶段的进程峰值）在导入后、环境构造后、模型构造后分别为 `339.2/1274.9/1274.9MiB`；不能将其当作本次训练进程的逐段增量。内存于启动前、环境/模型初始化后及每步/回调检查，低于 1.5GiB 会停止；未更改 swap。原始 checkpoint SHA256 为 `ca1c33b7…`；七个历史 checkpoint 训练前后哈希一致。

建议的**未执行**正式预算为每 seed 1,000,000 步、3 个独立 seed、相同网络/Reward/动作/物理及 PPO 超参数，按固定 final checkpoint 规则报告；每 100,000 步诊断/保存，Validation 与 Test 完全隔离。完整提案 hash 在预注册 JSON。进入正式训练前还需实现并验证 12 场景 reset 调度、训练中断/恢复与封存 Test 评估程序，并审查更多场景的 Reward 鲁棒性；不能把本次 512 步 checkpoint 作为基线继续训练或用于模型选择。

## 验证、独立审查与界限

全相关测试 **667 PASS = 627 继承 + 25 旧控制器定向 + 15 新增**，0 fail，2 条既有非算法 warning（Matplotlib 3D 包冲突、Gymnasium 未注册 render 检查）。新增测试包括当前航向、batch/single CNN、actor/critic 前后向与 checkpoint roundtrip、场景隔离/几何可达、冻结训练预算、Reward 原始哈希、跨 reset tick 计数、低内存熔断、训练诊断 NaN 拒绝、固定审计动作及证据重校验。独立只读**训练前**复审 GO，0 Critical/Important；复审明确限制为一次程序冒烟，不代表正式训练/避障/部署认证。最终报告生成器又核对预注册、模型、历史 raw、同场景 raw 及运行时源码 SHA；本次训练后未更改被哈希锁定的运行时代码。

未执行 PX4、SITL、FAST-LIO2、ROS、真实 MID360S/硬件，也未训练或覆盖任何旧 BC/PPO/World Model。机体碰撞包络是临时简化，LiDAR 为近似 MID360 模式、冻结整帧扫描且无真实逐点时间/运动畸变；heading/state 都取 MuJoCo 真值，未来须独立接入 PX4 估计状态与实传感器时间语义。**本轮验收 PASS 只指观测/网络/训练程序冒烟；避障学习与 Test 泛化 NOT TESTED，直接进行正式训练仍 BLOCKED。**
