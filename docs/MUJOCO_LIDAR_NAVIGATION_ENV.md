# Phase 3E-L5 — MuJoCo LiDAR 避障导航环境预检

日期：2026-10-14。基础提交 `371db05b8e8bcf6db3cbe7fd2c0c7c7a34c2b603`，分支 `feat/mujoco-lidar-navigation-env`。本轮建立并验证 **Gymnasium 环境接口**，不训练 PPO、不运行 PX4/FAST-LIO2、不接触实机。定量结果见 [JSON 报告](../mujoco/reports/mujoco_lidar_navigation_env.json)。报告中的 `PASS` 指环境预检，不表示已学会避障。

## 继承的物理与控制协议

[`LidarNavigationEnv`](../mujoco/lidar_integration/navigation_env.py) 继承原 [`MineUAVEnv`](../mujoco/rl/mine_uav_env.py) 的 `mj_step`、电机控制分配、速度→姿态→推力链与终止规则，不采用运动学轨迹回放。隔离场景只在内存中复制 MJCF 并添加障碍，不改 `mine_uav_dynamics_v2.xml`（SHA256 `cde862f424a86bda3676249859268c69dede9dde2ebadeef819813351684b520`）。模型质量 7 kg；惯量、重心、电机参数、四电机传动和 0.002 s 物理步长沿用原模型。外层速度 PI 使用原 [`VelocityCommandController`](../mujoco/control/velocity_command_controller.py) 及既有 [`configure_pi_env`](../mujoco/rl/audit_velocity_pi.py) 的固定 `ki_xy=0.5 s⁻²`、`ki_z=0.8 s⁻²`，其余 P/姿态/分配参数不变。未导入训练器，也未实现第二套控制律。

每策略 `step` 最多 20 次真实 500 Hz `mj_step`，每 5 个物理 tick 更新一次 100 Hz PI，策略 25 Hz。若接触或物理失败在策略周期内发生，则立即停止，故最后一步可以少于 20 tick。动作仍是 `float32[4]`、`[-1,1]^4`，依次为世界 ENU `[vx,vy,vz]` 和偏航角速度；实际尺度 `[1.5,1.5,1.0] m/s` 与 `1 rad/s`，与旧 7D 环境相同。重置时可用独立种子选择起点/目标/初始 Yaw；设置初始 qpos **仅限 reset**，运行期间姿态、速度和位移全由 MuJoCo 积分产生。目标额外检查不与障碍、地面或边界重叠。

## 场景与接触

[`navigation_scene.py`](../mujoco/lidar_integration/navigation_scene.py) 提供五个小型固定布局：`empty`、正前方箱体 `front_box`、左右箱体 `side_pair`、需绕行的 `detour`、有限宽通道 `channel`。都有真实地面和四面接触墙、确定性场景/起点/目标种子及几何哈希。Train/Validation/Test 各有 5 组不重复的场景/起点/目标身份，正式小规模运行**只用 Train 清单前三种代表场景**；未用测试组调奖励。布局并非矿井地图，随机化目前只是小幅定种扰动；可达性仅由几何开口和固定端点预检，不是任意随机目标的全局路径证明。

碰撞判定使用 MuJoCo `data.contact`，需涉及机体 `coarse_collision` 与障碍箱/边界墙/地面；接触立即失败，不用 LiDAR 阈值替代物理碰撞。近箱但保持间隙不判碰撞；边界墙实际接触也判失败。同一物理 tick 若同时过度倾斜并发生有效接触，有限状态下的真实接触优先归为 collision；非有限状态仍先按数值失败处理。原机体碰撞箱半尺寸约 `(0.12,0.12,0.05) m`，**TEMPORARY_PLACEHOLDER**：未覆盖桨叶、机臂或真实机体全包络，不能据此宣称真实避障安全。障碍是 `contype/conaffinity` 有效的实体，同时由真实射线求交可见；固定测试起点的全 24k 射线已检查四个含障碍布局的每个箱体均有合法命中，但这不保证所有后续视角。自机回波沿用 Phase3E-L2 扫描器过滤逻辑。

## 观测及因果时序

Gymnasium `Dict`：

| 键 | 形状 / 类型 | 来源 |
| --- | --- | --- |
| `lidar` | `float32[3,12,24]` | 24,000 条 MuJoCo CPU 射线，经原 [`AngularRangeEncoder`](../mujoco/lidar_integration/observation.py)；距离、采样 mask、有效回波 mask |
| `state` | `float32[7]` | `[target_position−position, world_velocity, wrapped_yaw_error]`，**SIM_GROUND_TRUTH_STATE** |
| `lidar_age` | `float32[1]` 秒 | 策略读取时刻减实际 LiDAR 采样时刻 |

雷达模式为固定上游 `MuJoCo-LiDAR` commit `e6b879df4b96daaed30ef10fe02650d29048fa9b`、`mid360.npy` SHA256 `9fa0165576f0060488254f04bd647a891004d4f4faa6e9b03b69d830c117c9a3`、CPU 后端；本地安装外参 `T_body_lidar` 平移 `(0.10,0,0.10)m`、零旋转，来源 **SIM_CONFIG**，不是实测 MID360S 标定。无回波距离通道为 `1`，另以两个 mask 分辨“有射线但无回波”与“未扫描”；缺测不是零距离障碍。编码器仅使用射线、距离、有效性、方向及扫描元数据，不从场景几何/障碍真值或未来轨迹提取策略特征。目标和自身状态是明确独立的非视觉输入；这也不是 PX4 EKF2 状态认证。

初始化在仿真 `t=0` 扫描，之后每 50 个物理 tick 扫描一次，恰为 10 Hz；策略每 20 tick、25 Hz 读取。复用原 [`AsynchronousObservationBuffer`](../mujoco/lidar_integration/async_observation.py)，原始 `frame_timestamp` 和同域 `arrival_time` 不被改写，`policy_time` 不能读取未来帧。缓存上限 4 帧；本阶段固定 150 ms 实验性新鲜度阈值。若进入下一策略动作前为 `STALE/MISSING`，立即传感器失败且物理推进 0 tick；若当前物理步发生碰撞、物理失败、成功或超时，则保持已有终止分类，不用同时发生的缺测覆盖它。失败观测的距离通道全 `1`、两个 mask 全 `0`，并带显式 `lidar_status`；**不授权继续盲飞**。该阈值不是实机安全阈值。重置同时清 PI 积分、MuJoCo 状态、扫描器和异步缓存。

## 冻结的候选 Reward 与终止

先审计旧 Reward V2：进度 `10(dₜ₋₁−dₜ)`、动作 `−0.005||aₜ||²`、近目标刹车 `−0.5w(dₜ)||vₜ||²`、成功 `+10`、物理失败 `−10`，其中 `w(d)=clip((0.5−d)/0.4,0,1)`。新环境仅冻结一份**候选** `NAV_V2_COLLISION_PREFLIGHT_V1`；没有看训练结果调节：

`R = 10Δd·1[非物理/传感器失败] − 0.005||a||² − 0.5w(d)||v||² + 10·success − 30·collision − 30·physical_failure − 10·sensor_failure − 0.01·executed_policy_step − 2·timeout`。

所有量是每策略步的无量纲奖励；动作和刹车项仅在已执行策略步时计入。碰撞、物理或传感器失败步的**当前步进度**归零，不能撞墙拿额外进度；历史已经获得的进度不撤销。成功、碰撞、物理失败、传感器失败、超时互斥；`reward_breakdown` 分别给出 `progress/action/brake/success/collision/physical_failure/sensor_failure/timeout_or_time_cost/total`。初版沿用旧 V2 的 `−10` 作为碰撞惩罚时，正前箱体撞击轨迹累计 `−5.31` 反而高于静止等待 15s 超时的理论 `−5.75`，这是明确的坏激励；该布局设计/Train 预检揭示后、没有任何 PPO 训练或 Test 组反馈，冻结碰撞与主动物理失败为 `−30`。固定清单初始距离小于 2m，即使假设不可能的“先拿满 20 分进度、无其他代价后才撞击”，累计也至多 `20−30−0.01=−10.01 < −5.75`。这个界仅适用于当前固定小型端点范围，不能证明所有未来随机任务或学习行为的 Reward 最优性。时间代价 `−0.01` 及超时 `−2` 也是本阶段新增的显式工程候选。

成功沿用距离 `<0.10m`、真实平移速度 `<0.15m/s`、连续 **5 个策略步**，且期间无碰撞/越界。碰撞和物理/传感器失败为 `terminated=True`；达到最多 15s 为 `truncated=True` 的 timeout，不能与成功混称。旧物理边界和倾角检查继续生效；边界实体墙先接触时分类为 `collision`。

## 受限真实物理运行与性能

只顺序运行 3 条单 worker episode。基线是原 [`scripted_action`](../mujoco/rl/test_env_scripted_policy.py) 仅使用 7D 状态的**障碍盲**启发式；它不是 LiDAR 策略，也不代表 BC/PPO 新输入能力。

| 布局 | 结果 | 策略步 / 物理 tick / LiDAR 帧 | 仿真时间 | 最终距离 | 累计奖励 |
| --- | --- | --- | --- | --- | --- |
| 无障碍 | 到达且停稳 | 91 / 1820 / 37 | 3.640s | 0.09067m | 25.1514 |
| 正前箱体 | 真实 `coarse_collision`–`nav_obstacle_0` 接触失败 | 43 / 848 / 17 | 1.696s | 1.28308m | −25.3092 |
| 通道 | 无接触通过并停稳 | 91 / 1820 / 37 | 3.640s | 0.09167m | 25.6107 |

合计 225 策略步、4488 物理 tick、898 次 PI 更新、91 帧，每帧 24k 射线。**完整环境 step**（物理+PI+到期扫描+编码）耗时 P50/P95/P99/Max `1.090/12.275/12.484/12.531ms`；扫描单帧 `6.628/7.021/7.069/7.168ms`；编码 `4.167/4.374/4.417/4.432ms`。扫描与编码是全 step 的组成部分，不能再加到 step 耗时上。该机器上预检的 p99 低于 25Hz 的 40ms，但这不是复杂场景、学习网络或 PX4 联合实时证明。RSS 启动/结束/峰值 `56.8/1105.5/1172.5 MiB`，系统最低 MemAvailable `8769.4 MiB`，未观察到 OOM；内存差包括 Python 图形库和 15 个场景编译，未单独归因于泄漏。原始轨迹 NPZ（含每步奖励分项/耗时、LiDAR 采样时刻）与绘图首帧真实点云，以及三次开发验证输出，保存在本地 ignored `mujoco/reports/mujoco_lidar_navigation_env_parts/`；四份正式 raw 的 SHA256 见 JSON，不提交大数组。报告的 contact/整体验收取决于实际接触事件：只跑空场景为 `NOT TESTED`，预期正前箱体不接触为 `FAIL`，正式三场景真实接触才为 `PASS`。

图像均基于本轮真实 MuJoCo 扫描/物理轨迹：[分场景俯视](../mujoco/reports/mujoco_lidar_navigation_env_figures/obstacle_topdown.png)、[实际点云与实体](../mujoco/reports/mujoco_lidar_navigation_env_figures/real_lidar_pointcloud.png)、[物理轨迹](../mujoco/reports/mujoco_lidar_navigation_env_figures/physical_trajectory.png)、[目标距离](../mujoco/reports/mujoco_lidar_navigation_env_figures/distance_vs_time.png)、[奖励分项](../mujoco/reports/mujoco_lidar_navigation_env_figures/reward_breakdown.png)、[10/25Hz 时序](../mujoco/reports/mujoco_lidar_navigation_env_figures/lidar_policy_timeline.png)、[完整 step 耗时](../mujoco/reports/mujoco_lidar_navigation_env_figures/environment_step_time.png)。

## 后续 PPO 接口与边界

Gymnasium Dict 与 SB3 `MultiInputPolicy` 的多键输入形状兼容；未来应使用小型 CNN 提取 `lidar[3,12,24]`，再与 7D 状态、1D age 合并为 actor/critic 特征。由于该 LiDAR 为**已归一化 float32** 而非 uint8 图像，默认图像提取器/自动缩放不能未经审计直接套用；本轮没有构建或训练自定义 CNN、没有启动 PPO optimizer，也没有声称模型能避障。旧 BC 与 PPO checkpoint 完全不变、未用于这三条 episode。未来要将 7D 状态来源替换成经过时间戳、坐标和有效性检查的 PX4 EKF2 估计，并独立验证真实 MID360S 逐点时间、运动畸变、安装标定、IMU 同步、复杂矿井几何、机体碰撞包络、故障处理及联合实时性。不能把此环境或 `SIM_GROUND_TRUTH_STATE` 当作可部署的 PX4/FAST-LIO2 闭环。

## 验证与复现

真实 MuJoCo Python 3.14.0；固定上游 CPU 扫描器；历史相关套件 **603 PASS**（修改前基线），最终相关套件 **627 PASS = 603 继承 + 24 新增**，另旧 `MineUAVEnv`/速度控制器/RewardV2/PPO-PI 定向回归 **25 PASS**。Gymnasium `check_env` PASS。2 条非算法警告：继承的 Matplotlib `Axes3D` 包冲突（仅用 2D 图），以及 `check_env` 因直接实例化而不能测试其他 render mode。原始动力学及上游扫描模式哈希未变；未运行训练、PX4、SITL、ROS、FAST-LIO2 或实机。独立只读最终复审 **GO，0 Critical / 0 Important**；审查者单独核验四份 raw SHA、225/4488/898/91 计数、逐步奖励、因果时序及性能分位数，但未独立重跑 627+25 项完整测试。

```bash
export MINEUAV_LIDAR_UPSTREAM=/tmp/mineuav-mujoco-lidar-audit
export PYTHONPATH=/home/keyousuke/MineUAV-Learning-px4-sitl/mujoco:/home/keyousuke/MineUAV-Learning-px4-sitl/mujoco/rl:/tmp/mineuav-mujoco-lidar-audit/src:/home/keyousuke/MineUAV-Learning/.venv/lib/python3.10/site-packages
python3 -m pytest -q mujoco/lidar_integration mujoco/px4_bridge mujoco/tests/test_mid360s_interface_audit.py
python3 -m pytest -q mujoco/rl/test_mine_uav_env.py mujoco/rl/test_velocity_command_controller.py mujoco/rl/test_reward_v2.py mujoco/rl/test_ppo_pi_env.py
```

`navigation_runner` 拒绝覆盖已有 JSON/图或非空 raw 目录；若需重跑，必须指定独立新路径/保存已有证据，且当前 runner 的 raw-part 路径固定为 `preflight_run_01`，应先为新运行独立配置该路径，不能直接覆盖历史。当前验收 **PASS**：物理链、真实点云、因果 Dict 输入、碰撞与奖励分类；**NOT TESTED**：LiDAR 网络学习、PPO 任务成功率、PX4 状态估计/飞行、FAST-LIO2、真实 MID360S 与碰撞安全。下一步仅建议单独预注册 LiDAR+PPO 训练/评估协议；本轮不执行。
