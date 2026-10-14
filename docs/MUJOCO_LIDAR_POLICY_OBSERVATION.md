# Phase 3E-L4 — LiDAR 固定尺寸观测与异步策略输入预检

日期：2026-10-14。基础提交 `d95552c86c87c4e749a238752082d5ffe00a3e0c`；分支 `feat/mujoco-lidar-policy-observation`。正式 66 帧分析**仅离线读取** Phase3E-L3 已生成的 MuJoCo 射线 NPZ；所有文件先与旧报告的 SHA256 逐一核对，没有为正式分析重新扫描。另有一项 Roll/Pitch 回归测试从真实 MuJoCo 扫描两帧、各 1000 射线；它不属于正式性能分析。没有启动 PX4/SITL、训练网络或运行 FAST-LIO2。完整实测及每帧来源身份见 [JSON 报告](../mujoco/reports/mujoco_lidar_policy_observation.json)。

## 编码契约

[`AngularRangeEncoder`](../mujoco/lidar_integration/observation.py) 只使用传感器本地 `points_xyz`、`ranges`、`valid_mask`、`ray_directions` 和必要的固定扫描配置/采样时间；世界位姿、场景 XML、障碍物真值、目标、未来帧均不参与编码。对单帧全部 24,000 条固定 `mid360` 近似模式射线，在本地 LiDAR 方位 `[0,360)°` 分 24 区、仰角 `[-10,55]°` 分 12 区。实测 66 帧射线仰角为 `−7.2075～52.1640°`；网格边界由该范围向外取整，**没有假设完整球面覆盖**。固定上游模式 SHA256 为 `9fa0165576f0060488254f04bd647a891004d4f4faa6e9b03b69d830c117c9a3`。

每个区统计采样射线数和有效命中数，取最近有效距离。`LidarObservation.network_tensor` 为有限 `float32[3,12,24]`，轴顺序为 `[通道, 仰角, 方位]`：

| 通道 | 含义 | 无有效回波 / 无射线覆盖 |
| --- | --- | --- |
| 0 | 最近有效距离 `(r−0.1 m)/4.9 m`，范围 `[0,1]` | 均为 `1`，**不是零距离障碍** |
| 1 | 该区是否有射线采样 | 有采样为 `1`，否则 `0` |
| 2 | 该区是否有有效命中 | 有命中为 `1`，否则 `0` |

两个 mask 使“有采样但无回波”与“根本未扫描”保持可区分。单个网络张量 `3456 bytes`，包括独立 mask/计数的完整观测数组 `7488 bytes`。66 帧累计有 438 个单帧未扫描格、4322 个已扫描但无有效回波格；288 格中有 4 格在这 66 帧里始终未采样。无效射线不会变成原点障碍。输出还保留原始 `frame_timestamp`、`MUJOCO_SIM_TIME`、近似传感器来源、编码版本、扫描配置和 NPZ SHA256 身份；不改历史 `PointCloudFrame`。此哈希身份仅在 `encode_npz` 或正式 runner 的文件校验路径具有密码学证明；任意调用 `encode_frame` 时可提供的身份标签本身不构成文件验证。

编码的几何回归使用**实际 MuJoCo 射线帧**：X 靠墙、Y 靠/离左右墙、箱体前方遮挡、Yaw 旋转、Z 升降、竖井高度变化；额外从真实 MuJoCo 在 Roll/Pitch 姿态直接扫描，验证本地编码随姿态变化而世界前墙仍在 `X=3m`。竖井近水平射线对 1m 竖直移动几乎不变，这是固定几何和角度区域的观察，不等于 Z 定位不可行或 FAST-LIO2 失效。非法方向、错位点/射线、NaN/Inf、残缺 NPZ、错误模式哈希均拒绝。改变世界位姿元数据而保持本地射线不变，编码不变，检查了世界真值不会暗中进入 LiDAR 张量。

## 10Hz→25Hz 时间契约

[`AsynchronousObservationBuffer`](../mujoco/lidar_integration/async_observation.py) 保存**已经编码**的观测，不会在每个 25Hz 策略 tick 重复编码同一帧。源采样时间、到达时间、策略读取时间三者分开，且在本轮均为 `MUJOCO_SIM_TIME`；只选择已到达、采样时间不晚于策略时间的最新帧。浮点积分误差仅在 `1ns` 容差内比较，不覆写原始时间戳。重复消息不刷新年龄；样本乱序、时间回退、跨时钟域和已经执行策略读取后的倒填到达时间均拒绝。4 帧容量硬上限；本轮最大缓存数组占 `29,952 bytes`。

`observation_age = policy_time − lidar_sample_time`。预定义 `150ms` **仅为离线实验阈值，非真实飞行安全认证值**：超过后返回 `STALE`，不给策略旧张量；尚无可用数据返回 `MISSING`。同一帧在多个策略 tick 被复用时，返回 `is_new_for_policy=False`。`reset(experiment_id)` 清空跨实验历史，读写均核验实验身份。用旧 66 帧的实际时间戳进行六组各 26 次策略读取：零到达延迟调度共 156 读、0 未来、0 过期、90 次有效旧帧复用；显式标记为**人工离线 +20ms 到达延迟**的对照有 6 次初始缺帧、0 未来读。额外测试单次/连续丢帧、延迟、重复、乱序、缓存溢出及过期，不能把模拟延迟当真实传输延迟。

[`policy_input_contract.py`](../mujoco/lidar_integration/policy_input_contract.py) 仅定义未来接口：LiDAR `[3,12,24]`、年龄 1D、独立 ENU 自身位置/速度/航向、独立目标位置/航向；没有将 LiDAR 或目标偷偷拼入旧 BC/PPO 的 7D 输入。自身状态显式标记 `SIM_GROUND_TRUTH_STATE`、`PX4_ESTIMATED_STATE` 或 `OFFLINE_TEST_STATE`；该字段是调用方提供的**来源声明**，不是 PX4 健康认证。旧 [`mine_uav_env.py::_get_obs`](../mujoco/rl/mine_uav_env.py) 是 `[target−position]` 世界坐标误差、世界速度和 wrapped yaw error；本轮没有构造或修改旧模型输入。

当前固定 PX4 源码 `msg/versioned/VehicleLocalPosition.msg` 声明局部 **NED** 位置/速度、`heading`、`xy_valid/z_valid/v_xy_valid/v_z_valid`、`heading_good_for_control` 和重置计数。未来可从该估计器话题经**已验证的 NED→ENU、航向、时间基准和原点/重置适配**构造对应自身状态；本轮并未实现该在线适配，也未采集新的 PX4 估计样本。目标来自独立任务/指令源，不来自 LiDAR。MuJoCo 真值和 PX4 估计不得互相冒名；真实 MID360S、FAST-LIO2 和飞行前状态健康仍未验证。

## 离线性能与证据

正式分析 66 帧顺序加载，无额外射线求交。最终代码重新实测：加载 NPZ 的 P50/P95/P99/最大值为 `5.848/7.141/8.340/10.228 ms`；编码为 `3.483/4.032/4.293/4.375 ms`；零延迟缓存读取为 `0.00137/0.00170/0.00231/0.00732 ms`。单独 SHA 校验 P95 `0.669 ms`。本轮单进程峰值 RSS `563.8 MiB`，未观察到 OOM。编码与缓存读取的实测最大值低于 25Hz 策略的 `40ms` 间隔，但**不**证明 raycast、模型推理、PX4、通信与编码合并后的硬实时性能。上阶段 24k 射线扫描耗时应单独查阅旧报告；本轮正式分析未重测。

图均来自实际编码/扫描时刻：[角度距离](../mujoco/reports/mujoco_lidar_policy_observation_figures/angular_range.png)、[命中 mask](../mujoco/reports/mujoco_lidar_policy_observation_figures/valid_hit_mask.png)、[未扫描 mask](../mujoco/reports/mujoco_lidar_policy_observation_figures/unscanned_mask.png)、[动态变化](../mujoco/reports/mujoco_lidar_policy_observation_figures/dynamic_encoding.png)、[10/25Hz 时间线](../mujoco/reports/mujoco_lidar_policy_observation_figures/lidar_policy_timeline.png)、[编码耗时](../mujoco/reports/mujoco_lidar_policy_observation_figures/encoding_cpu_time.png)。原始 NPZ 继续本地保留、不提交；图表和报告只存聚合/每帧来源、计数与耗时，不提交大数组。

## 复现与界限

使用历史真实 MuJoCo 3.14.0 Python 环境及本地 66 帧原始数据：

```bash
export PYTHONPATH=/home/keyousuke/MineUAV-Learning-px4-sitl/mujoco:/tmp/mineuav-mujoco-lidar-audit/src:/home/keyousuke/MineUAV-Learning/.venv/lib/python3.10/site-packages
export MINEUAV_LIDAR_UPSTREAM=/tmp/mineuav-mujoco-lidar-audit
python3 -m pytest -q mujoco/lidar_integration mujoco/px4_bridge mujoco/tests/test_mid360s_interface_audit.py
/home/keyousuke/MineUAV-Learning/.venv/bin/python -m lidar_integration.observation_runner \
  --raw-root mujoco/reports/mid360s_dynamic_scanning_parts/run_02 \
  --prior-report mujoco/reports/mujoco_mid360s_dynamic_scanning.json \
  --output-report mujoco/reports/mujoco_lidar_policy_observation_new_run.json \
  --figures-dir mujoco/reports/mujoco_lidar_policy_observation_figures_new_run
```

若本地 raw 缺失，历史数据测试会明确 `SKIP`，正式分析拒绝启动，绝不伪造旧帧。输出报告和图目录存在时拒绝覆盖。最终相关套件 **603 PASS、0 FAIL、0 SKIP**（575 项继承、28 项新增），1 个继承的 Matplotlib Axes3D 导入警告；独立只读复审 **GO，0 Critical、0 Important**。原始动力学模型和上游扫描模式 SHA256 保持不变。**PASS** 仅指固定编码、来源核验和离线因果缓存。**NOT TESTED**：真实 MID360S 校准/逐点时间/运动畸变、LiDAR–IMU 同步、PX4 在线估计适配、PPO 网络、策略闭环成功、FAST-LIO2 与联合实时性。当前数据足以设计 LiDAR+PPO 的输入形状和缺测语义，**不足以**批准训练或飞行；本轮未进入下一阶段。
