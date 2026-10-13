# MuJoCo–MID360S 接口设计（未实施）

日期：2026-10-13。目标是让现有 MuJoCo 物理模型以后能够生成**明确标识来源和精度**的 LiDAR 帧；本轮仅设计并离线审计，没有新增 LiDAR 扫描器、ROS 节点、PX4 运行或网络发布。依据及真机证据见 [MID360S_REAL_INTERFACE_AUDIT.md](MID360S_REAL_INTERFACE_AUDIT.md)。

## 先复用已存在的 MuJoCo 开源实现

优先评估 [`discoverse-dev/MuJoCo-LiDAR` 固定提交 `e6b879df`](https://github.com/discoverse-dev/MuJoCo-LiDAR/tree/e6b879df4b96daaed30ef10fe02650d29048fa9b)（MIT）而不是从零开发射线求交。其 [`README.md:18-47`](https://github.com/discoverse-dev/MuJoCo-LiDAR/blob/e6b879df4b96daaed30ef10fe02650d29048fa9b/README.md#L18-L47) 明确提供基于 MuJoCo `mj_multiRay` 的 CPU 后端、其它可选加速后端、`mid360` 扫描模式和 ROS 示例。固定版 [`src/mujoco_lidar/scan_gen.py:15-91`](https://github.com/discoverse-dev/MuJoCo-LiDAR/blob/e6b879df4b96daaed30ef10fe02650d29048fa9b/src/mujoco_lidar/scan_gen.py#L15-L91) 从 `scan_mode/mid360.npy` 顺序取射线角，每次 24,000 条；本轮只读检查该文件为有限值、`800000×2 float32`，SHA256 `9fa0165576f0060488254f04bd647a891004d4f4faa6e9b03b69d830c117c9a3`。这些事实只证明**有可用射线模式数据**；其角度序列来源、MID360S 真实等效性、强度、噪声、遮挡、运动畸变和逐点采样时刻均未确认。

该项目的 [`examples/lidar_vis_ros1_wrapper.py:46-78`](https://github.com/discoverse-dev/MuJoCo-LiDAR/blob/e6b879df4b96daaed30ef10fe02650d29048fa9b/examples/lidar_vis_ros1_wrapper.py#L46-L78) 发布普通 `sensor_msgs/PointCloud2`，字段主要是 XYZ 和强度，且示例强度填常数 `1.0`；它**没有**自动产出现场所需的 Livox CustomMsg 逐点时标、`tag`/`line` 或同步 IMU。因此结论是“可直接拿来作为 MuJoCo 射线/近似点云基础”，不是“可无缝替代真实 MID360S–FAST-LIO2 链路”。本阶段不安装或复制此依赖；后续只在隔离实验中固定版本、检查许可证和基准性能。旧 [`Killow1998/mid360_sim`](https://github.com/Killow1998/mid360_sim) 是 Gazebo 插件，不纳入本项目执行链；旧 `-frontier-` SDF 的 ray 近似亦不带入 MuJoCo。

## 组件边界

```text
MuJoCo MjModel/MjData + 明确的传感器位姿
   → LiDAR 射线后端（优先评估 MuJoCo-LiDAR，CPU 起步）
   → PointCloudFrame（独立于 ROS/PX4 的内部契约）
      ├─ NumPy / RL 点云特征适配器
      ├─ ROS1 PointCloud2 可视化/算法适配器
      └─ Livox CustomMsg + IMU 适配器〔必须过定位质量门，未实施〕
            → FAST-LIO2 → /Odometry → 既有 PX4 外部定位链〔未测试〕
```

后续应使用**独立场景副本或附加模型**定义 `lidar_site`；不修改历史 `mine_uav_dynamics_v2.xml` 或训练环境。站点相对机体的位姿必须包含来源标签：`UNMEASURED_SIM_PLACEHOLDER`、`SIM_CONFIG` 或 `MEASURED_REAL_CALIBRATION`，不可默认为真实标定。扫描后端仅使用 MuJoCo 几何求交，不串联 Gazebo。自机几何排除、最短/最长距离、网格反射、动态物体都需按可重复的场景测试验证。先 CPU、单进程测时延与内存，再考虑可选加速；不因开源 README 的通用速度宣称而预设本矿井场景性能。

## 建议内部 `PointCloudFrame` 契约

| 字段 | 要求 | 禁止的伪装 |
| --- | --- | --- |
| `source_identity` | `MUJOCO_RAYCAST_APPROX_MID360` 等不可误解的来源；记录扫描实现和模式文件 SHA | 写成真机 MID360S 数据 |
| `frame_timestamp`, `clock_domain` | 物理仿真采样时刻及其时钟域、严格单调 | 用 ROS 发送时间替代采样时刻 |
| `sensor_frame_id`, `T_body_lidar`, `extrinsic_provenance` | 坐标轴、刚体变换与来源显式记录 | 把视觉 mesh 或旧 Gazebo 外参当实测 |
| `points_xyz[N,3]`, `valid_mask[N]` | LiDAR 坐标系米制点，NaN/Inf 禁止，射线未命中有明确 mask | 让无命中点默认为真实零距离 |
| `per_point_time_offset[N]` | 仅有可信采样调度时提供；单位、相对基准、范围明示 | 把序列索引或均匀插值称为设备真值时标 |
| `intensity`, `tag`, `line` | 可选；标明测量、合成或不支持 | 将常数强度、伪造 tag/line 描述为 Livox 传感器数据 |
| `scan_configuration` | 扫描模式、采样数、FOV/范围、噪声开关、失真策略和哈希 | 用 `mid360` 名称证明 MID360S 精度 |

所有坐标变换以显式矩阵和单元测试实现。当前 MuJoCo 桥接约定 world ENU、body FLU，现有 [`frame_conversions.py`](../mujoco/px4_bridge/frame_conversions.py) 已有 ENU↔NED、FLU↔FRD 工具。约定 `T_body_lidar` **把 LiDAR 坐标转换至 body FLU**：`p_body = R_body_lidar @ p_lidar + t_body_lidar`；`T_imu_lidar` **把 LiDAR 坐标转换至 IMU 坐标**：`p_imu = R_imu_lidar @ p_lidar + t_imu_lidar`。两者取值尚未核实，不能把旧 SDF 数字填入。点云投射、IMU 和 Odometry 必须连同姿态/角速度/协方差方向一起测试。`camera_init` 是历史 FAST-LIO2 输出参考帧，不是 MuJoCo world/机体帧的同义词；与 PX4 NED 原点的对齐另行处理。

## 两条后续用途的不同准入条件

| 用途 | 可共享 | 必须额外验证 | 当前状态 |
| --- | --- | --- | --- |
| A：LiDAR 点云特征供 PPO | MuJoCo 射线、传感器位姿、帧契约、有效点 mask、数据身份 | 观测尺寸/吞吐、距离噪声和几何域差异；后续训练协议与安全接口需单独批准 | `NOT TESTED`，本轮不训练 |
| B：LiDAR+IMU → FAST-LIO2 → PX4 EKF2 | 同一射线/帧来源、位姿/时钟审计 | MID360S 扫描样式、逐点时标、运动畸变、同步 IMU、LiDAR↔IMU 实测外参、ROS1 CustomMsg 的**现场真实类型**、驱动字段/tag/line、质量/延迟/重置/协方差、动态移动验证 | `NOT TESTED`，不发送 Odometry |

FAST-LIO2 的参考 Livox 分支读取逐点 `offset_time` 用于时间/运动补偿，并读取 `tag`、`line` 做有效点过滤和扫描线处理；因此 ROS PointCloud2 能可视化 ≠ FAST-LIO2 兼容。另有上游驱动 `livox_ros_driver2/CustomMsg` 与上游 FAST-LIO2 `livox_ros_driver/CustomMsg` 的**包名差异**。必须先从真实 NUC 的 `rostopic type`、`rosnode info`、已安装源码和编译信息核对是否已有适配；不能凭历史 `/Odometry` 10 Hz 猜测编译链。未来只有在允许访问真实设备/已保存完整 rosbag 的新阶段才做这一步，本阶段不接触设备。

最初可在声明为 `SIMULATED GROUND TRUTH / APPROXIMATE LIDAR` 的离线场景使用 MuJoCo 真值用于时钟/坐标接口测试；不能冒充 FAST-LIO2。即使将来定位链运行，也应将真值与 FAST-LIO2 估计输出分通道保存。接到 PX4 时复用 Phase 3A–3D 的外部 Odometry 接口/融合证据门，要求真正的 EKF aid-source 融合状态，而非只发送消息。

## 后续实施次序与停止门（本轮均未执行）

1. 固定 MuJoCo-LiDAR 版本，在隔离场景以 CPU 生成简单几何点云；检验射线原点、自机遮挡、坐标旋转、有效点、单帧耗时。保留原动力学模型和历史基准不变。
2. 用许可的真实 MID360S 数据及产品资料审计 `mid360.npy` 扫描分布与帧率，定义“近似/不等效”的可量化界限；无证据时只用于用途 A 的早期原型。
3. 测量真实机体安装 XYZ/RPY 与 LiDAR–IMU 外参；固定时钟域与同步误差预算。缺失任一项即停止用途 B 的正确性宣称。
4. 独立实现并测试 ROS1 适配器；在无人机未解锁且不接 PX4 飞行控制的隔离流程核对 CustomMsg/IMU 的时间、帧、字段、FAST-LIO2 订阅与动态估计。只有后续单独授权才能进入 PX4 定位验证。

本轮验收：**接口设计 PASS**；MuJoCo-LiDAR 集成、MID360S 物理保真、FAST-LIO2 和 PX4 运行均 **NOT TESTED**。
