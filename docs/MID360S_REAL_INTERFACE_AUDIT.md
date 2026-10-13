# MID360S 真机接口审计（Phase 3E-L1）

日期：2026-10-13。此文是固定源码与历史记录的**离线审计**，不是设备当前在线状态或仿真验收。未启动 PX4、SITL、ROS、真机或 MuJoCo 扫描，未向设备发送数据。

证据级别：`VERIFIED_FROM_SOURCE` 表示在下列固定提交中直接读到的代码/配置；`HISTORICALLY_OBSERVED` 表示项目 A 保存的现场记录；`CONFIGURED_ONLY` 表示配置存在但没有本轮实际运行证据；`UNVERIFIED` 表示缺少可核查的已安装版本或实测数据。

| 代号 | 仓库与固定版本 | 本轮用途 |
| --- | --- | --- |
| A | [`dazhihe7005/-frontier-` @ `db897c4e`](https://github.com/dazhihe7005/-frontier-/tree/db897c4e3a1607f5852b484afd0a01523cb08c6b) | 真机工程配置、补丁和历史台架记录 |
| B | [`dazhihe7005/MineUAV-Learning` @ `ebbe6ed2`](https://github.com/dazhihe7005/MineUAV-Learning/tree/ebbe6ed2c69ef00d0bb37466f5d8278c9f2dd652) | 当前 MuJoCo 模型与桥接约定 |
| D | [`Livox-SDK/livox_ros_driver2` @ `21445540`](https://github.com/Livox-SDK/livox_ros_driver2/tree/21445540f0d100dc86a7e6df312dd70bbdb4afdf) | 可读取的上游驱动**参考版本**，并非已证实的 NUC 已安装版本 |
| F | [`hku-mars/FAST_LIO` @ `7cc4175d`](https://github.com/hku-mars/FAST_LIO/tree/7cc4175de6f8ba2edf34bab02a42195b141027e9) | 可读取的上游算法**参考版本**，并非已证实的现场编译版本 |
| M | [`discoverse-dev/MuJoCo-LiDAR` @ `e6b879df`](https://github.com/discoverse-dev/MuJoCo-LiDAR/tree/e6b879df4b96daaed30ef10fe02650d29048fa9b) | 可复用的 MuJoCo 扫描候选，未集成、未测真实性 |

## 真机通信与现场证据

A 的 [`README.md:116-125`](https://github.com/dazhihe7005/-frontier-/blob/db897c4e3a1607f5852b484afd0a01523cb08c6b/README.md#L116-L125) 记载 NUC 专用网口 `enp89s0=192.168.1.10/24`、雷达 `192.168.1.157/24`，该连接不设默认网关；属于 **HISTORICALLY_OBSERVED**，不能据此声称现在设备在线。A 的 [`README.md:282-290`](https://github.com/dazhihe7005/-frontier-/blob/db897c4e3a1607f5852b484afd0a01523cb08c6b/README.md#L282-L290) 记载发现应答 device type **35 = MID360S**，而 type 9 = MID360，要求 `msg_MID360s.launch`、`MID360s_config.json`；`SDK2 1.4.3` 也是该文的现场版本记载，本轮未获得安装二进制或 SDK 源码身份，**不可独立验证**。D 的 `lidar_summary_info.lidar_type=8` 是 JSON 字段，不能与发现应答的 type 35 混作同一枚举。

A 的 [`launch/task1_real.launch:15-23`](https://github.com/dazhihe7005/-frontier-/blob/db897c4e3a1607f5852b484afd0a01523cb08c6b/launch/task1_real.launch#L15-L23) 与 [`launch/px4_fastlio_localization.launch:10-14`](https://github.com/dazhihe7005/-frontier-/blob/db897c4e3a1607f5852b484afd0a01523cb08c6b/launch/px4_fastlio_localization.launch#L10-L14) 都包含驱动的 `msg_MID360s.launch` 与 FAST-LIO2 的 `mapping_mid360.launch`。这两个被包含的具体安装文件不在 A 中，因此其现场最终参数仍是 **UNVERIFIED**。A 的 [`docs/real_mid360_px4_bench_integration_2026-09-18.md:1-12`](https://github.com/dazhihe7005/-frontier-/blob/db897c4e3a1607f5852b484afd0a01523cb08c6b/docs/real_mid360_px4_bench_integration_2026-09-18.md#L1-L12) 是拆桨台架历史记录：`/livox/lidar` 约 10 Hz、`/livox/imu` 约 200 Hz、FAST-LIO2 `/Odometry` 约 10 Hz；[`docs/task1_live_mid360_decider_2026-09-18.md:7-8`](https://github.com/dazhihe7005/-frontier-/blob/db897c4e3a1607f5852b484afd0a01523cb08c6b/docs/task1_live_mid360_decider_2026-09-18.md#L7-L8) 还记录 `/cloud_registered` 约 10 Hz、帧 `camera_init`。这是历史观测，非本轮实测频率。

### UDP 配置不是 ROS Topic

A 的 [`patches/livox_ros_driver2_mid360s_reconnect.patch:1-16`](https://github.com/dazhihe7005/-frontier-/blob/db897c4e3a1607f5852b484afd0a01523cb08c6b/patches/livox_ros_driver2_mid360s_reconnect.patch#L1-L16) 明确将设备 IP 改为 `192.168.1.157`，NUC 接收 IP 改为 `192.168.1.10`，补丁上下文出现 NUC 侧 `cmd_data_port=56101`、`push_msg_port=56201`、`point_data_port=56301`。D 的 [`config/MID360s_config.json:1-32`](https://github.com/Livox-SDK/livox_ros_driver2/blob/21445540f0d100dc86a7e6df312dd70bbdb4afdf/config/MID360s_config.json#L1-L32) 还列有 NUC 侧 IMU 56401、日志 56501，以及设备侧 56100/56200/56300/56400/56500；这些**未在 A 补丁中完整定版**，只能视为上游参考配置，待核对现场安装文件。UDP 控制/推送/点云/IMU 是 Livox 驱动收发端口；`/livox/lidar` 和 `/livox/imu` 是驱动解包后的 ROS Topic，并非 UDP 端口。A 未包含与安装 SDK2 完全匹配的 UDP 收包/解析源码，**底层包字节字段 UNVERIFIED**，不从 ROS 字段倒推 wire format。

### ROS 消息与字段

D 的 [`launch_ROS1/msg_MID360s.launch:6-27`](https://github.com/Livox-SDK/livox_ros_driver2/blob/21445540f0d100dc86a7e6df312dd70bbdb4afdf/launch_ROS1/msg_MID360s.launch#L6-L27) 参考默认 `xfer_format=1`（自定义消息）、`multi_topic=0`、`publish_freq=10`、`frame_id=livox_frame`。D 的 [`src/lddc.h:44-57`](https://github.com/Livox-SDK/livox_ros_driver2/blob/21445540f0d100dc86a7e6df312dd70bbdb4afdf/src/lddc.h#L44-L57)、[`src/lddc.cpp:573-635`](https://github.com/Livox-SDK/livox_ros_driver2/blob/21445540f0d100dc86a7e6df312dd70bbdb4afdf/src/lddc.cpp#L573-L635) 将 `/livox/lidar` 广告为 ROS1 `livox_ros_driver2/CustomMsg`，`/livox/imu` 广告为 `sensor_msgs/Imu`。这是**上游参考实现条件下**的消息类型；A 没有保存 `rostopic type` 输出和已安装驱动构建哈希，现场最终类型尚不能独立确认。

D 的 [`msg/CustomMsg.msg`](https://github.com/Livox-SDK/livox_ros_driver2/blob/21445540f0d100dc86a7e6df312dd70bbdb4afdf/msg/CustomMsg.msg) 含 header、`timebase`、`point_num`、`lidar_id`、点数组；[`msg/CustomPoint.msg`](https://github.com/Livox-SDK/livox_ros_driver2/blob/21445540f0d100dc86a7e6df312dd70bbdb4afdf/msg/CustomPoint.msg) 每点含 `offset_time`、XYZ（米）、`reflectivity`、`tag`、`line`。没有单独的逐点序号或 scan ID；数组顺序和消息标头可以作为结构索引，但不应虚构设备序列号字段。D 的 [`src/lddc.cpp:354-399`](https://github.com/Livox-SDK/livox_ros_driver2/blob/21445540f0d100dc86a7e6df312dd70bbdb4afdf/src/lddc.cpp#L354-L399) 用包基准时间填 `timebase` 与 header stamp，并计算逐点偏移。D 的 [`src/lddc.cpp:480-496`](https://github.com/Livox-SDK/livox_ros_driver2/blob/21445540f0d100dc86a7e6df312dd70bbdb4afdf/src/lddc.cpp#L480-L496) 将 IMU gyro 与 acc 复制到 ROS `angular_velocity`、`linear_acceleration`，设 frame `livox_frame`。ROS 语义要求角速度 rad/s、线加速度 m/s²；但本轮未取得匹配 SDK 的设备单位转换证据，现场数值单位一致性须核验。

## FAST-LIO2 实际链路与未确认接口

A 的 [`launch/task1_real.launch:18-24`](https://github.com/dazhihe7005/-frontier-/blob/db897c4e3a1607f5852b484afd0a01523cb08c6b/launch/task1_real.launch#L18-L24) 与 [`README.md:164-166`](https://github.com/dazhihe7005/-frontier-/blob/db897c4e3a1607f5852b484afd0a01523cb08c6b/README.md#L164-L166) 记录链路：MID360S → `/livox/lidar` 和 `/livox/imu` → FAST-LIO2 `/Odometry`、`/cloud_registered` → 视觉位姿桥 → PX4。`/Odometry` 是 FAST-LIO2 估计输出，不是雷达原始输出。F 的 [`config/mid360.yaml:1-23`](https://github.com/hku-mars/FAST_LIO/blob/7cc4175de6f8ba2edf34bab02a42195b141027e9/config/mid360.yaml#L1-L23) 参考配置使用这两个输入、`lidar_type=1`、LiDAR–IMU 外参；F 的 [`src/laserMapping.cpp:845-859`](https://github.com/hku-mars/FAST_LIO/blob/7cc4175de6f8ba2edf34bab02a42195b141027e9/src/laserMapping.cpp#L845-L859) 发布 `nav_msgs/Odometry` 与 `sensor_msgs/PointCloud2`。F 的 [`src/preprocess.cpp:92-123`](https://github.com/hku-mars/FAST_LIO/blob/7cc4175de6f8ba2edf34bab02a42195b141027e9/src/preprocess.cpp#L92-L123) 使用 `tag`、`line`、每点 `offset_time`；[`src/laserMapping.cpp:382-399`](https://github.com/hku-mars/FAST_LIO/blob/7cc4175de6f8ba2edf34bab02a42195b141027e9/src/laserMapping.cpp#L382-L399) 和 [`src/IMU_Processing.hpp:310-325`](https://github.com/hku-mars/FAST_LIO/blob/7cc4175de6f8ba2edf34bab02a42195b141027e9/src/IMU_Processing.hpp#L310-L325) 将逐点时间用于扫描结束估计与运动补偿。只有 XYZ 点的 PointCloud2 不能证明等效。F 的 `MARSIM` PointCloud2 分支属于另一种预处理模式，也不能冒充真实 Livox 路径。

**重要兼容性缺口：**F 的 [`src/laserMapping.cpp:59,302`](https://github.com/hku-mars/FAST_LIO/blob/7cc4175de6f8ba2edf34bab02a42195b141027e9/src/laserMapping.cpp#L59-L59) 参考订阅类型为 `livox_ros_driver/CustomMsg`；D 的 ROS1 发布参考类型是 `livox_ros_driver2/CustomMsg`。A 的 [`patches/fast_lio2_sensor_restart.patch:92`](https://github.com/dazhihe7005/-frontier-/blob/db897c4e3a1607f5852b484afd0a01523cb08c6b/patches/fast_lio2_sensor_restart.patch#L92-L92) 仍出现前者。A 的历史数据链实际有 `/Odometry` 更新，但这**不足以复原现场实际消息定义/适配代码**。在取得已安装 FAST-LIO2 源码哈希、`rostopic type /livox/lidar`、订阅连接信息前，直接自制 `livox_ros_driver2/CustomMsg` 供现场 FAST-LIO2 使用的兼容性 **UNVERIFIED**。A 的时间回跳/断流重启补丁亦不构成时间同步证明。

## 安装外参：三种来源严格分开

1. A 的旧 [`models/iris_mid360/iris_mid360.sdf:12-15`](https://github.com/dazhihe7005/-frontier-/blob/db897c4e3a1607f5852b484afd0a01523cb08c6b/models/iris_mid360/iris_mid360.sdf#L12-L15) 配置相对旧 `iris::base_link` 位姿 `(0.1315, 0, 0.223)m`、俯仰 `0.436332313rad`。这是 **CONFIGURED_ONLY 的旧 Gazebo 仿真安装值**，不用于当前 MuJoCo，也不是实机标定。旧模型本身称其 ray 为近似而非 Livox 包/时序模拟。
2. B 的 [`mujoco/models/mine_uav_dynamics_v2.xml:11-23`](../mujoco/models/mine_uav_dynamics_v2.xml) 有 `lidar_visual` mesh/geom，但没有 LiDAR site、传感器扫描定义或安装位姿标定；视觉网格**不是**传感器。原模型 SHA256 为 `cde862f424a86bda3676249859268c69dede9dde2ebadeef819813351684b520`，本轮保持不变。
3. 实机：A 的 [`docs/real_mid360_px4_bench_integration_2026-09-18.md:1-6`](https://github.com/dazhihe7005/-frontier-/blob/db897c4e3a1607f5852b484afd0a01523cb08c6b/docs/real_mid360_px4_bench_integration_2026-09-18.md#L1-L6) 明说雷达曾独立移动、该次未做刚性安装外参验收。本轮固定源码中**未找到可核实的实机标定值**；这不证明其它地点/日期从未标定。机体→LiDAR 平移 XYZ、Roll/Pitch/Yaw、LiDAR→IMU 外参与时间偏差对本轮均为 **UNVERIFIED**；后续需记录机体 FLU/FRD、LiDAR 轴向和方向矩阵定义。不能用旧 SDF 或 MuJoCo mesh 反推这些量。

## 结论与仍需取得的证据

真实链路的网络地址、项目启动拓扑和历史 topic 频率有明确记录；D/F 可提供消息与算法代码层面的**参考**。但现场两个包的确切安装提交、MID360S 与 MID360 扫描等效性、实时 ROS 消息类型、UDP 包解析、真实安装外参及同步质量均未通过本轮验证。具体 MuJoCo 集成设计见 [MUJOCO_MID360S_INTEGRATION_PLAN.md](MUJOCO_MID360S_INTEGRATION_PLAN.md)。本阶段接口审计完成；模拟扫描、FAST-LIO2 运行和 PX4 Odometry 迁移均为 **NOT TESTED**。

离线验证：新增 6 项报告/模型/证据边界测试通过，原 MuJoCo 模型 SHA256 复核通过，独立只读复审修正 1 项 Important 证据措辞及 2 项 Minor 接口表述后结论 `GO`、无剩余 Critical/Important。旧 Bridge 物理接口测试在当前默认与 `px4-venv` Python 中均不能完整运行：本机缺少 Python `mujoco` 安装包，combined pytest 曾报 9 项依赖相关失败、9 项通过；这不是本轮修改导致的功能回归，也不能记为旧测试 PASS。本轮未安装新依赖。
