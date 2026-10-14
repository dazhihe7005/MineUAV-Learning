# Phase 3E-L3 — MuJoCo MID360-like 动态位姿扫描与时序验证

日期：2026-10-14。基础提交 `afd7d0fda3431dedcf180f49e59f0a01dbd940de`；分支 `feat/mujoco-mid360s-dynamic-scanning`。本轮只验证运动学位姿回放上的**真实 MuJoCo 几何射线**；PX4/SITL、真实雷达、ROS、FAST-LIO2 和神经网络训练均未运行。完整数值见 [JSON 报告](../mujoco/reports/mujoco_mid360s_dynamic_scanning.json)。

## 复用与实验边界

使用历史 `/home/keyousuke/MineUAV-Learning/.venv` 中真正的 MuJoCo Python 3.14.0（`site-packages/mujoco/__init__.py`、`MjModel` API 和发行包元数据均核对），NumPy 2.2.6。外部 MIT 仓库 [`discoverse-dev/MuJoCo-LiDAR`](https://github.com/discoverse-dev/MuJoCo-LiDAR/tree/e6b879df4b96daaed30ef10fe02650d29048fa9b) 固定到 `e6b879df4b96daaed30ef10fe02650d29048fa9b`；`mid360.npy` SHA256 为 `9fa0165576f0060488254f04bd647a891004d4f4faa6e9b03b69d830c117c9a3`。继续调用已有 [`Mid360Scanner`](../mujoco/lidar_integration/scanner.py) 的 `MjLidarWrapper(..., backend="cpu")`，没有重新实现射线求交或启用 GPU。

原始 [`mine_uav_dynamics_v2.xml`](../mujoco/models/mine_uav_dynamics_v2.xml) SHA256 仍为 `cde862f424a86bda3676249859268c69dede9dde2ebadeef819813351684b520`；7 kg 质量、质心、惯量、四电机和 0.002 s 物理步长不变。新的 [`trajectory.py`](../mujoco/lidar_integration/trajectory.py) 每个 500 Hz tick 用预注册位置/四元数覆写机体 `qpos`，再以 `mj_step` 推进 MuJoCo 时钟；这是 **KINEMATIC_POSE_REPLAY**，不是闭环或物理一致飞行。每 50 tick 触发一次 10 Hz 扫描。每帧在**单一冻结位姿**下求交；真实逐点时刻、运动畸变、MID360S 光学真实性均 **NOT TESTED**。没有用 `time.sleep` 控制采样，也没有手工改写 `data.time` 来伪造新观测。

六条 0–1 s、每条 11 帧的预注册路径：X 直线 `+0.4 m`、Y 直线 `+0.3 m`、Z 三角路径 `0.7→0.9→0.7 m`、原地 Yaw `0→90°`、组合运动 `(X,Y,Z)=(+0.25,+0.10,+0.10)m` 且 Yaw `+60°`、竖井内 `Z=0.7→1.7 m`。LiDAR 相对机体 FLU 为平移 `(0.10,0,0.10)m`、零 RPY，来源 **SIM_CONFIG**，不是实测 MID360S 外参。世界坐标为 ENU；点云保存本地 LiDAR XYZ 与世界变换所需的完整位姿。自机回波继续以 `bodyexclude` 排除；无回波 `-1` 对应无效 mask/NaN 点，不伪造原点障碍。

## 实测几何与时序

六条路径 66 帧完整 24,000 射线，加三档性能对照 33 帧，合计 **99 次模式扫描**。所有 11 帧的仿真时刻为 `0,0.1,…,1.0 s`，相邻周期的最大数值误差 `8.33×10⁻¹⁷ s`；每段确实执行 500 次 `mj_step`，采样/CPU 耗时分开。该结论是**仿真时钟采样**正确，不代表 PX4+LiDAR 联合实时系统已验证。

用同一 CPU 射线后端发出的固定方向诊断射线与独立平面公式对比：X 靠墙真距离 `2.9−0.4t m`，最大误差 `1.31×10⁻⁸ m`；Y 靠墙真距离 `2.0−0.3t m`，最大误差 `1.19×10⁻⁸ m`。点云在世界 ENU 中的前墙点位置跨五条室内轨迹最大偏差 `7.45×10⁻⁸ m`；姿态变换后墙不随无人机“移动”。Yaw 轨迹在 `t=1 s` 的雷达 +X 轴转为世界 +Y，局部朝前探测从前墙转向左侧墙。Z 路径存在上升及下降两段，不是只抽取上升样本。

竖井是独立内存模型中的 24 边、半径 `2.5 m`、高度 `6 m` 的**均匀竖直壁近似**，不改历史动力学模型。固定水平射线的预期前/左/右距离 `2.4/2.5/2.5 m`，从机体 `Z=0.7` 到 `1.7 m` 的实测距离跨度为 `0 m`，同一局部测距不能独自约束竖直位移；世界坐标命中点的 Z 则随传感器位姿变化。存在地面、有限高度、斜向射线和时序几何等其它约束，本轮**不**把“水平射线退化”扩展为 FAST-LIO2 必然失效结论。

### CPU 性能（组合轨迹，每档 11 帧）

| 真实射线/帧 | 降采样 | 扫描整帧 P50 / P95 / P99 / Max | 有效点范围 |
| ---: | ---: | ---: | ---: |
| 1,000 | ×24 | `0.446 / 0.719 / 0.734 / 0.737 ms` | `645–712` |
| 4,000 | ×6 | `1.190 / 1.301 / 1.303 / 1.304 ms` | `2,589–2,846` |
| 24,000 | 无 | `6.150 / 6.919 / 7.012 / 7.036 ms` | `15,530–17,034` |

计时包含模式取角、MuJoCo CPU 求交和点转换；不含模型初始化、轨迹 `mj_step`、额外的五方向诊断射线、NPZ 压缩或绘图。完整帧数组占 `1,368,256 bytes`（约 `1.30 MiB`），单进程峰值 RSS `1,193.4 MiB`，未观察到 OOM。完整模式 P95 小于 100 ms 的 10 Hz **扫描自身**预算；11 帧的小场景样本不代表矿井复杂场景最坏时延，也不是 PX4 联合性能。

## 点云、图与 IMU 时间契约

66 帧连续 NPZ 和各帧 SHA256 存在本地忽略目录 `mujoco/reports/mid360s_dynamic_scanning_parts/run_02/`，约 65 MiB；原始数组不提交。六张图均由**实际扫描射线**绘制，解析线只作为显式对照：[动态轨迹](../mujoco/reports/mid360s_dynamic_scanning_figures/dynamic_trajectory.png)、[连续点云](../mujoco/reports/mid360s_dynamic_scanning_figures/continuous_pointclouds.png)、[局部/世界坐标](../mujoco/reports/mid360s_dynamic_scanning_figures/local_vs_world_pointcloud.png)、[墙距](../mujoco/reports/mid360s_dynamic_scanning_figures/wall_distance_vs_time.png)、[竖井几何](../mujoco/reports/mid360s_dynamic_scanning_figures/shaft_vertical_geometry.png)、[扫描时延](../mujoco/reports/mid360s_dynamic_scanning_figures/scan_timing.png)。

已审查 [`SensorAdapter.sample`](../mujoco/px4_bridge/sensor_adapter.py)：旧 SITL 以 250 Hz 调用，`mj_forward` 后用 COM Jacobian、`jacDot`、`qacc/qvel` 得到加速度/角速度，减去重力并转到 FRD。当前 LiDAR 与预留 IMU 调度可共享 500 Hz 整数 tick：LiDAR 每 50 tick，IMU 每 2 tick；但本轮**没有生成 IMU 数据**。直接改 `qpos/qvel` 不提供物理一致的加速度，不能把旧 SensorAdapter 的输出用于本回放的真实 IMU 验证。这里的微秒是从仿真 `t=0` 起算的相对时间；旧 PX4 `SimClock` 另加 `1,000,000 µs` 启动偏移，不能直接把两个时间域相减称为延迟。LiDAR–IMU 实测外参、真实逐点 offset、运动补偿及 FAST-LIO2 同步输入全为 **NOT TESTED**。

## 验证、复现及准入边界

固定上游仓库可在独立目录 clone 后 checkout 上述提交，运行时会核对核心文件和扫描模式身份；`LICENSE` 保留在上游副本。相关命令：

```bash
export MINEUAV_LIDAR_UPSTREAM=/tmp/mineuav-mujoco-lidar-audit
export PYTHONPATH=/home/keyousuke/MineUAV-Learning-px4-sitl/mujoco:$MINEUAV_LIDAR_UPSTREAM/src:/home/keyousuke/MineUAV-Learning/.venv/lib/python3.10/site-packages
python3 -m pytest -q mujoco/lidar_integration mujoco/px4_bridge mujoco/tests/test_mid360s_interface_audit.py
python3 -m lidar_integration.dynamic_runner --upstream "$MINEUAV_LIDAR_UPSTREAM" --output mujoco/reports/mid360s_dynamic_scanning_parts/new_run --figures-dir mujoco/reports/mid360s_dynamic_scanning_figures_new_run --report-file mujoco/reports/mujoco_mid360s_dynamic_scanning_new_run.json
```

原始输出目录必须为空，正式报告不能已存在，图目录不能含旧文件；运行前检查拒绝覆盖任何这些证据。最终测试和独立审查结果见本轮 JSON 的 `verification` 记录及实验日志。

**PASS**：移动站点姿态、连续真实点云、10 Hz MuJoCo 时钟、解析距离、静态世界墙位置、竖井固定水平射线几何及全量 24k 扫描预算。**NOT TESTED**：动力学一致的 IMU、真实 MID360S 逐点时间/失真、LiDAR–IMU 同步、FAST-LIO2、PX4/SITL、飞行和 PPO。未来构造 PPO 观测还须定义固定尺寸编码、无回波 mask、视场/距离噪声及矿井场景泛化；本轮不启动训练。
