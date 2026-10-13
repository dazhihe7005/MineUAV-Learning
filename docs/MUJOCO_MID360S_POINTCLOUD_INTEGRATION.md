# MuJoCo MID360-like 点云集成（Phase 3E-L2）

日期：2026-10-13。分支：`feat/mujoco-mid360s-pointcloud-integration`；基线 `5ffd67922eab7e1549ee87bc94184e37b8194700`。本轮仅在静态隔离场景执行真实 MuJoCo 射线求交；没有 PX4、SITL、飞行、ROS 设备发布或网络训练。机器测量见 [`mujoco_mid360s_pointcloud_integration.json`](../mujoco/reports/mujoco_mid360s_pointcloud_integration.json)，原始 NPZ 保留本地。

## 来源、安装与物理隔离

- 历史解释器 `/home/keyousuke/MineUAV-Learning/.venv/bin/python` 能加载真正的 MuJoCo 3.14.0 包；检查了 `importlib.metadata.version("mujoco")`、`mujoco.__file__` 位于 venv `site-packages` 和实际 `MjModel` API，避免仓库同名 `mujoco/` namespace 的假阳性。NumPy 2.2.6；该 venv 没有 SciPy 和 pytest。本轮未安装或升级任何系统依赖；用系统 pytest 配合该 venv 的 site-packages 运行测试。
- 复用 [`discoverse-dev/MuJoCo-LiDAR`](https://github.com/discoverse-dev/MuJoCo-LiDAR/tree/e6b879df4b96daaed30ef10fe02650d29048fa9b) MIT 许可固定提交 `e6b879df4b96daaed30ef10fe02650d29048fa9b`，本地只读副本 `/tmp/mineuav-mujoco-lidar-audit`；运行时核对 Git HEAD、扫描核心源码未修改、导入模块路径、版本 0.3.5 和 `mid360.npy` SHA256 `9fa0165576f0060488254f04bd647a891004d4f4faa6e9b03b69d830c117c9a3`。未复制上游源码；其 `LICENSE` 保留在独立上游副本。使用 `MjLidarWrapper(..., backend="cpu")`、`LivoxGenerator("mid360")` 和底层 MuJoCo `mj_multiRay`。Taichi/JAX/Warp 均未启用。
- [`scene.py`](../mujoco/lidar_integration/scene.py) 在**内存 XML 副本**中为 `mine_uav` 增加 `lidar_site`，并增加地面、前墙、两侧墙和箱体。历史 [`mine_uav_dynamics_v2.xml`](../mujoco/models/mine_uav_dynamics_v2.xml) 文件 SHA256 始终为 `cde862f424a86bda3676249859268c69dede9dde2ebadeef819813351684b520`。测试比较原模型与隔离模型，机体质量均 7 kg、质心/惯量、电机控制范围/传动、四电机数量、0.002 s 物理步长均不变。
- 假设安装外参 `T_body_lidar`：机体 FLU 下平移 `(0.10, 0, 0.10) m`、RPY `(0, 0, 0) rad`；来源 **SIM_CONFIG**，不是实测 MID360S 标定，也未采用旧 Gazebo SDF 外参。世界为 ENU；LiDAR 局部 XYZ 到世界由 MuJoCo 站点的实际 `xmat/xpos` 刚体变换完成。`bodyexclude=mine_uav` 排除同一机体自回波；专设自机球体测试证明可排除自身而仍保留外部墙体。

## 帧契约及真实扫描验证

[`PointCloudFrame`](../mujoco/lidar_integration/pointcloud.py) 保存仿真采样时钟 `frame_timestamp`、`MUJOCO_SIM_TIME`、`lidar_site`、所有射线原始 `ranges`、`valid_mask`、局部 `points_xyz`、局部射线方向、世界传感器位姿、`T_body_lidar`、扫描配置和来源 `MUJOCO_RAYCAST_APPROX_MID360`。无回波的原始距离是 `-1`，对应点为 NaN 且 mask 为 False；位于最短/最长配置范围外的原始命中同样被过滤，不构造原点假障碍。固定 CPU 后端的 `cutoff_dist` 不总是把超量程原始命中改写为 `-1`，因此必须由适配层再按范围过滤。NPZ 原子保存、不使用 pickle；绘图只取有效射线。

本轮单帧是**单一冻结机体位姿**的几何扫描；帧时钟按 0.1 s 前进，但不积分无人机飞行，也不模拟每条射线的物理采样时刻。`per_point_time_offset=UNSUPPORTED`；无运动畸变、反射强度、扫描线/tag、噪声或 IMU 同步。`mid360.npy` 只是上游 MID360 **近似角度模式**，不得解释为 MID360S 的真实光学/时序模型。

真实 MuJoCo 射线的解析几何对照：前墙 2.9 m、左右墙各 2.0 m、地面 0.8 m；最大绝对距离误差 `1.19×10⁻⁸ m`。箱体前表面校验误差 `1.58×10⁻⁹ m`，证明遮挡；无回波射线为 `-1` 且不形成点。原地偏航 +90° 后的朝前射线命中侧墙，距离误差 `1.49×10⁻⁹ m`；Roll/Pitch、平移和安装偏移另有单元测试。局部→世界→局部最大往返误差 `2.22×10⁻¹⁶ m`。24,000 射线全量帧的世界点高度跨度 `2.58 m`，不是二维伪点云。

图像全部来自真实射线结果（黑/红轮廓仅是解析场景参考）：[墙壁](../mujoco/reports/mid360s_pointcloud_figures/wall_pointcloud.png)、[箱体](../mujoco/reports/mid360s_pointcloud_figures/box_pointcloud.png)、[旋转前后](../mujoco/reports/mid360s_pointcloud_figures/rotated_pointcloud_comparison.png)、[高度剖面](../mujoco/reports/mid360s_pointcloud_figures/pointcloud_height_profile.png)。最后的全量局部点云保存在被忽略的 `mujoco/reports/mid360s_pointcloud_integration_parts/local_frame.npz`，SHA256 `eeddfadacb6b8c7b5e716e3295b520b5577193ae8a8de5a099f486c339d63e17`。

## 本机 CPU 性能

每档 12 帧，计时不含模型初始化、NPZ 保存或绘图；以下是本机静态小场景测量，不能外推至复杂矿井或动态飞行。

| 射线/帧 | 有效点均值 | 扫描整帧 P50 / P95 / P99 / Max | CPU 求交 P50 / P95 | 10 Hz 静态扫描预算 |
| ---: | ---: | ---: | ---: | --- |
| 1,000（显式 ×24 降采样） | 686 | 0.418 / 0.687 / 0.804 / 0.834 ms | 0.182 / 0.249 ms | PASS |
| 4,000（显式 ×6 降采样） | 2,770 | 1.155 / 1.213 / 1.248 / 1.257 ms | 0.610 / 0.646 ms | PASS |
| 24,000（完整模式） | 16,396 | 5.906 / 6.545 / 6.613 / 6.630 ms | 3.435 / 3.972 ms | PASS |

完整帧点转换 P50/P95 为 `2.432/2.533 ms`；初始化 `564.77 ms`（场景编译）与 `8.43 ms`（模式/适配器）。保存 NPZ `27.06 ms`，图像绘制每张约 `53–81 ms`，均独立于扫描预算。单 worker、进程峰值 RSS `845.5 MiB`，无 OOM 迹象。本轮不宣称 10 Hz 动态 LiDAR+控制一体运行，仅验证该静态场景下完整扫描耗时低于 100 ms。

## 复现与验收边界

在上游只读目录固定到上述提交且 `LICENSE` 存在后，使用。临时 `/tmp` 副本在重启后可能消失，可重新执行 `git clone https://github.com/discoverse-dev/MuJoCo-LiDAR.git /tmp/mineuav-mujoco-lidar-audit`，然后 `git -C /tmp/mineuav-mujoco-lidar-audit checkout e6b879df4b96daaed30ef10fe02650d29048fa9b`；运行时仍会核对固定身份，不信任目录名称。

```bash
export MINEUAV_LIDAR_UPSTREAM=/tmp/mineuav-mujoco-lidar-audit
export PYTHONPATH=/home/keyousuke/MineUAV-Learning-px4-sitl/mujoco:$MINEUAV_LIDAR_UPSTREAM/src:/home/keyousuke/MineUAV-Learning/.venv/lib/python3.10/site-packages
python3 -m pytest -q mujoco/lidar_integration
python3 -m lidar_integration.runner --upstream "$MINEUAV_LIDAR_UPSTREAM" --output mujoco/reports/mid360s_pointcloud_integration_parts --figures-dir mujoco/reports/mid360s_pointcloud_figures --report-file mujoco/reports/mujoco_mid360s_pointcloud_integration.json --repetitions 12
```

此外，现有 Bridge 与上一阶段 6 项接口审计回归共 **520 PASS**；本轮定向集成测试 **31 PASS**。Matplotlib 发出既有 Axes3D 多版本警告；所有图使用 2D 正交投影，未影响 PNG 生成。

**PASS**：真实 MuJoCo 点云、已知几何距离/遮挡/无回波、自机排除、刚体坐标转换、NPZ、历史模型哈希、静态场景全量 24k 扫描 10 Hz 预算。**NOT TESTED**：真实 MID360S 精度/端口/动态扫描、逐点时序、运动畸变、IMU 同步、Livox CustomMsg、FAST-LIO2、ROS、PX4/SITL、飞行及 PPO 训练。后续 LiDAR+PPO 可复用有限值有效点、mask、位姿和 10 Hz 帧接口，但仍需单独定义固定尺寸观测、矿井场景覆盖和训练协议；FAST-LIO2 还需要可信逐点时间、同步 IMU、测量外参、噪声/运动畸变及真实消息兼容性验证。

独立只读复审：**GO，0 Critical / 0 Important**。复审者核对了固定上游 CPU `mj_multiRay` API、自回波过滤、无回波 `-1` 语义、点云变换、原模型/模式/NPZ 哈希及报告边界，并独立运行复审时的 30 项真实 MuJoCo 测试（之后仅新增一项“真实 MuJoCo 分发包身份”测试，最终本地 31 项通过）。唯一非阻塞建议是日后扩大上游完整性清单；当前上游仓库本身 clean、固定核心运行文件与模式已校验。
