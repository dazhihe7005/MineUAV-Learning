# MuJoCo UAV baseline（四旋翼 4-rotor wrench）

本目录只用于 MuJoCo 学习和外观核对，不包含真机可用的动力学标定或控制器。环境为项目根目录的 `.venv`，不使用系统 Python。

## 官方 Crazyflie 2

`references/crazyflie/` 是 Google DeepMind [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/bitcraze_crazyflie_2) 的 Bitcraze Crazyflie 2 原样参考副本，包括其 `LICENSE`；完整上游仓库作为只读参考放在 `../third_party/mujoco_menagerie/`，检出的 commit 为 `c96a32d28fb5da84da38c1da4d749e7a13212855`。

从项目根目录运行：

```bash
.venv/bin/python mujoco/scripts/inspect_model.py
.venv/bin/python mujoco/scripts/crazyflie_thrust_test.py
.venv/bin/python -m mujoco.viewer --mjcf mujoco/references/crazyflie/scene.xml
```

`scene.xml` 有地面；固定推力脚本用 `cf2.xml`（没有地面）隔离检查 `F < mg`、`F = mg`、`F > mg`，所以零推力试验结束时 z 可小于 0；这不是穿地碰撞错误。每个试验重新创建 `MjData`，没有 PID。此模型的 4 个 actuator 是机体总推力及 X/Y/Z 力矩，**不是四个独立电机**。

`MjModel` 存放编译后的静态模型（质量、几何、关节、actuator 等）；`MjData` 存放随仿真步变化的状态和控制。自由关节的 `qpos[0:3]` 是世界系 xyz，`qpos[3:7]` 是姿态四元数 `(w,x,y,z)`；`qvel[0:3]` 是线速度，`qvel[3:6]` 是角速度。对 **Crazyflie** 而言，`ctrl[0]` 是机体总推力，`ctrl[1:4]` 是三个力矩通道的命令；自有 UAV 模型的 `ctrl` 语义见下文。每次 `mj_step(model, data)` 基于当前控制和物理模型推进一个 timestep，并更新状态；它本身不计算目标控制命令。

## 自有 UAV 几何与四旋翼 wrench 骨架

`reports/stl_report.csv` 与 `reports/assembly_preview_iso.png` 来自此前 `../cad_check/` 的只读总装检查。177/177 个 STL 均读入；原始边界为 `(65.798, 5.774, 657.558)` 到 `(950.682, 535.486, 1542.442)`，extents 为 `(884.883, 529.713, 884.883)` 原始单位。四组机臂、电机、桨叶在装配四角且预览连贯，确定保留了总装位置；由于 STL 无可靠单位，结合无人机尺寸判断更可能是毫米。

```bash
.venv/bin/python mujoco/scripts/prepare_mine_uav_visual.py
.venv/bin/python -m mujoco.viewer --mjcf mujoco/models/mine_uav_visual.xml
.venv/bin/python -m unittest discover -s mujoco/scripts -p 'test*.py' -v
```

生成脚本只读原始 STL，按已有顶点装配坐标选取 57 个主要结构件、8 个 MID360 部件、1 个四目相机部件，生成 3 个 visual mesh 工作副本；螺柱与小连接件不单独建 body。由于 MID360 有单个超过 20 万面的 STL，雷达工作副本使用 OBJ；没有简化三角面。几何副本以 `0.001` 缩放为米，并将原 CAD 的 Y 向上旋转为 MuJoCo 的 Z 向上；这只改变生成的工作副本，不改变源文件。所用几何原点是可视化原点，**不是实测质心**。`mine_uav_visual.xml` 只有一个运动 body、freejoint、3 个 visual mesh 和一个粗略 box collision；质量、质心、惯量、碰撞盒均为 `TEMPORARY_PLACEHOLDER`，不能用于飞行动力学或训练。

## 每个 rotor 一个输入，同时产生推力和 yaw 反扭矩

`ctrl[0:4]` 依次是四电机的 **`ω_i²`，单位 `(rad/s)²`**，不是 RPM、rad/s 或油门百分比。每个 motor site 的单个 MuJoCo site actuator 使用六维 `gear = [0, 0, k_f, 0, 0, s_i k_m]`，因此一个标量输入同时产生 `F_i=[0,0,k_f u_i]` 和 `Q_i=[0,0,s_i k_m u_i]`。不再有独立 yaw-only 输入。四个 site 从 `U7-1` 至 `U7-4` STL 的几何 centroid 得到，单位由 mm 换成 m，轴向与 visual mesh 一致；它们**不是实测推力作用点**。[四电机标记俯视图](reports/motor_sites_debug.png)中红/绿/蓝/黄对应 1/2/3/4 号；为避免被电机 mesh 遮挡，图中的临时标记沿 Z 抬高 0.07 m，XY 仍为真实 site 投影。

`k_f = 4.9977603696661616e-5 N/(rad/s)²`：当前项目此前没有落盘的 `k_f` 值，故从 [T-MOTOR 官方 U7 KV490 + 15×5CF 静态表](https://store.tmotor.com/product/tmotor-u7-v2-motor-u-power.html)的 5 组 RPM/推力数据复现过原点最小二乘拟合。原始表在 `references/tmotor_u7_kv490_15x5cf_static_tests.csv`，脚本在 `scripts/fit_thrust_coefficient.py`；用 `9.80665 m/s²` 把克力读数换算为 N。官方测试 RPM 范围为 5300–8500；下述临时 1 kg 模型的“平衡输入”会落在此范围以下，所以**只能用于方向/符号检查，不能当作推力预测验证**。

`k_m = 8.6e-7 N·m/(rad/s)²`，出处标记 `ESTIMATED_FROM_TMOTOR_15x5CF_STATIC_TESTS`。原始参考表保存在 `references/tmotor_15x5cf_mn5212_kv340_static_tests.csv`，记录 T-MOTOR 官方 **MN5212 KV340 + 15×5CF** 的 RPM/Torque 数据及[原始页面](https://store.tmotor.com/product/mn5212-kv340-motor-navigator-type.html)。它**不是 U7 KV490 官方直接公布的扭矩系数**。`scripts/fit_yaw_torque.py` 可复算：7 个数据点的过原点拟合为 `8.787241e-7`，与采用值相差约 2.13%。

四个旋向符号 `s=[+1,-1,+1,-1]` **只是假设**，集中在 `models/rotor_config.json`，并标为 `TEMPORARY ROTATION CONFIG`。修改它后运行 `scripts/build_rotor_model.py` 重生成 `models/rotor_actuators.xml`，无需逐个修改 actuator。未测真实 CW/CCW 对应关系，不能声称符号已确认。没有加入 RPM 上限、motor lag、ESC dynamics 或控制器。

```bash
.venv/bin/python mujoco/scripts/fit_thrust_coefficient.py
.venv/bin/python mujoco/scripts/fit_yaw_torque.py
.venv/bin/python mujoco/scripts/build_rotor_model.py
.venv/bin/python mujoco/scripts/verify_rotor_wrench.py
```

验证脚本用 `F=ΣF_i`、`τ=Σ(r_i×F_i)+ΣQ_i` 独立计算理论 wrench；这里的 `r_i` 相对 **TEMPORARY body origin**，不是实测 COM。它同时按 `[Fz,Tx,Ty,Tz]^T = B[u1,u2,u3,u4]^T` 自动生成 `reports/mixing_matrix.txt`，再逐项比对 MuJoCo actuator 的广义力。四组差动试验只查方向、符号与对称性；0.02 秒短仿真的角速度大小受临时质量/惯量支配，不是真机响应时间。等速 collective 存在小的 `Tx/Ty` 残差，因为临时 body 原点不在四电机几何中心，验证脚本不会人为消除它。

旧 `mine_uav_visual.xml` 的质量、COM、惯量仍是临时占位；下述 v2 才采用确认的整机质量和估算的 COM/惯量。仍需核实电机真实旋向和推力中心、单电机最大推力、实际 `k_f/k_m` 曲线、电机响应时间及 ESC 动态。现有碰撞盒也是 `TEMPORARY_PLACEHOLDER`。

## 7 kg 工程近似模型 v2

`models/mine_uav_dynamics_v2.xml` 是**新文件**，不覆盖 `mine_uav_visual.xml`。总质量采用用户确认的 `7.0 kg`；两块电池各 `1.60 kg` 为估计值；NUC `0.5775 kg`、四目相机总成 `0.880 kg` 为已测值；MID360 `0.265 kg`、四个 U7 各 `0.299 kg`、四个桨各 `0.013 kg`、飞控 `0.0593 kg` 采用用户提供值（未另行标注测量状态）。已单独赋值质量合计 `6.2298 kg`，余下 `0.7702 kg` 全部分给 structure。

```bash
.venv/bin/python mujoco/scripts/estimate_dynamics_v2.py
.venv/bin/python mujoco/scripts/verify_dynamics_v2.py
.venv/bin/python -m mujoco.viewer --mjcf mujoco/models/mine_uav_dynamics_v2.xml
```

估计脚本只读 177 个原始 STL，沿用 visual 模型的 CAD Y-up → MuJoCo Z-up 坐标变换。将 2 块电池、NUC、四目相机、8 个 MID360 相关 STL、4 个 U7、4 个桨和飞控从 structure 清单排除，剩余 **156 个结构 STL 均为 watertight**。每件结构 mesh 按自身封闭体积占比获得质量，并用该均匀体积几何求自身 COM 与惯量。四个 U7 和 MID360 中两个 STL 非 watertight；赋质量的 U7、MID360 总装 mesh 因而使用**均匀三角面面积分布**估计自身 COM/惯量，而非伪称封闭体积。MID360 仅总装 mesh 承担 `0.265 kg`；其余七个子件文件质量置零，避免重复计入。每件的分类、质量、几何方法和中心见 `reports/dynamics_v2_component_manifest.csv`；完整数值及假设见 `reports/dynamics_v2_report.json`。

整机 `ESTIMATED_COM_V2 = Σ(m_i r_i)/7`，`ESTIMATED_INERTIA_V2` 则是各组件自身几何惯量加平行轴项后的完整 `3×3` 张量。二者均相对当前 body origin，**不是实测值**。MJCF 的 `fullinertia` 使用 `Ixx Iyy Izz Ixy Ixz Iyz` 顺序。`verify_dynamics_v2.py` 用 `r_motor = motor_site − ESTIMATED_COM_V2` 重算 `reports/mixing_matrix_v2.txt`，再核对四组差动。MuJoCo freejoint 的 `qfrc_actuator` 力矩数值相对 body origin；验证脚本将其减去 `COM × F` 后，才与关于估计 COM 的理论 wrench 比较。

请勿把等速 collective 默认视为零力矩：估计 COM 的 x 坐标相对电机几何中心有偏移，因此 v2 的 collective pitch 力矩比旧 body-origin 结果更大。这是输入质量分布的计算结果，不是混控符号错误。飞行控制器、PID、RL、电机延迟均未加入；碰撞盒、实机旋向、推力中心以及 `k_m` 仍待验证。

## 开环 hover trim 与有界 control allocation

`control/control_allocator.py` 从 v2 MJCF 的 COM、四个 motor site 及统一的 `rotor_config.json` 构造 B。输入 `[Fz, Tx, Ty, Tz]`，输出 `u_i=ω_i²`；若无界解满足 `0≤u_i≤u_max`，直接精确求解。否则枚举四个电机的下限、自由、上限三种状态（最多 81 种），求非负且不超过上限的最小加权残差解。不同物理量的残差按各行在给定转速上限下的最大绝对作用量归一化；不可达请求会返回实际 wrench、误差和触发的界。`8500 rpm` 是本阶段采用的测试数据转速上限，**不是经过验证的安全额定限速**。无需安装 SciPy，也没有 PID 或电机延迟。

```bash
.venv/bin/python mujoco/scripts/hover_trim_test.py
.venv/bin/python -m unittest discover -s mujoco/scripts -p 'test*.py' -v
```

`hover_trim_test.py` 求解 `B u_hover=[7×9.81,0,0,0]`，打印四个 `u/ω/rpm` 与 `B u`，并从 body-origin `z=1 m`、水平姿态、零速度起施加恒定输入 5 秒。它还从相同初态对等转速输入运行 0.3 秒，以避开较晚可能出现的地面接触对 pitch 对比的干扰。`reports/hover_trim_report.json` 保存最终状态、初始角加速度、MuJoCo 相对 COM 的初始 wrench、五个分配请求及一个不可达请求；`reports/hover_trim_trajectory.csv` 每 0.1 秒记录位置、速度、姿态和角速度。数值悬停是此估计模型中的理想开环平衡，不能推断真机无需闭环控制。

## 第一版闭环：高度 / 姿态 PD

`control/hover_controller.py` 仅依据世界系位置与线速度、MuJoCo `wxyz` 姿态四元数、**body 系**角速度请求 `[Fz,Tx,Ty,Tz]`，由既有 allocator 转成四个 `ω_i²`。高度项为 `mg + Kp_z(z_target−z) − Kd_z vz`，除以机体 +Z 与世界 +Z 的夹角余弦作倾角补偿，再限制到可实现总推力；机体倒置时不命令向下的“悬停推力”。姿态使用四元数最短旋转的 body 系轴角误差，不直接相减 Euler 角；期望角加速度为 `Kp_att·error − Kd_att·ω`，扭矩为 `I·α_cmd + ω×(Iω)`。叉乘项补偿刚体旋转耦合；`I` 是 `ESTIMATED_INERTIA_V2`，并非实测惯量。没有积分项或 x/y 位置控制。

```bash
.venv/bin/python mujoco/scripts/run_closed_loop_hover.py
```

初始保守参数即最终采用值：`Kp_z=14 N/m`、`Kd_z=14 N/(m/s)`；roll/pitch/yaw 的角加速度型 `Kp_att=(6,6,2) s⁻²`、`Kd_att=(4,4,2.8) s⁻¹`。`Kp` 决定恢复力度，`Kd` 提供阻尼，未做暴力搜索。脚本对低/高高度、±15° 单轴姿态、组合扰动及 2.0–2.1 秒的短时 `−21 N` 世界系竖直外力分别仿真 10 秒。每个案例的全步长 CSV 保存 time、xyz/vxyz、RPY、角速度、目标与实际 wrench、四电机 RPM、allocator 饱和标记；`reports/closed_loop_hover_summary.json` 保存超调、最大姿态误差、最大 RPM 和稳定时间。稳定时间定义为此后持续满足 `|z−1|≤0.03 m`、`|vz|≤0.05 m/s`、各姿态误差 `≤2°`、各角速度 `≤0.1 rad/s` 的最早时刻；外力案例从扰动结束后算起。三张 PNG 为高度、姿态和转速曲线。

**限制：** 姿态倾斜引起的横向加速度会留下横向速度，模型没有气动阻力，当前控制器也不调节 x/y。因此“恢复悬停”在本阶段仅指高度与姿态重新稳定，**不表示 x/y 定点悬停**；10 秒单轴倾斜案例可横移约 16.6 m。不能把本数值实验外推为真机飞行性能。

## XY + Z 位置闭环（新增）

`control/position_controller.py` 是新的统一位置外环；上节 `HoverController` 的旧高度控制入口保留不变，同时开放共用 `attitude_torque(current_q, body_omega, desired_q)`。位置控制器不直接生成电机指令，完整链路为：

`world p_d,v_d → world a_des → world F_des → body-to-world R_des/q_des + scalar thrust → shared quaternion attitude PD → body [Fz,Tx,Ty,Tz] → bounded allocator → ctrl[0:4]=ω_i²`。

四个独立函数分别是：

1. `position_error_to_acceleration()`：输入世界系 p、v、p_d、v_d；输出 `a_des=Kp(p_d−p)+Kd(v_d−v)`，水平向量模长限 `3 m/s²`，垂直加速度限 `±3 m/s²`。
2. `acceleration_to_thrust_vector()`：读取 MJCF 重力 `g_world=[0,0,−9.81] m/s²`，输出世界系 `F_des=m(a_des−g_world)`。在零加速度时得到 `[0,0,68.67] N`。
3. `thrust_vector_to_attitude()`：期望 body +Z 轴沿 `F_des`；将目标 yaw 的水平朝向 `h=[cosψ,sinψ,0]` 与其做正交基，令 `b_y=normalize(b_z×h)`、`b_x=b_y×b_z`，输出 `R_des=[b_x,b_y,b_z]` 及 MuJoCo `wxyz` 四元数。矩阵列是 **body 轴在 world 中的方向**，Euler 角只用于日志。
4. `attitude_to_wrench()`：输入推力模长、当前/目标姿态和 body 系角速度；调用共享姿态环求 Tx/Ty/Tz，输出 body 系 `[Fz,Tx,Ty,Tz]`；推力限制在 allocator 可提供总推力内。

位置增益最终为 `Kp=(0.5,0.5,2.0) s⁻²`、`Kd=(1.2,1.2,2.0) s⁻¹`；姿态增益沿用 `(6,6,2) s⁻²` 与 `(4,4,2.8) s⁻¹`。初试水平增益 `(1.5,1.5)` / `(2.4,2.4)` 导致姿态环滞后、位置持续振荡；调低水平外环带宽后六个案例均收敛。没有积分项。运行：

```bash
.venv/bin/python mujoco/scripts/run_position_loop.py
.venv/bin/python -m unittest discover -s mujoco/scripts -p 'test*.py' -v
```

`reports/position_loop_summary.json` 记录六例最终误差、稳定时间、最大倾角/RPM 与饱和计数；`reports/position_loop_*.csv` 每 0.02 秒记录位置/速度、期望加速度、期望/实际姿态、wrench、电机 RPM 和误差。`reports/position_response.png`、`reports/xy_trajectory.png`、`reports/attitude_command_response.png` 是直接可看的曲线。E 在 `t=2.0–2.1 s` 对机体 COM 施加世界 +X `21 N` 短时水平外力；F 在 `t=2 s` 将 yaw 目标从 `0°` 阶跃到 `45°`，xyz 目标不变。稳定时间要求从该时刻以后持续满足位置误差 `<0.05 m`、速度模长 `<0.05 m/s` 和 yaw 误差 `<2°`。这些仅是当前估计惯量/理想电机/临时旋向模型中的仿真结果，不是实机性能保证。
