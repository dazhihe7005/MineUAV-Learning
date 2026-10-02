# MineUAV-Learning：项目全过程、问题诊断与面试准备

> 这个文档用于记录 MineUAV-Learning 从真实无人机建模、经典控制、Gymnasium/PPO，到鲁棒性、Sim2Real 前置分析的完整过程。重点不是“做了什么”，而是“为什么这么做、遇到什么问题、怎么定位、证据是什么”。

## 1. 项目目标

目标是为约 7 kg 的矿井无人机构建一条 Robot Learning 主线：

```text
真实无人机参数
→ MuJoCo 动力学
→ 经典低层控制
→ Gymnasium
→ PPO 高层速度策略
→ Robustness / Sim2Real
→ ROS2 / PX4
→ LiDAR / Camera
→ World Model / WAM
```

RL 不直接控制四个电机，而是输出：

```text
[vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd]
```

底层仍负责姿态、推力和电机分配。

## 2. MuJoCo 动力学建模

已完成：
- 7 kg 整机质量建模；
- U7 KV490 + 15×5CF 推力模型；
- `F_i = k_f * omega_i^2`；
- yaw reaction torque 模型；
- COM / inertia 工程估计；
- 四个电机相对 COM 的力臂；
- control allocation；
- hover trim。

关键认识：不是追求“仿真绝对真实”，而是建立可解释、可校准、可做控制和学习实验的物理基线。

## 3. 经典控制基线

依次完成：
1. 高度控制；
2. 四元数姿态控制；
3. XYZ position controller；
4. velocity-command controller。

结构：

```text
position / velocity command
→ desired acceleration
→ desired thrust vector
→ desired attitude
→ quaternion attitude controller
→ desired wrench
→ allocator
→ motor thrust
```

关键工程经验：一开始水平外环太快、姿态内环跟不上，导致 XY 振荡。降低外环带宽后改善，说明级联控制中外环应明显慢于内环。

## 4. Gymnasium RL 环境

Observation 7D：

```text
[target_error_x, target_error_y, target_error_z,
 vx, vy, vz,
 yaw_error]
```

Action 4D：

```text
[vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd]
```

归一化范围 `[-1,1]`，映射到：

```text
vx/vy: ±1.5 m/s
vz:    ±1.0 m/s
yaw:   ±1.0 rad/s
```

时序：

```text
physics    = 500 Hz
controller = 100 Hz
policy     = 25 Hz
```

一次 policy step = 0.04 s = 20 个 physics step + 4 次 controller update。

Episode 最大 15 s，即 375 policy steps。

## 5. Scripted Policy：先证明任务可完成

手工策略核心：

```text
v_cmd ∝ position_error
```

离目标越近，速度指令越小。

Scripted policy 可以稳定成功，证明：
- MuJoCo 模型可完成任务；
- 低层控制器可完成任务；
- observation 基本足够；
- action 接口足够；
- waypoint 任务本身可实现。

## 6. 第一版 PPO Baseline

Actor / Critic：

```text
[64,64] MLP
```

PPO 核心参数：

```text
learning_rate = 3e-4
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.2
ent_coef = 0
vf_coef = 0.5
max_grad_norm = 0.5

n_envs = 8
n_steps = 256
batch_size = 256
n_epochs = 10
```

一次 rollout：

```text
8 × 256 = 2048 transitions
```

20k smoke：训练链正常、reward 改善、无 NaN，但 0/20 成功，目标附近振荡。

约 200k baseline：仍接近 0% 成功。策略会飞向 waypoint，但无法稳定停住。

## 7. Reward V2 / V3 / V4

### V2：near-target brake penalty

增加：

```text
r_brake = -0.5 * w(distance) * ||v||^2
```

结果：
- near-target speed 下降；
- action saturation 显著下降；
- 但仍不会稳定停靠。

### V3：持续距离惩罚

加入 `-lambda * distance` 后更差，策略更少进入目标区域。

### V4：切向速度惩罚

针对目标附近 tangential motion 加 penalty，结果仍更差。

关键经验：

> Reward shaping 不是 penalty 越多越好。Agent 可能学会规避惩罚区域，而不是学会任务本身。

## 8. Braking Diagnostic

将速度分解为：

```text
radial velocity
tangential velocity
```

并比较：
- scripted policy；
- PPO V2。

结果发现：
- PPO 不是完全不刹车；
- PPO 会提前给反向指令；
- 但 near-target command 仍然很大；
- 切向速度很大；
- 经常穿越目标、反复纠偏。

Scripted 的 command 则会随距离平滑缩小。

## 9. Curriculum Learning

尝试：
- Stage A：0.25–0.5 m；
- Stage B：0.5–1.0 m；
- Stage C：完整 waypoint 分布。

结果仍未可靠学会停稳。

这排除了一个简单解释：

> 问题并不只是“完整任务太难”。

## 10. 两个关键 Sanity Check

### 10.1 Reward Alignment

同一批 waypoint 比较：
- Scripted；
- PPO V2；
- Zero action；
- Random。

Scripted 的 undiscounted / discounted return 均明显更高。

结论：

> Reward V2 没有出现“成功策略得分反而更低”的简单错位。

### 10.2 Policy Representation

用 scripted 产生：

```text
observation -> scripted action
```

训练 `7 -> 64 -> 64 -> 4` MLP，仅用 MSE。

闭环结果：
- Local：100/100；
- Full：100/100。

说明：
- 7D observation 足够表达 scripted 行为；
- 4D action 足够；
- [64,64] MLP 容量足够；
- 问题转向 PPO 的探索 / 优化过程。

## 11. 关键突破：PPO Exploration Std

SB3 连续动作 PPO：

```text
a ~ Normal(mu(s), sigma^2)
sigma = exp(log_std)
```

默认：

```text
log_std_init = 0
sigma ≈ 1
```

但 XY action 的 normalized 1 对应约 1.5 m/s，因此默认 exploration noise 对精确停靠任务过大。

只改：

```text
log_std_init: 0 -> -2
```

得到：

```text
sigma ≈ 0.135
```

其它不变。

结果单 seed：
- 20k：2/100 full；
- 50k：62/100；
- 100k：100/100。

这是最关键的 PPO 诊断结论之一：

> Exploration 不是越大越好，尺度必须匹配 action 的物理意义与任务精度。

## 12. Multi-seed Validation

5 个独立训练 seed：

```text
fixed benchmark success mean ≈ 95.6%
holdout success mean         ≈ 95.6%
```

最弱 holdout 约 86%。

因此冻结：

# MineUAV RL Baseline v1

```text
Reward V2
+ PPO
+ [64,64]
+ log_std_init=-2
+ 7D observation
+ 4D velocity action
+ P velocity controller
```

## 13. Robustness / Sensitivity Audit

冻结 5 个 PPO，不训练，只改变：
- mass；
- inertia；
- thrust coefficient k_f；
- motor lag；
- constant external force。

结果：
- inertia、motor lag 相对不敏感；
- mass、k_f、constant force 极敏感。

不能马上断言“PPO 不鲁棒”，所以进一步做 attribution。

## 14. Robustness Attribution

比较：
- scripted；
- PPO；
- 低层 zero-velocity hold。

在 mass / k_f / constant force 下，scripted 也显著失败。

+5 N 恒力时，上层持续给约 `-0.47 m/s` 补偿速度命令，实际速度接近 0，但位置仍偏离约 1.3–1.4 m。

结论：

> 主要瓶颈是低层 P velocity loop 对静态模型偏差和恒定扰动的拒斥不足，而不是 PPO 单独过拟合 nominal dynamics。

## 15. P → PI

升级：

```text
a_des = Kp * velocity_error
      + Ki * integral_error
```

使用：
- `Ki_xy = 0.5`
- `Ki_z = 0.8`
- integral acceleration limit：XY/Z 各 1.5 m/s²；
- anti-windup；
- reset 清零积分。

Scripted + PI：

```text
Nominal     100%
mass ±5%    100%
kf ±5%      100%
force ±5 N  100%
```

说明 PI 明显改善恒定扰动拒斥。

但冻结旧 PPO 直接换 PI 后 nominal 从约 95.6% 降到 70.2%，crossing 显著增加。

关键认识：

> 改低层控制器等价于改变 RL 所面对的闭环 dynamics，旧 policy 不能假设无损迁移。

## 16. PPO + PI 重训

仍用 7D observation，从头在 PI 上训练。

seed 0，100k：

```text
benchmark 74/100
holdout   82/100
```

有所恢复，但没有回到原 P baseline。

于是提出新假设：

> PI 引入了隐藏内部状态。

## 17. PI Hidden State / Partial Observability

PI 内部存在：

```text
integral_error
```

同样的：
- position；
- velocity；
- yaw；
- target；

如果 integral state 不同，下一步低层 acceleration 也会不同。

因此原 7D observation 不再完整描述系统状态。

诊断实验：增加三维 PI integral acceleration：

```text
10D =
[target error xyz,
 velocity xyz,
 yaw error,
 integral_accel xyz]
```

其它全部不变。

同一 holdout、seed 0：

```text
A 7D PPO + P                  100/100
B old 7D PPO 直接换 PI          78/100
C 7D PPO 在 PI 上重训          82/100
D 10D PPO + PI               100/100
```

同时：
- near-target actual speed：0.206 → 0.111 m/s；
- crossing：20% → 1%。

结论：

> PI hidden state / partial observability 是 PPO+PI 性能下降的重要因素之一。

目前只完成单 seed，不能宣称跨 seed 稳定。

## 18. 当前技术结论

### RL
- PPO 可以完成 waypoint 停靠；
- Reward V2 基本合理；
- 网络 / observation / action 表示能力足够；
- 默认 exploration std 是早期失败的关键因素之一；
- low-std multi-seed baseline nominal 成功率约 95%。

### Control
- P velocity loop 对恒定扰动存在 steady-state 问题；
- PI 提升 disturbance rejection；
- PI 改变闭环 dynamics；
- PI 引入 hidden state；
- 显式提供 integral state 后 PPO nominal 性能恢复。

### 下一阶段

工程 / Sim2Real：

```text
10D + PI multi-seed
→ disturbance robustness
→ Domain Randomization
→ sensor noise / delay
→ ROS2 / PX4
```

Robot Learning：

```text
hidden state
→ history observation
→ recurrent policy
→ latent dynamics
→ World Model
→ WAM
```

## 19. 面试时最值得讲的一条主线

> 我先根据真实无人机参数建立 MuJoCo 动力学和经典低层控制，再封装 Gymnasium waypoint 环境，让 PPO 输出高层速度指令。最初 PPO 能接近目标却停不住。我没有直接堆 reward，而是依次通过 scripted policy、reward alignment、behavior cloning、action distribution audit 排除环境、表示能力和 reward 的简单问题，最终发现默认 Gaussian exploration std 与实际速度 action 尺度不匹配；只把 `log_std_init` 从 0 改到 -2，多 seed holdout 平均成功率提升到约 95%。随后做 robustness audit，发现 mass/k_f/恒力扰动导致断崖式下降，再通过 scripted/PPO/hold 三路归因到低层 P 速度环的静态扰动问题。加入 PI 后扰动鲁棒性恢复，但 PPO 性能下降；通过 observability ablation 将积分器内部状态加入 observation 后，单 seed holdout 恢复 100%，说明 partial observability 是重要原因之一。

## 20. 高频面试问题

1. 为什么 RL 不直接控制电机？
2. 7D observation 为什么包含 velocity？
3. 500 / 100 / 25 Hz 分别是什么？
4. step / episode / rollout 分别是什么？
5. Actor / Critic 分别做什么？
6. Advantage 是什么？
7. PPO 为什么需要 clip？
8. continuous action 的 `mu` / `sigma` / `log_std` 是什么？
9. 为什么默认 exploration 会导致你的任务失败？
10. 为什么不能只跑一个 seed？
11. 为什么需要 holdout waypoint？
12. 为什么先 sensitivity audit，再 Domain Randomization？
13. 为什么 P 环对恒定扰动存在稳态误差？
14. PI 为什么能改善？
15. 为什么 PI 会造成 PPO 退化？
16. 什么是 partial observability？
17. 为什么不把所有控制器内部状态都直接塞给 policy？
18. history / recurrent policy 与 hidden state 有什么关系？
19. Sim2Real gap 是什么？
20. World Model / WAM 为什么和这个项目有关？

## 21. 简历项目表述建议

**MineUAV-Learning｜矿井无人机 Robot Learning / Sim2Real**

- 基于约 7 kg 实机参数构建 MuJoCo 动力学模型，完成 COM、惯量、电机推力模型、control allocation、hover trim 与级联位置/姿态控制；
- 构建 Gymnasium waypoint 环境，采用 PPO 输出 4D velocity/yaw-rate 高层指令；
- 通过 reward alignment、behavior cloning、action distribution audit 定位 PPO 停靠失败根因；将连续动作 `log_std_init` 从 0 调整至 -2 后，5-seed holdout 平均成功率约 95.6%；
- 设计 mass / inertia / thrust coefficient / motor lag / external-force robustness audit，并通过 scripted/PPO/hold 三路归因定位低层 P velocity loop 的静态扰动问题；
- 引入带 anti-windup 的 PI 速度环，并通过 observability ablation 验证 PI hidden state 导致的 partial observability；显式加入积分加速度状态后，单 seed holdout 恢复 100/100；
- 后续推进 multi-seed PI baseline、Domain Randomization、history-based policy、ROS2/PX4 Sim2Real 与 World Model/WAM。

## 22. 一句话总结

> MineUAV-Learning 不是一个 PPO demo，而是一套围绕真实无人机参数建立的 Robot Learning 系统，通过经典控制、强化学习、可观测性、鲁棒性和 Sim2Real 的逐层实验，研究学习策略如何可靠接入真实无人机控制栈。
