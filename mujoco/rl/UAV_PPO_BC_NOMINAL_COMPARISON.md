# PPO vs Behavior Cloning — Matched Nominal Evaluation

日期：2026-10-10。Branch：`feat/uav-ppo-bc-nominal-comparison`。
Base：`7e93edda8e1625862dc6dafb4baae58dd62ca3f5`（上一阶段 attitude-thrust audit）。

结论：当前 **7D PI-PPO seed0未达到BC的nominal成功率**；旧PI-PPO结果可复现。可以公平比较同一闭环接口/信息条件，但不能据此比较相同训练预算下的算法优劣或样本效率。本轮没有训练，没有改PI/reward/physics/task。

## 1. 兼容性与历史结果辨析

主对照固定：Scripted、Original BC、Yaw-Augmented BC、`ppo_waypoint_pi_lowstd_100k.zip`。在新target生成和结果获取之前预声明最终100k checkpoint；未依据新Test挑选模型。

四者全部使用 `MineUAVPIEnv(reward_version='v2',target_distribution='full')`：

- 7D：world target-position error XYZ(m)、world velocity XYZ(m/s)、wrapped target-minus-actual yaw(rad)。不传入PI integral、attitude、world-model latent或teacher动作。
- 4D：[-1,1] normalized world vx/vy/vz/yaw_rate commands；物理尺度[1.5,1.5,1.0]m/s及1rad/s。
- 同一固定PI：Kp[1.5,1.5,2.0]、Ki[0.5,0.5,0.8]，同一姿态环、allocator、rotor/MuJoCo dynamics；没有外力。
- 500Hz physics、100Hz control、25Hz policy；15s/375policy步。distance<.10m且speed<.15m/s连续5步才success；原物理failure规则不变。
- 每target恢复完整MjData/controller/allocator/task/RNG/previous-action snapshot，零初速、yaw0、position[0,0,1]、PI integral0。四个controller的初始hash逐target相同。

Scripted仅使用同一7D observation；它不是PI状态oracle。BC推理独立，未调用Scripted。PPO用SB3 `deterministic=True` clipped Gaussian mean，未将随机推理混入统计。

历史三种协议不可混用：

| 实验 | 观测/低层 | 独立训练seed | 历史结果 | 本轮用途 |
|---|---|---|---|---|
| RewardV2 lowstd旧PPO | 7D / P速度环 | 20260929,0,8,16,24 | benchmark/holdout均mean95.6%；sampleSD4.827/5.771个百分点 | 历史参考，不作为PI主对照 |
| PI lowstd PPO | 7D / 当前PI | 0 | 100k benchmark74/100，holdout82/100 | 唯一现有兼容主PPO |
| PI-observable PPO | 10D / 当前PI | 0 | 100k两组100/100 | 额外信息历史参考，不重跑/不混入主表 |

10D增加三个world `Ki*integral_error` 加速度项，BC不可见。不能用其100%声称同观测PPO达到BC。原约95.6%没有在本轮旧P环五seed上重跑；本轮复核的是当前可比的PI7D旧74/82。

PPO网络是独立actor/critic7→64Tanh→64Tanh→4means/1value，另4个global log_std，总9673参数。BC均7→128ReLU→128ReLU→4，18052参数。PPO训练seed0，requested100000、actual100352 transitions；8环境×256步rollouts、Adam.0003、batch256、10epochs/update。BC训练预算/专家数据不同，不是random-initialization或training-budget matched replication。

仓库发现642个ZIP文件，但大量为中间epoch、rollout replay、recurrent诊断分支，不能当作642个seed。只有一个满足当前主协议的独立PI7D seed；跨训练seed mean/SD为N/A，不用200targets冒充200个trainingseeds。

## 2. Checkpoint身份

| Frozen model | Checkpoint SHA256 |
|---|---|
| PPO PI7D | `a5bc3c2ed9032dfa8b11e41f1ea9e0d94eef1a2287a5b2611ef829cf265e6953` |
| Original BC | `7d7cf248c1fc672ec859bb906f4e8e0f442fbee9c6daea8fec54d4eee64c772c` |
| Yaw-Augmented BC | `d29b93555ef46f231228c23345761f76b5758f6451648375fcc2b7e4501f03ce` |

三模型checkpoint与parameter hash前后一致，eval mode、requires_grad=False。完整parameter hashes见报告`frozen`。Canonical world-model checkpoint没有使用或改变。没有新模型提交。

PI7D历史训练report没有保存当时checkpoint SHA。本轮验证了现存ZIP的seed、network、training-step数，并固定新SHA；旧success count和mean final distance精确复现。这不等同于补造历史checkpoint身份hash或宣称历史所有raw trajectory逐元素一致。

## 3. Target与实验预算

固定seed2026101501生成200target，reset seeds950150000–950150199；沿用原任务独立uniform x/y[-2,2]m、z[.7,1.5]m。新集合与现存BC Train/Val/Test、benchmark/holdout、adaptation、yaw-final、external-final的700条target记录按坐标/ID/reset-seed严格检查不重叠。来源文件hash、target、initial-state hash均入manifest。

历史PPO完整训练target流没有保存，**无法证明与全部PPO训练target完全隔离**。连续分布的新seed/不同已知目标减少已知复用，不替代严格全训练隔离证明。

正式800个新target闭环episode；另200个旧benchmark/holdout仅用于PI-PPO复现；四次first-target exact technical replay不进入成功率分母。1000正式unique记录、1004实际实验flight executions。单worker、按controller顺序只保留一个模型。Resume完整复核缓存，新增flight=0。单元测试中的小型模拟fixture不算新增benchmark或独立样本。

## 4. 新nominal主结果

| Controller | Success /200 | Timeout | Physical failure | Mean final error m | Mean final speed m/s | Successful time s | Mean peak speed m/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Scripted | 200 | 0 | 0 | .069621 | .064108 | 5.4632 | .754371 |
| Original BC | 200 | 0 | 0 | .064944 | .067654 | 5.7060 | .747857 |
| Yaw-Augmented BC | 200 | 0 | 0 | .064173 | .070202 | 5.5780 | .747726 |
| PPO PI7D seed0 | 165 | 35 | 0 | .100744 | .088151 | 8.7738 (n=165) | .616836 |

Success Wilson95%区间：BC/Scripted98.115–100%；PPO76.636–87.139%。这是固定target集合的描述性不确定性；零失败不等于总体风险为零。报告中的target-metric SD不是训练seed SD。

Near-target实际speed（全部pre-action distance<.10m样本，非终点，sample-weighted）：Scripted.188058、Original.187323、Yaw.182048、PPO.201468m/s。Near-target平均超过.15阈值并不矛盾：包含尚未停稳的进入/穿过阶段，而success需要终止时连续5步达标。

Normalized command-element |a|>=.95比例全部0。这里只测Policy command saturation，**不是**allocator saturation或PI integral saturation。

Mean25Hz轨迹chord length：Scripted1.903747、Original1.932721、Yaw1.889307、PPO2.339865m（后者包含timeout轨迹，不能只用此均值作完成任务的效率比较）。

### Paired comparison

同target、同initial snapshot；alternative−baseline，时间/路径的负值较好：

| 对照 | Success差异 | 共同成功数 | Paired mean time差 s | Paired successful path差 m |
|---|---:|---:|---:|---:|
| Original BC − Scripted | 0pp | 200 | +.2428 | +.028973 |
| Yaw BC − Scripted | 0pp | 200 | +.1148 | −.014440 |
| Yaw BC − Original BC | 0pp | 200 | −.1280 | −.043413 |
| PPO − Original BC | −17.5pp | 165 | +3.292848 | +.264823 |
| PPO − Yaw BC | −17.5pp | 165 | +3.394182 | +.301698 |

PPO与两BC各35个“BC成功/PPO失败”discordant pairs，0个相反方向。Paired completion只统计双方都成功，timeout不记15s“完成时间”。PPO在该paired subset没有表现出更快或更短路径优势；较低峰值速度是观测到的行为，不等于整体性能更好。

## 5. 历史复现与失败分析

PI7D旧benchmark：74/100，本轮74/100；mean final error .10751775177629046m，与旧report相等。旧holdout82/100，本轮82/100；mean final error .09892600587780538m，也相等。原训练checkpoint预算的评估结果可复现，不是95.6%旧P环五seed的重现证明。

新PPO35个timeout：19曾进入distance<.10m；10曾同时满足position+speed但没有连续5步；5个episode的最大success streak恰为4；15个出现当前既有crossing诊断。Timeout平均最终distance.190992m、speed.065723m/s。

**Confirmed**：没有物理失败或command saturation；残余位置偏差/未在15s内持续同时达标是实测失败表型。一部分曾经进入目标后再次离开，不能解释为所有episode平移爆炸；最后速度偏低也不能称完成任务。

**Supported hypothesis**：7D隐藏PI/姿态状态、有限100k训练预算与RewardV2的progress/near-speed目标可能限制停稳精度。10D历史高分仅是额外信息条件参考，不能据此把原因唯一归给PI隐藏状态。

**Unresolved**：不能用这一次确定性seed0 evaluation分离训练收敛不足、POMDP信息不足、reward/task alignment或策略分布训练/确定性执行差异。没有进行随机动作评估、调参或控制器干预。

## 6. PPO训练源码审计与后续判断

[PPO_FROM_SOURCE.md](PPO_FROM_SOURCE.md) 给出actor/critic/Gaussian采样、真实V2 reward、GAE/lambda return、clipped objective、entropy/value loss、Adam步骤和实际Tensor形状，全部对应仓库/SB3函数。BC是专家监督；PPO是环境奖励策略梯度，不混淆方法。

没有确认action/observation兼容性bug或明显错误的优化公式。现有baseline存在实验设计局限：单PI7Dseed、训练target流缺失、没有独立val用于终点checkpoint选择；历史95.6%不适用当前PI。100k后仍未完全成功，不能据此宣称PPO方法本身不适合任务。

**是否重训**：建立本轮公平baseline不需要重训，已完成；若下一阶段要作PPO算法结论，应先完成独立seed和完整数据身份的预注册设计。不要为了追上BC而即刻调reward/PI/网络。

唯一建议：预注册一个**7D PI-PPO独立seed复现实验**，优先固定obs/action/PI/physics/reward、训练交互预算、Gaussian初始化/deterministic inference、独立val/test target manifest、checkpoint selection和seed规则，完整记录训练目标流；本轮不执行。

## 7. 验证、资源与产物

15个本轮tests与23个相关非训练tests，共38个通过。实际SB3输出shape、GAE公式、V2 reward、10D拒绝、teacher-independent inference、checkpoint freeze、exact snapshot、atomic缓存、duplicate/failure/timeout/paired censoring和memory guard都有覆盖。没有运行裸全项目discovery：其部分fixtures会训练模型或构建8个环境，不符合本轮授权。

监控simulation/resume/analysis/validation phases；1 MuJoCo worker，无OOM、无swap调整，最终采样process-tree RSS峰值1.444GiB（validation），HWM汇总最大1.572GiB（历史高水位之和，不是同时RSS）。simulation RSS1.091GiB，resume检查RSS1.161GiB。系统保留至少1.5GiB；4GiB phase cap、2GiB早停余量。

所有1000正式record/NPZ身份hash、400initial-state identities、四组exact repeat逐元素检查；resume未重复或遗漏。图按第一个PPO success/timeout索引选轨迹，不挑选极端效果。

独立只读review没有Critical/Important问题。一个Minor保留并披露：冻结acquisition metadata的`training_source`字符串误写`train_ppo_pi_lowstd.py::train`，实际入口为`run_training`，PPO_FROM_SOURCE.md已使用正确函数。原字段保持不变以保留cache身份；不影响checkpoint、训练预算或数值结果，后续provenance schema再修正。

提交3个分析/评估source、测试、JSONreport/manifest、6幅必要图、本文、PPO源码教学文档、EXPERIMENT_LOG。RawNPZ/逐episodeJSON/resource logs/process ledger留在ignored `mujoco/reports/uav_ppo_bc_nominal_comparison_parts/`。历史checkpoint、untracked和原报告全部保留。

### 复现命令（只评估，不训练）

```bash
.venv/bin/python mujoco/rl/uav_bc_safety.py --directory mujoco/reports/uav_ppo_bc_nominal_comparison_parts --phase simulation -- env PYTHONPATH=mujoco/rl .venv/bin/python mujoco/rl/uav_ppo_bc_nominal_run.py
.venv/bin/python mujoco/rl/uav_bc_safety.py --directory mujoco/reports/uav_ppo_bc_nominal_comparison_parts --phase analysis -- env PYTHONPATH=mujoco/rl .venv/bin/python mujoco/rl/uav_ppo_bc_nominal_analysis.py
env PYTHONPATH=mujoco/rl .venv/bin/python -m unittest test_uav_ppo_bc_nominal -v
```

在相同源码/runtime/model身份下cache可恢复；身份变化拒绝复用。不要把不同环境下缺失raw cache的summary当作轨迹已完成。完成commit/push后停止，不启动PPO/SAC或下一实验。
