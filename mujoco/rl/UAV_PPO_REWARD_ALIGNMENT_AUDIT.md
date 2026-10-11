# Offline RewardV2 and Task-Success Alignment Audit

日期：2026-10-11。Branch：`feat/uav-ppo-reward-alignment-audit`。
Base：`b36d7539dbf53fa3f1c17a510ac3a12f8b3bcc62`。

结论：实际五 seed 使用 RewardV2。它既奖励距离进展，也惩罚近目标速度，并在任务成功时奖励 +10；不能说它“没有制动奖励”。但它没有超时惩罚，训练将 time-limit 截断作为可 bootstrap 的状态，任务评估却将超时计为失败。这些是可核实的结构特征，不是 PPO 失败的唯一因果解释。

最重要的数据限制：全部 1,000 条 Final 轨迹均未保存 reward/breakdown。无法验证同一条轨迹的重算奖励与原始奖励日志一致，因此本轮不重算 Final 奖励；累计奖励、折扣回报、分量、超时前奖励趋势、“高奖励 Final 失败”全部标为 **not observable**。训练日志的真实奖励补充证据单独列出，不能替代 Final。

本轮新 episode = 0、MuJoCo 积分 step = 0、optimizer step = 0。未改变 reward、PI、physics、训练协议或任何模型。

## 1. 源码与真实奖励

审查的是当前仓库及本机安装的 Stable-Baselines3 2.9.0，而非用一般理论推测历史行为。相关实现身份均在 JSON 的 `identity.input_files_sha256` 中保存并验证分析前后相同。

| 链路 | 实际文件与函数 | 核实内容 |
|---|---|---|
| 训练工厂 | `uav_ppo_multiseed.py::AuditedPIEnv.__init__/step` | 明确 `reward_version='v2'`，被动累计真实 reward components |
| 环境继承 | `ppo_pi_env.py::_PIConfiguredMixin/MineUAVPIEnv` | 配置固定 PI，不重写 reward 公式 |
| 奖励/终止 | `mine_uav_env.py::MineUAVEnv._compute_reward/step` | reward components、success streak、terminated/truncated |
| PPO 配置 | `train_ppo_waypoint.py::make_ppo`、五个 ZIP 的 `data` | gamma=.99、GAE lambda=.95、seed、100352 timesteps |
| 原始采集 | `uav_bc_robustness.py::rollout` | `obs, _, terminated, truncated, info=env.step(action)`；丢弃奖励 |
| 环境 return | SB3 `common/monitor.py::Monitor.step` | 原始环境奖励之和，CSV 保留六位小数，发生在 bootstrap 之前 |
| 截断 | SB3 `common/vec_env/dummy_vec_env.py::DummyVecEnv.step_wait` | 保留 reset 前 `terminal_observation` 和截断标志 |
| Bootstrap | SB3 `common/on_policy_algorithm.py::OnPolicyAlgorithm.collect_rollouts` | 对 timeout 增加 gamma × terminal Value |
| GAE | SB3 `common/buffers.py::RolloutBuffer.compute_returns_and_advantage` | done mask、GAE、returns=A+V |

SB3 文件位于 `.venv/lib/python3.10/site-packages/stable_baselines3/`。源码说明不是重新执行历史训练。

令 d 为 world-frame 目标距离（m），v 为 world-frame 线速度（m/s），a 为实际送入环境的四维 normalized action。

| 分量 | 每个 25 Hz policy transition 的实际公式 |
|---|---|
| progress | `10 * (d_previous - d_current)` |
| action | `-0.005 * sum(a**2)`，包括 yaw-rate 维 |
| brake | `-0.5 * b(d_current) * ||v_current||²` |
| braking weight | `b(d)=clip((0.5-d)/0.4,0,1)`：d≥0.5 无效，d≤0.1 完全启用 |
| success | 成功终止时 +10；不是每个 near-target step 都 +10 |
| failure | 物理/非有限状态失败时 −10；不包括 timeout |
| total | 上述五项之和 |

V2 没有独立绝对距离惩罚（V3 才有）、yaw 专属奖励、yaw 成功门槛、连续停稳中间奖励或 timeout penalty。没有 reward clipping、VecNormalize 或 reward normalization，也没有乘 dt。braking weight 的 clip 不是总 reward clipping；PPO 的 likelihood-ratio clip/advantage normalization 也不是 reward clipping。

动作语义：a∈[-1,1]⁴，映射为 `[1.5*a_x, 1.5*a_y, 1.0*a_z]` m/s 和 `1.0*a_yaw` rad/s。SB3 将非 squashed Gaussian action 裁剪到 Box bounds 后送入环境；`velocity_command_controller.py::map_normalized_action` 校验合法范围，而不是静默裁剪任意非法输入。动作惩罚使用 normalized action，不使用物理速度命令或 Train action std。观测为 raw 7D `[target_error_xyz(m), world_velocity_xyz(m/s), yaw_error(rad)]`，没有观测归一化或额外 PI state。

physics/control/policy = 500/100/25 Hz。成功条件未变：`distance<0.10m AND speed<0.15m/s` 连续 5 个 policy steps。成功和物理失败 `terminated=True`；其余在 375 steps/15s 时 `truncated=True`。超时不是物理失败，不能误套 −10。

## 2. Discount 与 time-limit bootstrap

采用 gamma=.99，每 0.04s discount 一次，半衰时间约 2.758703s。假设任务在 5/10/15s 成功，单独终止 +10 在起点的折扣贡献分别是 2.875836/0.818773/0.233111。这只是公式诊断，不是实测回报；不证明 +10 一定太小或会造成训练失败。

未折扣 progress 的累计值严格 telescoping：

```text
sum progress = 10 * (d_0 - d_T)
```

它受初始目标距离影响，不能直接用不同难度 episode 的 raw return 评价控制质量。折扣 progress 为：

```text
10 * [d_0 - (1-gamma)*sum(t=1..T-1) gamma^(t-1)*d_t
      - gamma^(T-1)*d_T]
```

因此它仍隐含偏好较早接近目标；“没有独立距离项”不等于“没有距离激励”。

实际 SB3 路径：DummyVecEnv 设置 `done=terminated OR truncated`，保存 reset 前 terminal observation，且 `TimeLimit.truncated=truncated AND NOT terminated`。在收集 rollout 时，仅对于该 time-limit 标志，将 `gamma*V(terminal_observation)` 加入 float32 buffer reward。Monitor 已先记录环境原始 reward，故 Monitor return 不包含这项。

GAE 使用 `delta=r_buffer+gamma*next_value*nonterminal-V`，`A=delta+gamma*lambda*nonterminal*A_next`，`return=A+V`。跨 episode 的 mask 防止将新 reset 的 value 串入旧 episode；timeout 的末状态 value 已通过上述 reward correction 注入。真实 success/failure 不进行该 timeout correction。纯数组测试验证：环境 reward0、terminal value2、gamma.99 时，单步终止 buffer return=1.98，不是后续 reset value。

Confirmed：训练截断具有 critic continuation 语义，评估 timeout 是失败。Unresolved：历史 terminal values、advantages、bootstrap 的实际幅度均未保存，无法确认它是否导致过高估值或失败；不能把标准 SB3 处理直接称为 bug。

## 3. 原始数据身份与可观测性

完整重用前阶段固定 200 targets × 5 PPO seeds；1000 个 task ID、record seal、manifest、target/初始 snapshot 身份、NPZ SHA、step/action/previous-action 索引、success_streak、最终距离和原始 JSON/aggregate 一致性均验证。所有旧输入（含模型、源码、日志）共 2108 个文件逐一保存 SHA 并复查未改变。

Final NPZ 含：float32 observations(n+1,7)、float64 positions(n+1,3)、float32 actions/previous_actions(n,4)、step_index、success_streak、physical_state_valid。缺 reward、reward_breakdown、per-step Value/Advantage；内部 float64 qvel 也未保存。位置和动作可直接分析，速度由实际保存的 float32 observation 得到，success gate 以环境实际记录的 streak 为准，不用舍入后的速度重造严格门槛。

| Final 指标 | 状态 |
|---|---|
| 状态、动作、实际成功/超时、距离/速度/停稳 | 可观测并重算验证 |
| 每步/累计/折扣 reward、各分量、reward 与 success 的相关性 | not observable |
| 高 reward 的 Final 失败、超时前 reward trend | not observable |
| historical bootstrap value/advantage | not observable |

五个原 checkpoint 的 SHA256（参数 SHA 及所有输入身份在 JSON）：

| Seed | ZIP SHA256 |
|---|---|
| 0 | `9a1e211611436d058a9fafe218c8f0e965572dfe6389c4dff869a9dcedcfd49b` |
| 1 | `96d66efd3d1a943d6320d1013abb20cb429cffb1f992ecbb24e94775e91a9a33` |
| 2 | `61da0c0f000f544f6963254a1f4c3aa086161592988900ff0d4bfd0a3d2e737d` |
| 3 | `0f0bddee3c711b925b7a9ac7336d23ca47afb250196773a3bf6cda40851c8cd6` |
| 4 | `80224e40f26faf1802388d9f9f264e5ec120bb40cfdc9efe91c77f921d6a5125` |

ZIP 的完整 SHA 与之前 sealed completion 相同，证明文件及其序列化参数未变化；本轮不加载策略执行 episode。JSON parameter hashes 引用并核对之前的参数身份，不伪称新训练。

## 4. Final 状态与近目标制动

| Seed | 成功/200 | 超时 | 失败曾 d<.1 | 失败曾 joint gate | 失败 max streak4 | 失败均值最终距离/速度 |
|---|---:|---:|---:|---:|---:|---|
| 0 | 27 | 173 | 21 | 7 | 3 | .672895m / .125962m/s |
| 1 | 77 | 123 | 99 | 14 | 3 | .094308m / .299411m/s |
| 2 | 33 | 167 | 48 | 4 | 1 | .380641m / .209576m/s |
| 3 | 121 | 79 | 36 | 6 | 3 | .189047m / .242561m/s |
| 4 | 45 | 155 | 27 | 5 | 0 | .452344m / .177875m/s |

303 成功、697 超时、0 物理失败；各失败全为 15s。失败中 231 曾进入距离门槛、36 曾同时达到距离/速度但不满 5 步、10 曾达 streak4。其余 466 未进入距离门槛；不能把全部失败归为“差一个停稳 step”。

对于有 pre-action d<.10m 样本的失败 episode，先算该 episode 平均速度再跨 episode 等权平均：seed0–4 为 .271288/.286856/.368923/.268429/.301625m/s，实际 N=21/95/48/36/27。合并 N=227，均值 .301603m/s；成功 N=303，.145803m/s。另有 4 个 seed1 timeout 只在最后一帧进距离门槛，没有对应近目标 pre-action 样本；保持缺失，不填零。不使用任意最后一帧替代持续停稳。

对 timeout 的固定 12–15s 窗口，平均距离减少量 seed0–4 为 −.137996/+.062523/−.058431/−.040080/−.074755m；负值代表距离扩大。此窗低速但仍在距离阈值外的样本比例，按 episode 等权平均为 62.36/33.25/45.03/36.91/40.10%。这是状态趋势，绝不是 reward trend。说明残余位置、较高近目标速度和未持续 joint gate 并存，种子间失效模式不同。

速度命令由实际四维 action 的前三维物理映射得到，与同一时刻 pre-action measured velocity 对齐；相反方向命令/追踪误差只是诊断，不证明 PI 或 policy 的唯一故障。本轮未读取不存在的内部 PI 日志。

## 5. 长度、难度、seed 控制

不把 pooled success/failure 均值当作因果对照。Final 成功平均 d0=1.333765m、平均长度9.278020s；失败 d0=1.686488m、长度15s。失败也能产生较大绝对距离进展，不能据此断言获得更高真实 reward。

JSON 为每个 seed 分别输出初始距离 `<1/1–2/>=2m` × duration `<=8/8–12/>12s` × outcome 分层；每格报告实际 N、observable N、mean/median/sample SD/quantiles，空格为 null。短成功窗口没有相应短失败，不能强行匹配。

另作共同 target、共同前 2s 的跨策略 outcome-discordant 描述性配对。只比同 target 且一成一败的两个 seed；不延长已终止 episode。当前最短成功2.08s，均具有完整2s prefix，censored=0。

| Seed pair | N | 失败−成功的2s距离进展均值(m) | 失败−成功的2s速度均值(m/s) |
|---|---:|---:|---:|
| 0–1 | 78 | −.073636 | −.031664 |
| 0–2 | 60 | +.093405 | +.049876 |
| 0–3 | 98 | +.038198 | +.028347 |
| 0–4 | 42 | +.074325 | +.046060 |
| 1–2 | 68 | +.045776 | +.032497 |
| 1–3 | 88 | +.051413 | +.035383 |
| 1–4 | 74 | +.077272 | +.047718 |
| 2–3 | 96 | +.085340 | +.057557 |
| 2–4 | 62 | −.002427 | +.007176 |
| 3–4 | 100 | +.039227 | +.033126 |

较快 early approach 并不保证最终成功，方向也不是所有 pair 一致。不同 seed 是不同 policy，不能把这些重叠 target/pair 当作额外独立训练 seed，更不能将此表称为 reward 干预。统计独立训练单位仍仅5seed。

## 6. 独立标注的真实 Train reward 补充

使用已存在五 seed 的 `training_episodes.json` 和 40 个 Monitor CSV，1434 个已完成 stochastic training episodes。逐 rank/episode index/target/长度/reason 匹配；component sum 与 Monitor return 在六位小数误差≤5.01e−7一致，progress 与独立 Monitor terminal distance 的 `10*(d0-dT)` 在1e−9内一致。没有用状态公式伪造奖励分量。

此数据来自训练过程中不断变化的 stochastic policy，与冻结 deterministic Final 不同。未记录 Train 每步 reward，因此也不能计算可靠 discounted Train return 或时间趋势。

| Seed | Train N(S/F) | 成功 return均值 | 失败 return均值 | 失败return≥本seed所有return Q75 |
|---|---|---:|---:|---:|
| 0 | 278(46/232) | 19.953485 | 10.088949 | 37 |
| 1 | 308(113/195) | 20.956201 | 9.710614 | 5 |
| 2 | 278(38/240) | 18.911856 | 8.951618 | 45 |
| 3 | 285(62/223) | 20.126822 | 10.000377 | 30 |
| 4 | 285(74/211) | 20.815432 | 10.417927 | 27 |

实际 Train 各 seed 成功 mean return 均高于失败，不能宣称 reward 总体完全不区分成功。分布仍重叠，共144条失败在各自 seed Q75以上；“high”是明确的描述性相对阈值，不是通用质量标准。

例如 seed4 实际最高失败 return24.070131：d0=2.656587m、dT=.183417m、progress24.731695、action−.153973、brake−.507590、success/failure项0。这是有原日志支持的高 Train-return timeout，不是 Final 对应飞行，也未满足位置阈值。各 seed最高失败return为21.537355/22.875279/23.698499/23.574347/24.070131。

| Seed | 成功mean progress/action/brake/success | 失败mean progress/action/brake/success |
|---|---|---|
| 0 | 11.427127/−.102858/−1.370784/+10 | 11.440450/−.149707/−1.201794/0 |
| 1 | 12.159124/−.091686/−1.111237/+10 | 12.574373/−.148043/−2.715715/0 |
| 2 | 10.264210/−.099669/−1.252685/+10 | 10.849807/−.153525/−1.744665/0 |
| 3 | 11.408491/−.096713/−1.184957/+10 | 11.069177/−.145253/−.923547/0 |
| 4 | 12.338548/−.102160/−1.420956/+10 | 11.902755/−.142254/−1.342575/0 |

所有 Train failure项0，因为这些失败是 time_limit。Raw Train return 与 d0 的 seed内 Pearson .455007/.299561/.441022/.462450/.524891；相关不代表 causal effect。JSON 同时报 `R-10*d0` 诊断量、per-step return、原始距离/长度分层；前者只移除已知初始距离偏移，既非新奖励，也不是完整难度调整。Train age/policy阶段仍然混杂。

训练末尾日志 rolling100 Train return=13.114870/12.709926/9.956405/15.161040/14.317924，对应 frozen Final success=13.5/38.5/16.5/60.5/22.5%。5seed描述性 Pearson=.600317、Spearman=.5，方向部分一致但非单调；n=5且 Train/Final条件不同，不做p值/统计显著性、不能称为 Final reward-success correlation。

## 7. 证据等级与停止结论

Confirmed：真实RewardV2及discount/bootstrap路径；有near-target braking和success bonus但没有timeout penalty；全部Final reward不可观测；Final失败异质、231曾进距离门槛、36曾joint gate；真实Train日志存在相对高return失败且初始距离混杂明显，成功meanreturn仍更高。

Supported hypothesis：早期进展、晚期成功discount和time-limit critic continuation的组合值得对齐检查；残余位置、制动与joint dwell不足与reward/task不完全对齐相容。但这个解释不能排除有限训练预算、seed敏感性、7D隐藏PI状态、策略分布或级联动力学等因素。

Unresolved：Final高reward失败、timeout奖励趋势、历史terminal V/Advantage幅度、改变reward是否改善策略，以及reward是否主要/唯一根因。没有足够证据支持本轮直接调reward，亦不支持自动开始SAC。

唯一下一建议：另行授权最小被动日志完整性验证，保存每步原始reward/breakdown、terminated/truncated以及训练末状态bootstrap/value，再讨论reward干预。本轮不实现、不运行此下一任务。

## 8. 复现、资源与交付

只需现有本地 raw/log 文件，不重新生成。运行：

```bash
env PYTHONPATH=mujoco/rl .venv/bin/python -m unittest test_uav_ppo_reward_alignment test_decision_fidelity_metrics -q
env PYTHONPATH=mujoco/rl .venv/bin/python mujoco/rl/uav_bc_safety.py --directory mujoco/reports/uav_ppo_reward_alignment_parts --phase audit -- .venv/bin/python mujoco/rl/uav_ppo_reward_alignment_analysis.py
```

26项相关纯离线测试（21新+5已有ranking），不构造/step MuJoCo。未运行裸全仓库test discovery，因为其中有训练和新episode fixtures，违反本轮边界。测试覆盖缺失reward不重造、动作/时序、target/record/duplicate身份、Monitor一致性、strata null、gamma索引、实际RolloutBuffer数组语义、JSON计数类型和2s配对/截断。

1个分析process，0个simulation worker，逐NPZ加载。使用已有4GiB phase cap、1.5GiB OS reserve、2GiB headroom/OOM提前中止、原子JSON/PNG写入；没有swap调整。当前各分析phase实际进程树采样RSS/HWM最高345591808 bytes（0.322 GiB），0OOM；最终运行精确self-HWM在报告`resource`，监督器phase记录留本地。第一次报告发布遇NumPy整数JSON序列化异常，经纯fixture RED→GREEN修复；另一次重复分析验证因native tuple与JSON list比较误报，改为统一JSON表示后所有非resource字段（包括输入/figure哈希）完全一致。失败phase保留，未产生残缺报告或遗漏episode。

7张科学图存于 `mujoco/reports/uav_ppo_reward_alignment_figures/`。S/F分别为成功/失败；箱线图隐藏outlier点仅为可读性，JSON保留全部episode和范围，并未删数据。source probe图是明确标注的静态公式示例，不是新增flight。

Git仅交付两个analysis source、tests、summary JSON、7张figure、本文和EXPERIMENT_LOG。既有raw trajectories/models/untracked和本轮资源/过程记录保持local-only，旧报告不改；commit/push后停止，不启动任何实验。
