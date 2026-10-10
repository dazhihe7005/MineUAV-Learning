# Fixed-Protocol 7D PPO Multi-Seed Replication

日期：2026-10-10。Branch：`feat/uav-ppo-multiseed-replication`。
Base：`77ce81a99cc6306167a91a9fd4f804417433f655`。

## 预注册与可核实历史协议

本轮只训练五个新的PI–7D PPO，seed0/1/2/3/4，均从随机网络初始化开始。历史seed0 checkpoint、两BC及canonical world model不变。新模型使用独立实验文件名，旧模型不覆盖。

预注册：`mujoco/reports/uav_ppo_multiseed_preregistration.json`；配置SHA256：`e9c1fbb99bdcaad6fed9cf8d837777f7f43fa46863675c120e7a0e101d4aebf3`；manifestSHA256：`14be61bea7c191eef5259273c0e88f47b0523e9ef624f49d5f9293b1cc2d38df`。首次optimizer step之前已写入，之后不改训练协议或目标集合。

使用原 `train_ppo_waypoint.make_ppo` 工厂和 `MineUAVPIEnv(reward_version='v2',target_distribution='full')`。

| 项目 | 锁定值 |
|---|---|
| Observation | 7D：world目标位置误差xyz(m)、world速度xyz(m/s)、wrapped yaw error(rad) |
| Action | 4D [-1,1] world velocity xyz/yaw-rate；尺度[1.5,1.5,1]m/s及1rad/s |
| Actor / critic | 各7→64 Tanh→64 Tanh，actor输出4Gaussian均值、critic输出1value；4global log_std；9673参数 |
| Exploration | 初始log_std=-2，非squashed Gaussian；训练执行环境clip，评估deterministic clipped mean |
| Adam | lr=.0003，eps=1e-5，max_grad_norm=.5 |
| PPO | gamma=.99，GAE lambda=.95，clip=.2，entropy coefficient=0，value coefficient=.5 |
| Rollout / update | 8逻辑env×256steps=2048samples；batch256，n_epochs10；normalize_advantage=True |
| 预算 | requested100000；49完整rollout，actual100352transitions；490优化epochs，3920minibatch optimizer steps/seed |
| 其他 | clip_range_vf=None、target_kl=None、use_sde=False；无VecNormalize；CPU、Torch1thread、torch.use_deterministic_algorithms(True)（已锁定训练source） |
| Selection | 唯一最终预算checkpoint；Val/Test不选best，不延长训练 |
| Timing/task | 500Hz physics、100HzPI、25Hzpolicy；15s；distance<.10m且speed<.15m/s连续5policy步 |

八个逻辑env历史上也是同步`DummyVecEnv`，一个进程内顺序执行，**不是8个MuJoCo workers**。本轮5seed顺序训练，保留原rollout结构。低层PI/姿态/allocator/physics/termination/reward源代码身份锁定。

Reward V2保持不变：`10(d_prev-d) - .005||a||² - .5 b(d)||v||² +10*success -10*physical_failure`，`b(d)=clip((.5-d)/.4,0,1)`。没有新增distance shaping，也没有reward搜索。`entropy_loss`是−H；连续Gaussian differential entropy可以为负，不能把其符号直接当作错误。`explained_variance`使用rollout values和GAE returns，非未来Test拟合指标。

历史可核实网络、接口、PI、reward、预算、optimizer及SB3配置均保持一致。旧报告没有当时checkpointSHA、完整Train target流或完整运行时/BLAS线程配置，不能声称重建了原来逐元素相同的随机训练轨迹。新seed0首个训练目标与旧报告一致，首次update已有约1e-8量级metric差异；后续性能差异不能唯一归因于此浮点差异。

## 数据隔离与统计单位

Train维持原continuous sampler：独立uniform x/y[-2,2]m,z[.7,1.5]m。被动`AuditedPIEnv`记录每次真实reset的rank、episode_index、原始target coordinates和coordinate hash。若实际Train目标距Val/Final任何目标<1e-8m，停止，不拒绝重采样。该wrapper的reset和step已与原env逐元素比较。

固定seed2026101601生成Val40和Final200；reset seeds分别950160000–950160039、950170000–950170199。两集合互不重叠，并排除现存BC/adaptation/yaw/external和上一轮nominal的900条已知target身份/坐标/reset seed。实际Train target审计在训练完成后再次检查；必要目标身份aggregate单独提交，不提交raw轨迹。

继承历史`training_seed+rank`环境seed规则。相邻训练seed共享7/8个底层target RNG seed，但episode终止和策略不同会改变访问进度。网络初始化、action sampling及minibatch随机seed不同；五条独立训练run不是五个完全不重叠环境数据流，也不是upstream/end-to-end预训练复制。

Val只在20480/51200/100352做诊断，每次40targets，共600Val episodes。独立环境推理前后保存/恢复Python、NumPy和Torch RNG以及模型mode，不能消耗训练随机流。Final只有训练完成后执行；最终checkpoint无论Val回退与否都保留。

Final共有1600formal episodes：5PPO×200 + 3冻结baseline×200。8次first-target逐元素technical repeat仅用于确定性验证，不进入成功率分母。所有controller逐target恢复相同完整physical/controller/task/RNG snapshot，不仅复制7D观测。

跨seedSD采用ddof=1；统计单位是5training runs，不是200targets。报告exploratory t95 interval(df4)，共享固定targets/部分环境流、小n，不能夸大总体显著性。每policy Wilson95区间是target-cohort描述，零失败不等于总体风险零。Paired time/path仅双方成功targets；timeout完成时间N/A。

## 结果

五个seed均完整训练，全部初始parameter hashes不同，均不是历史checkpoint fine-tune。每个finalZIP的seed、100352timesteps、490优化epochs及所有Adam参数的3920step计数均从reload实际核对。总501760训练transitions、19600optimizer steps。所有seed仅1次完整attempt。

| Controller | Success /200 | Timeout | Physical failure | Mean final distance m | Mean final speed m/s | Successful time s |
|---|---:|---:|---:|---:|---:|---:|
| Scripted | 200 | 0 | 0 | .071174 | .063058 | 5.5910 |
| Original BC | 200 | 0 | 0 | .066400 | .066210 | 5.8488 |
| Yaw BC | 200 | 0 | 0 | .064493 | .069536 | 5.7178 |
| PPO seed0 | 27 | 173 | 0 | .593735 | .122649 | 12.8385 (n27) |
| PPO seed1 | 77 | 123 | 0 | .085757 | .222221 | 8.4410 (n77) |
| PPO seed2 | 33 | 167 | 0 | .330345 | .190817 | 9.8485 (n33) |
| PPO seed3 | 121 | 79 | 0 | .121501 | .146019 | 9.2714 (n121) |
| PPO seed4 | 45 | 155 | 0 | .368440 | .158346 | 8.1733 (n45) |

PPO成功率分别13.5%、38.5%、16.5%、60.5%、22.5%。五seed均值30.3%，sampleSD19.4474个百分点，median22.5%，范围13.5–60.5%。以5training runs为单位的exploratory t95 interval为6.153–54.447%。不能把1000个PPO-target组合视为1000个独立训练seed；当前均值没有复现约82.5%历史checkpoint表现，也不是历史optimizer trajectory的精确重放。

| PPO seed | Mean25Hz path m（所有结果） | Mean sampled peak speed m/s | Near-target actual speed m/s | Command saturation |
|---|---:|---:|---:|---:|
| 0 | 3.006432 | .581733 | .202124 | 0 |
| 1 | 2.875041 | .593288 | .223677 | 0 |
| 2 | 3.529028 | .637577 | .298215 | 0 |
| 3 | 2.563465 | .502457 | .204813 | 0 |
| 4 | 3.229708 | .582930 | .239820 | 0 |

轨迹长度定义为25Hz物理position chord之和；peak也是25Hz采样peak，不是假称500Hz连续峰值。Near-target取pre-action距离<.10m的所有样本，包括尚未停稳的进入过程，均值可以超过success速度阈值。三个baseline command saturation也全部0；这是policy command元素|a|>=.95，不是allocator或PI saturation。

### Paired comparison

所有5seed在success上都低于两BC（5/5，反向win0/5）。每seed“BC成功/PPO失败”discordance为173/123/167/79/155，反向discordance均0。Success差−86.5/−61.5/−83.5/−39.5/−77.5个百分点，跨seed平均−69.7pp。

| Seed | 共同成功数 | PPO−Original time s | PPO−Original successful path m | PPO−Yaw time s | PPO−Yaw successful path m |
|---|---:|---:|---:|---:|---:|
| 0 | 27 | +8.087407 | +.679829 | +7.551111 | +.589495 |
| 1 | 77 | +3.354805 | +.195181 | +3.476883 | +.229647 |
| 2 | 33 | +4.830303 | +.430164 | +5.075152 | +.496069 |
| 3 | 121 | +3.976529 | +.270910 | +4.119669 | +.311675 |
| 4 | 45 | +3.619556 | +.244865 | +3.538667 | +.232941 |

这里只比较同目标且双方都完成的配对，不能用失败episode低速/短路径声称效率好。本固定7D PI/100k协议下BC更可靠，且共同成功时更快、更短路径；不能扩大为PPO算法普遍弱于BC或样本效率结论。

### 失败表型

合计697timeout、0physical failure。曾进入距离阈值的失败数各21/99/48/36/27（合计231）；曾同时达position/speed但未连续5步成功的各7/14/4/6/5（合计36）。最大streak恰4的共10。不是所有失败都集中“进入目标之后停稳”。

| Seed | 失败终点：位置过、速度不过 | 速度过、位置不过 | 二者均不过 |
|---|---:|---:|---:|
| 0 | 0 | 106 | 67 |
| 1 | 76 | 1 | 46 |
| 2 | 2 | 44 | 121 |
| 3 | 4 | 16 | 56 |
| 4 | 0 | 70 | 84 |

少数失败末帧瞬间joint达标但dwell不足，所以上表未强行把所有failure塞入三类。Seed0主要残余位置/进度不足，seed1很多位于目标区域但速度仍过大，seed2/3还有位置与速度同时不达标。源代码定义time_limit未改；没有证据把共同失败唯一归因于PI、reward或value。

### 学习过程

| Seed | Val20k /40 | Val50k /40 | Valfinal /40 | Final value loss | Final explained variance | Final approx KL | Final clip fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 9 | 6 | .227843 | .970211 | .006659 | .075049 |
| 1 | 1 | 24 | 17 | 1.270617 | .814656 | .007894 | .086523 |
| 2 | 1 | 14 | 8 | 1.015111 | .808159 | .009578 | .097705 |
| 3 | 1 | 9 | 25 | .919556 | .770088 | .007060 | .064990 |
| 4 | 0 | 9 | 10 | .269363 | .797542 | .005284 | .047705 |

原始49update×5 history包含episode return、stochastic Train recent-success、policy-gradient loss、value loss、entropy loss/actual differential entropy、approx KL、clip fraction、explained variance。Finite优化指标无NaN/Inf。Seed0/1/2出现50k→final Val回退，seed3/4没有；不能据三次诊断确定精确collapse时刻，也没有挑较好的中间checkpoint。

**Confirmed**：固定预算成功率差异大；所有新seed在新Final集合均低于BC。Seed0 value EV=.970但success13.5%，较高rollout value-fit不保证确定性task success。没有command saturation或physical failure。

**Supported hypothesis**：预算内训练/representation/策略执行稳定性及reward–task停稳对齐值得检查。Train stochastic success与Final deterministic success是不同目标/随机性条件，差异不能直接归因于deterministic inference。

**Unresolved**：不能从单一loss/EV/KL曲线证明训练不足、奖励缺陷、7D隐藏状态或value拟合是唯一根因。历史单seed82.5%仍是可复现checkpoint结果；当前CPU1thread/固定Val诊断协议的新五seed并不证明旧模型身份错误。历史未记录完整thread/BLAS配置，初次update微差与最终大差之间未做因果实验。

唯一下一建议：预注册一次**离线RewardV2与任务停稳目标的对齐审计**，检查现有完整轨迹/returns中的到达、制动、残余误差与奖励关系；不自动调reward、不延长训练、不运行SAC。当前证据不要求立刻引入新算法作为补救。

### Checkpoint SHA256

| Seed | `uav_ppo_fixed_protocol_seed{seed}.zip` SHA256 |
|---|---|
| 0 | `9a1e211611436d058a9fafe218c8f0e965572dfe6389c4dff869a9dcedcfd49b` |
| 1 | `96d66efd3d1a943d6320d1013abb20cb429cffb1f992ecbb24e94775e91a9a33` |
| 2 | `61da0c0f000f544f6963254a1f4c3aa086161592988900ff0d4bfd0a3d2e737d` |
| 3 | `0f0bddee3c711b925b7a9ac7336d23ca47afb250196773a3bf6cda40851c8cd6` |
| 4 | `80224e40f26faf1802388d9f9f264e5ec120bb40cfdc9efe91c77f921d6a5125` |

每模型initial/final parameter SHA在JSONreport的training中；不同initial已核实。OriginalBC/YawBC/旧PI-PPO/canonical world-model的文件hash与启动前一致；BC/PPO eval parameters before/after也相同。

## 完整性与复现

模型按完整rollout更新后保存，最终reload核对seed/100352transitions/490epochs及parameter hash。每seed独立目录；中断attempt保留，未完成seed从相同seed重启，不假称精确恢复optimizer中间状态；只有已验证完整seed可跳过。正式evaluation采用atomic JSON/NPZ、unique task IDs、trace/record hashes和逐episode恢复。没有把未完成cache标记complete。

内存：单worker，8同步逻辑env，seed顺序；existing safety supervisor监控进程树RSS、系统可用物理内存和kernel oom计数。4GiB phase cap，系统必要reserve1.5GiB，available<2GiB早停；不修改swap。

实测process-tree sampled RSS peak **2.088GiB**（seed0），全部资源phase exit0/oom delta0。五seed耗时约145–149s/seed（含三次Val），正式Final+technical repeats450s。无OOM、无中断；Train visits数286/316/286/293/293，总1474记录（包含各rank终点尚未完成episode的当前target），与固定Val/Final实际坐标重叠0。训练预算按完整rollouts计，并不要求最后8个训练episode都终止。

58项相关unittest全部通过，无skip：16项新增+38项前阶段相关+4项原PPO训练构造/保护测试。本轮测试不执行额外PPO训练，只有模型构造/reload及小型physics fixtures。全量1600正式record/raw hash、指标重算、初始snapshot配对、8exact-array repeats核实；Final缓存恢复新增flight0，5training caches恢复新增训练step0。Val600+Final1600+technical8=2208评估episode executions；测试fixtures单列，不作为正式统计样本。

没有裸全项目discovery：它包含与本轮无关的旧实验训练jobs，不能为测试启动额外学习阶段。原始数据齐全；完整cache可恢复，无重复、遗漏或不完整记录冒充complete。

独立只读审查：0 Critical、0 Important。审查者独立复原1474个实际Train targets及Val/Final生成，核对1600条record/raw hash和指标、15组paired比较、五个ZIP身份/optimizer预算及八张图。次要项暂缓：预注册source hash列表没有覆盖复用的`uav_ppo_bc_nominal.py`、`uav_ppo_bc_nominal_run.py`、`uav_ppo_bc_nominal_analysis.py`；这些历史tracked文件本轮均未改变，当前结果已独立重算。保留本轮冻结manifest，后续预注册应补齐依赖覆盖，不以改写manifest掩盖此限制。

### 复现

```bash
# 验证已锁定manifest；不要对既有结果重新生成不同配置。
env PYTHONPATH=mujoco/rl .venv/bin/python -c 'from uav_ppo_multiseed import read_manifest; print(read_manifest()["sha256"])'
# seed逐一执行，完成cache会验证并跳过；未完成attempt重启而非无证据续训。
.venv/bin/python mujoco/rl/uav_bc_safety.py --directory mujoco/reports/uav_ppo_multiseed_replication_parts --phase train_seed0 -- .venv/bin/python mujoco/rl/uav_ppo_multiseed_train.py --seed 0
# 同样顺序执行seed1..4，再评估；不要并发启动五个训练。
.venv/bin/python mujoco/rl/uav_bc_safety.py --directory mujoco/reports/uav_ppo_multiseed_replication_parts --phase evaluation -- .venv/bin/python mujoco/rl/uav_ppo_multiseed_eval.py
env PYTHONPATH=mujoco/rl .venv/bin/python mujoco/rl/uav_ppo_multiseed_analysis.py
env PYTHONPATH=mujoco/rl .venv/bin/python -m unittest test_uav_ppo_multiseed -v
```

Raw轨迹、actual reset/episode logs、intermediateZIP、resource logs、process ledger保留在ignored parts目录。Git提交五个必要finalZIP、source/tests、预注册及target-audit manifests、aggregate report、figures、文档和EXPERIMENT_LOG。历史untracked/checkpoints保留。

本轮结束即停止；不自动reward tuning、SAC或其他训练。
