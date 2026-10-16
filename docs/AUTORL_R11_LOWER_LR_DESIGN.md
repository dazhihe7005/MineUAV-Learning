# AUTORL-R11：解析先验的全局低学习率对照（事前设计）

## 问题、范围与对照

R10已推送21831fcc90d06756f431373643e75822b633eb7a，完整65536步但四项科学标准FAIL。短Validation从初态6成功降至32k/末态0成功；R10最初32次更新approx_kl均值0.07136825448833406、clip_fraction均值0.33740234375，明显大于R8随机初态。它支持“更新幅度破坏先验”假设，但不能排除探索、Critic或任务信用分配等原因。

本轮仅改变全局PPO学习率3e-4→3e-5；共享优化器同时影响Actor/Critic，不称Actor-only改进。相对固定R10对照，其余CNN、Reward、std.3可训练、目标课程、seed0、PPO128/64/2及全部控制/传感器配置不变。Critic也可能学得更慢，先验保留不等于学会避障。

## 新协议和身份

新schema11、ARL_R11_PRIOR_LOWER_LR_SEED0，65536环境步/3600s保守完整wall预留、CPU单worker；不加载任何旧checkpoint，不重跑R10。余额1475712步/57600s以真实账本核验，不初始化/退款。唯一预留后应1410176步/54000s；未预留前不能报告已消耗。RESTART_REQUIRED。

R10 summary c9fa8bcc9b14fe441903e1f733e7496393e39a36a72c0d2f8e907fc9778ba2ca，已发布报告0847525a939fee3eaf23a0cbf32511a7828aeaa7059084ed86ce81ce25ebc876作为固定对照。R9/R7/R5 proof继续核验，不修改旧版本。

12short .20m任务至32768后下次自然reset恢复原12目标。500/100/25/10Hz、24krays、7kg动力学/PI、NAV_V2_COLLISION_PREFLIGHT_V1、PPO gamma.99/GAE.95/clip.2/ent0/vf.5/maxgrad.5不变。新随机seed0赋解析均值及std.3，完整初态policy6312e99c87fd5a4dca066bd1ef64acd17d5ee44b563c152c59c133ee659d7f74。

## 学习率、Checkpoint与历史身份语义

不改原make_ppo、CheckpointManager和L6/L7协议。只在全新、0步、空optimizer、未reset网络上设置learning_rate=3e-5、重建常数schedule并设置所有optimizer组，发生在先验赋值和step0保存之前。每次训练更新前后与每次重载均核实际schedule/group学习率。所有实际checkpoint含真实新学习率；sidecar的L6/L7 hash只是base provenance，新execution_protocol绑定覆盖值，不声称R11沿用旧LR。

checkpoint0/32768/65536，完整rollout对应优化后保存；关闭Train再执行30新Validation。原step0六原任务评价复用R3 PRIOR同完整policy6312的评价，不复用optimizer/文件身份；0更新下deterministic预测不依赖LR，需离线测试验证。short step0是手写先验而非学习。Test0、Final Checkpoint，不按Val择模。

## 事前五项独立判据

1. 短任务候选：final至少4/6success、至多1collision，physical/sensor0。
2. 短先验保留：final短成功>=自身初态短成功、collision<=自身初态，physical/sensor0；若初态6则必须保住6，不能虚构严格提升。
3. 学习率收益：original final成功数>R10 final0且collision<=R10 final1，physical/sensor0。相对R10旧第3项R8初始化对照，现明确换成同先验R10学习率对照；不是悄悄放宽原阈值。
4. 原任务真实学习：original final成功数>自身初态1，collision<=自身初态3，physical/sensor0。
5. 更新幅度机制：最初32完整更新（128..4096）approx_kl均值<=R10同期0.07136825448833406的一半。SB3日志为最后epoch minibatch近似reverse KL，不是精确最终分布KL；机制PASS不能替代导航PASS，也不能唯一证明因果。

原成功标准distance<.10m、speed<.15m/s连续5步，碰撞优先不变；任何结果不得事后改阈值。

## 受限实施顺序与安全

先新固定profile/实际LR初始化与重载/no-update tests RED→GREEN，再全当前OPS/Tensor/历史选择回归、源码绑定、独立只读GO、V2登记/recover/doctor、所有历史文件/预算/stop/资源核验，精确prepareSHA后唯一启动。没有普通阶段人工确认等待，但不跳过审查。source运行期冻结，低于1536MiB/2GiB、NaN/身份/未知终止/时间限制等即停保留，旧attempt永不重试。

复用原PPO worker只加schema11固定分支，不复制算法，不改历史环境。沿R10 goal-stop诊断，不作为观测/奖励。真实结果MD/JSON/图/独立复审/长期文档/Commit/OPS sync/远端后再进入下一安全工作。单seed/小固定Val、粗略碰撞体、R7结构损坏诊断Minor、全程含Val最低Mem采样不足等限制继续披露。PX4/FASTLIO/实机/封存Test均禁止。
