# AUTORL-R16：中心化策略初始探索噪声单变量对照

## 目标与现有证据

R14已真实65536步并发布1ab46a3d…，R15已发布59e34070a166f195261abd6a3b62e051f20695b3。R14短Val6→6→5，原Val始终1成功3碰撞2超时；短Train108结束有34成功，2077个近目标采样中1867仍速度过高。R15新模型形成负速度响应，反驳“完全没学到速度反馈”，但不能证明真实闭环停稳。

待证伪假设：std约0.3的随机动作使接近目标时稳定成功经验稀少；在相同中心化均值及课程下，仅降低初始std可能提高停稳经验比例及最终原目标导航表现。Gaussian指令标准差不是机体实际速度标准差，PI/物理仍必须实际推进。

## 选择与范围

这是已有schema14受监督训练流程的有界扩展，不增加新训练框架。备选为增加速度制动先验或改Reward，但R15已观察负速度反馈，暂不同时改这两项。用户长期授权允许独立冻结、审查后在剩余累计预算内自主研究，无需重复普通阶段确认。

唯一算法变量：新模型初始log_std从log(0.3)改为log(0.1)，std仍可训练。初始Actor均值、Critic、CNN、参数数目及optimizer参数对象/初始空状态均不变。相同seed0新随机网络赋原手写均值先验，不加载任何旧ZIP。旧state_centered_policy.py保持字节身份，不改变R13/R14/R15旧源码与结果。

保持LR3e-5，n_steps128/batch64/epochs2及其他PPO超参数，NAV_V2_COLLISION_PREFLIGHT_V1、动作尺度、12短Train至32768后下一自然reset切12原Train、500/100/25/10Hz、24k rays、7kg动力学/PI均不变。只新schema16/ARL_R16_CENTERED_STD01_SEED0，最多65536actualstep/3600s保守预留，CPU1/seed0，RESTART_REQUIRED，FinalCheckpoint选择。0/32768/65536各6原+6短全新Val，共36，Test0。

## 预先冻结的判据

以下单独报告，不能以机制PASS替代导航：

1. 短候选：final短Val成功≥4/6、碰撞≤1、physical/sensor均0。
2. 短初态保留：final短成功≥本轮新初态成功，碰撞≤本轮新初态。
3. 原目标学习：final原成功严格大于本轮新初态，碰撞不增加，physical/sensor均0。
4. 相对R14收益：final原成功>历史R14 final1，碰撞≤历史R14 final3，physical/sensor均0。
5. 停稳经验机制：短Train结束episode成功比例>34/108且近目标高速采样比例<1867/2077，无physical/sensor；若分母0则FAIL。这是训练采样机制，不能称泛化或导航成功。

五项合取才科学总PASS。实际新初态不是学会导航；旧R14/R15先验保留诊断容差与本轮导航阈值不同，不混淆。无需也不允许通过Test反馈选择模型或改门槛。

## 实现与验证顺序

1. 先新profile纯门禁及真实张量/step0/重载hook测试RED，独立无更新校准新初态参数SHA。逐参数比较只log_std变，62固定历史输入的均值/Value严格一致、std分别.3/.1；错误seed/非fresh拒绝且不改权重。
2. 新centered_noise_profile/training/tests，原budget/runtime/CLI/worker增加固定schema16分支，复用原PPO算法、距离记账、真实goal_stop和CheckpointManager。旧环境/Reward/Actor核心不改。
3. 完整当前无optimizer OPS/Tensor/885历史回归，明确15旧排除；分析方法合成契约RED→GREEN。冻结新REG/SOURCE/execution/初态身份，独立预审C/I0精确GO。
4. 新OPS register/recover/doctor、全部旧授权文件/停止/预算/资源、精确prepare SHA后唯一受监督start。预留前余额应1344640step/50400s，预留后1279104step/46800s；预算不重置/退款。已有run永不重试，不warmstart。
5. 运行期源码冻结、监控、真实结果封存，中文MD/JSON/五图及长期文档，独立实际结果复审、显式Commit、正常OPS sync及远端核验后才下阶段。

## 局限

单seed及每域6个小Val，历史R14对照而非多seed同时实验；较小初始噪声会改变采样分布及PPO梯度尺度，这是同一std变量的结果，不等于其他超参数独立不变就可声称完全等价。没有LiDAR依赖、泛化、真实安全或百万步稳定性证明。PX4/FASTLIO/实机/封存Test始终0。
