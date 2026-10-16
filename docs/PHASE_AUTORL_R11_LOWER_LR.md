# AUTORL-R11：解析先验全局低学习率对照

## 【1】阶段目标

检验R10先验迅速丢失是否与过大的PPO函数更新有关。相对R10只将全局LR3e-4降至3e-5；不是重试R10、不是从旧checkpoint继续，也不预设必然改善导航。

## 【2】技术背景

R10已完整发布21831fcc90d06756f431373643e75822b633eb7a，65536步后四项导航/保留标准FAIL。短Val初态6成功到final0，最初32更新approx_kl平均.0713683、clip_fraction.337402。先验可完成性R9/R5已验证，不重复。较大更新是假设证据，不是唯一根因证明。

## 【3】系统架构

独立grant→新schema11固定prepare/start→既有supervisor/flock/guard→原worker→R8/R7/R5 scheduler/身份adapter→同PPO/CNN/PI/MuJoCo/24k raycast→rollout优化后checkpoint→关闭Train再独立Validation。单训练worker，主管/guard不是额外训练worker。

## 【4】代码实现

lower_lr_profile.py固定ID/配置及R10/R9历史proof/current admission；lower_lr_training.py在新鲜0step网络、未reset和空optimizer时覆写learning_rate、constant schedule及全部optimizer组，再复用原先验初始化与step0保存。每次优化前后和checkpoint重载核验实际LR。budget/runtime/CLI/worker仅加schema11分支，不改grant、旧CheckpointManager、历史make_ppo/环境/Reward/控制器/动力学。

## 【5】算法原理

PPO仍128rollout/64batch/2epoch、gamma.99/GAE.95/clip.2/ent0/vf.5/maxgrad.5。初始手写state反馈均值和可训练std.3不变，LiDAR[3,12,24]+state7+heading2+age1不变。全局LR同时影响Actor/Critic，可能在保留先验的同时减缓Critic拟合。比率clip不是KL硬界限；SB3近似reverse KL日志为最后epoch的minibatch估计，不冒充精确最终分布KL。

## 【6】实验协议

ARL_R11_PRIOR_LOWER_LR_SEED0，65536实际环境步/3600s保守预留、CPU1/seed0，新随机网络赋先验，不加载旧权重。REG6e6907a34e99547967d7cfcbd85740eb4fe20fde23edc205e47b9679a3ecb003、SOURCEab7544a0c895f3a406bb3123138fe82dbe059932d748bf7ae107b319f54897d8、execution25c6571ecf76757b052a73ca41d620c5d42e39c42d473c36223ae65ee5b80e39。原L6/L7 hash只是来源，新execution明确LR覆盖，不改旧协议hash。

short .20m至32768下次自然reset换原目标，同500/100/25/10Hz/24krays/7kg模型/PI/NAV_V2_COLLISION_PREFLIGHT_V1。checkpoint0/32768/65536，30新Val；原step0六条仅复用R3 PRIOR同6312…完整policy的确定性评价，不声称optimizer或ZIP相同。Final Checkpoint/Test0/RESTART_REQUIRED，原distance<.10、speed<.15连续5step标准不改。

五项预注册判据独立报告：short final≥4success/≤1collision；short先验保留不劣自身初态；original成功>R10 final0/collision≤R10 final1；original成功>自身初态1/collision≤3；前32更新approxKL均值≤R10同期一半(.03568412724416703)。各导航标准另要求physical/sensor0。学习率对照不同于R10的R8初始化对照，明确变更基准而非事后调阈值。机制PASS不等于学习PASS。

## 【7】实验结果

工程执行PASS/主管与账本COMPLETE；四项导航与先验保留判据全部FAIL，更新幅度判据PASS。唯一65536实际step/512rollout/1024PPO epoch update/2048Adam，1306104实际物理tick、合计2612.208s仿真，不能用step×20外推。summary SHA7f34a1e98eae2ef289becacb4d1086f7e5dbb0d686d2cae750610452d90b8f26，final ZIP9ec87ba5379a2fc6473d0b6bc501bae9b3699119292605e99f5a3604230c8cf5。23原始文件local-only；全部3checkpoint权重/optimizer/确定性预测重载通过，3个实际重载学习率及512条train/learning_rate均为3e-5。Actor/Critic真实变化且权重/梯度有限；旧L6/L7来源hash未改，不假称实际仍用旧LR。

574个结束Train episode=3success+481collision+90time_limit、physical/sensor0。SHORT106=3成功24碰撞79超时，ORIGINAL468=0成功457碰撞11超时；自然reset切换32811，最后episode恰在65536结束、未结束尾步0。12base场景访问36–59次/575reset（最后自动reset无新step），全部24Train任务身份正确。三次成功均在短任务，终点总step273/705/15246，后两次也不能作为稳定学会导航的证据；原任务始终无成功。总体mean return−24.4869065。

| checkpoint | short成功/碰撞/超时 | original成功/碰撞/超时 |
|---|---|---|
| 0 | 6 / 0 / 0（手写先验） | 1 / 3 / 2（历史R3 PRIOR，仅评价复用） |
| 32768 | 0 / 0 / 6 | 0 / 4 / 2 |
| 65536 | 0 / 6 / 0 | 0 / 6 / 0 |

30新Val全部真实结束，无probe_cutoff或physical/sensor失败，Test0。初态short6成功在32k已经退化为6超时，final进一步变为6碰撞；不能把全部遗忘归因后半课程，也不能把32k零碰撞当任务成功。相对R10 final原任务0/1/5，本轮0/6/0并无导航收益。四项预注册导航标准均FAIL，不改成功或碰撞阈值。

SHORT25个episode采样进入目标球，333近目标采样=312速度不满足+21joint；max连续joint=5仅在3成功episode，其余103短episode max joint=0。ORIGINAL14个episode采样进入球，但29个近目标采样全部速度不满足、joint0。聚合不能代替连续物理轨迹或证明单一速度根因。481次实际collision中3条有floor接触，其他478条来自箱体/边墙；不混称全部撞地，也不把collision改作physical_failure。阶段接触对完整计数见JSON。

预注册前32次完整更新：R10/R11平均approx_kl=.07136825449/.002472223481，后者为前者约3.46%，小于一半阈值.03568412724；机制PASS。对应clip_fraction均值.33740234375/.00817871094。全512更新approx_kl均值.00516177074/最大.02004942857，clip_fraction均值.02879333496；value_loss均值36.5166021/最大171.6190681/最后93.0170555，explained_variance均值.00281618/最后.00254911，std最终.299824357。降低步长确实降低更新幅度，但没有保住先验或改善导航；不能由此宣布学习率无关或另一个因素是唯一原因。Critic解释方差偏低是诊断事实，不是唯一因果证明。

主管wall828.7288s、worker827.4179s；训练health末记录wall794.3600s/CPU782.1694s。whole-worker HWM1790.242MiB含重载/Val，训练观察HWM1689.367MiB；训练记录累计最低MemAvailable7296.363MiB，全程含Val最低NOT_AVAILABLE。环境step P50/P95/P99/Max=1.176/12.416/12.824/18.391ms；scan6.539/7.078/7.348/12.743ms、encode4.204/4.457/4.651/6.909ms，均25872新增训练帧、不含reset/Val；512优化update16.840/17.695/19.219/23.019ms。未观察OOM，不证明百万步内存稳定，不把系统Mem变化唯一归因泄漏。

新6纯契约+1真实LR/重载/首step测试RED→GREEN；最终完整当前131OPS+36Tensor+885历史选择回归=1052项PASS、15明确排除、2既有warning、0测试optimizer，不称全仓PASS。另5合成分析契约RED→GREEN；分析源码ec0332…/纯判据方法、finalizer aa6134…均获独立只读方法GO，不冒充实际结果认证。准入SHA90816f9d…，独立预审C0/I0/新Minor0/继承R7 Minor1 GO后才登记/恢复/doctor/历史/预算/stop/resources/prepare精确SHA及唯一启动。初始管理辅助脚本误读recover返回形状KeyError：实际register PASS/recover PENDING；仅继续doctor和历史检查，没有重登记、预留或重复attempt，真实收据保留。

SOURCEab7544a0…执行前后未变；194旧不可变授权文件（只排除可变head）及R9原17文件字节不变，R10旧23raw和报告身份保持。只新增R11协议/run、事件17RESERVE/18COMPLETE及head。累计589824step/32400s保守预留，余1410176step/54000s、stop=false，不退款。5图来自真实raw，分箱只从右端可用，主代理目视检查无裁切。最终实际结果独立复审、Git交付与远端核验以真实收据为准，不预先认证。

## 【8】问题分析

CONFIRMED：新LR真实生效，早期及全程更新幅度下降，但短任务先验仍丢失、原任务最终全碰撞；“只降到3e-5即可解决”的本轮假设被反驳。SUSPECTED：累计反馈漂移、探索相对近目标容差过大、Critic和奖励信用分配等仍值得分别研究。UNVERIFIED：其中任何一项是唯一原因，或再降低LR/增加步数必然有效。

## 【9】局限性

单seed/小固定Train/Val/noTest、不证明LiDAR依赖或泛化。粗机体碰撞体TEMPORARY_PLACEHOLDER；近似MID360整帧静态raycast不是真机时序。R7任意active_task结构损坏可能遮蔽context的继承Minor保留。可选cv2/NumPy ABI及Axes3D警告未全局修复，不用未调用功能认证训练。现有资源记录不能给全程含Val连续最低Mem。

## 【10】下一阶段

先完成真实结果独立复审/历史audit/Commit/OPS sync/远端；本ID永久不能重跑或warmstart。随后优先R12零optimizer固定反馈诊断：复用R4真实背景和已预定义62反事实输入，只读分析R10/R11新冻结32k/final策略，明确新模型而不是重做R4旧实验；比较零目标偏置、目标方向与速度敏感性，区分低KL和累计控制反馈漂移。先冻结新身份/准入/独立审查再执行，不先盲调奖励或加步。没有重要阻塞则在原授权内继续下一独立研究，PX4/FASTLIO/实机/Test禁止。

## 【11】给负责人的总结

这次训练完整跑完，学习率也确实降了，网络每次变化明显变小，但原来能到点的手写反馈仍然丢失，最终两组评价全碰撞。因此不能再简单说“只要学习率小一点就会学会”。3次训练成功不等于掌握导航；下一步先用不训练的固定输入检查新模型具体把方向和减速反馈改成了什么，再决定单变量改进。余额约141万步/15小时保守wall，仍在原授权内，无需普通阶段重复确认，不启动PX4或真实设备。
