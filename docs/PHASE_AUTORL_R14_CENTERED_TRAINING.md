# AUTORL-R14：中心化Actor的真实PPO对照（工程PASS，科学总验收FAIL）

## 【1】阶段目标

研究R12确认的固定输入均值偏置：相对R11仅改变Actor均值参数化，观察是否保留短目标能力及改善原目标导航。不把结构零输出或手写先验当学习结果。

## 【2】技术背景

R11仅降低全局LR后小KL不等于导航成功；R12只读诊断支持累计偏置假设但非唯一根因。R13零更新验证新Actor接口；R13P修复Git已跟踪图发布问题并完整推76bd8065056fbb392e45d8f79b9e7c6aa5b394a5。原R13实际源码身份与修复后管理聚合身份不同，已披露；模型模块字节未改，不重跑旧run。

## 【3】系统架构

原MuJoCo/24k射线/AngularRangeEncoder/异步缓存→LiDAR64+state7+heading2+age1→两次共享Actor MLP求差→原Gaussian探索、动作clip→原PI/四电机物理。Critic仍用真实特征。原监督/预算/flock/lifetime/resource/source/失败记录保护不变。

## 【4】实现

新centered_training_profile.py固定schema14/ID/预算/历史proof/当前准入；新centered_training.py复用原先验赋值、已验policy、scheduler和CheckpointManager。budget/runtime/__main__/worker仅增加显式schema14分支，不开放任意脚本/参数/旧模型输入。worker更新前后和checkpoint重载核实际结构/LR/参数引用；评价36新episode，不走旧初态复用。旧导航环境/Reward/终止/动力学/PI/模型权重不改。

## 【5】算法原理

mu(x)=head(actor(x))−head(actor([same LiDAR64,zero state7,same heading2,age1]))。一次CNN、两次Actor MLP、170857参数，不复制PPO优化算法。零state均值零是结构，不保证实际停稳；Gaussian std仍可训练，部分可观测性/探索/表达受限/Critic拟合问题仍可能导致失败。

## 【6】冻结实验协议

ARL_R14_CENTERED_ACTOR_SEED0，65536训练步/3600s保守预留、CPU1/seed0；新随机seed0网络赋原手写先验及std.3，不加载任何旧ZIP。全局LR3e-5，128/64/2 PPO rollout/batch/epochs，其余超参数、NAV_V2_COLLISION_PREFLIGHT_V1、500/100/25/10Hz与24k射线同R11。12短Train先训练到32768，下一自然reset切12原目标，不改几何。0/32768/65536各checkpoint关闭Train环境后实际6原+6短Val，共36，Test0，选择Final。

REG3630bf9efc2b87f2c3d96860a269a8750aaa1e64bdd950f438f7b4fe90e6acd9、SOURCE87cd42df2c4cd497f609fea44f2b1f1105548e19e1cc9043d2047df335a1651e（104文件）、executionaaa06e48b1f1b4b6b23389b238064e3a027179b7e6ae9f6a370a16003118fa4d。详见AUTORL_R14_CENTERED_TRAINING_DESIGN.md四个预注册判据：短候选≥4/≤1，保留own短初态，原目标严格胜own初态且碰撞不增，相对R11 final0成功6碰撞改进。各自判定，不看到结果改标准。

## 【7】实际结果

完整准入152OPS+51Tensor+885历史=1088选择PASS、15排除、0测试optimizer，另4合成分析契约GREEN，非全仓测试。新7纯+1真实初始化/保存重载/20tick一步hook已RED→GREEN。早期绿色命名收据实际失败是测试错误要求缺省reuse字段，而实现明确false；修正测试并保留收据。独立C0/I0/新Minor0/继承R7Minor1 GO；register/recover/doctor/prepare精确SHA实际通过。

唯一run已主管exit0及账本COMPLETE，实际65536step/512rollout/1024个PPO epoch/2048个Adam更新；1309387真实物理tick，共2618.774s训练物理时间。部分终止step提前停止物理推进，不能假定每step固定20tick。306条结束episode，307次reset，尾23步未结束episode不纳入结束统计；下一自然reset在32901步切课程。全部24个Train任务及12个基础几何身份访问可核，Test0。

| 域 | 结束episode | success | collision | timeout | physical/sensor |
|---|---:|---:|---:|---:|---:|
| 短距离Train | 108 | 34 | 10 | 64 | 0/0 |
| 原距离Train | 198 | 4 | 134 | 60 | 0/0 |

| 实际checkpoint | 短Val 成功/碰撞/超时 | 原Val 成功/碰撞/超时 |
|---|---|---|
| 初态0 | 6/0/0 | 1/3/2 |
| 32768 | 6/0/0 | 1/3/2 |
| final65536 | 5/0/1 | 1/3/2 |

36条全部新执行，无旧初态评价复用；三个checkpoint均实际保存、校验预测/权重/optimizer并重载，两个训练checkpoint包含非空Adam状态。初态新policy20f5c226…，final policy参数SHA fddca27533edbd8df551eae297ef9f26a548214fd37fc6bb00048b9c22ec3fc8。完整ZIP/sidecar身份见机器报告，不与policy参数SHA混淆。

预注册短候选PASS（5成功0碰撞）；短先验保留FAIL（5<6）；相对R11原目标final的0成功6碰撞改善PASS（本轮1成功3碰撞）；相对自身原目标初态严格学习提升FAIL（1不大于1）。四项合取科学总验收FAIL，不改门槛，不把保留手写能力称为学会避障。全512个记录点approx_kl均值0.0000204402、最大0.000354198，clip_fraction全部0；这是每次最后epoch的minibatch诊断，不是累计策略KL。explained_variance均值0.09148、final0.50909，不用单点代替全程趋势。

主管实测777.031s，wrapper788.928s；整个reaped child-tree HWM1789.898MiB，训练记录最低可用内存7049.176MiB。包含Val的连续最低可用内存NOT AVAILABLE，不能冒充已测。无观察到OOM。环境step p50/p95/p99/max=1.168/12.058/12.420/15.498ms；扫描6.363/6.807/7.025/10.236ms；编码4.096/4.299/4.482/6.333ms；PPO update18.642/19.257/21.105/27.933ms。

summary SHA0cd11a7161b2518ee9aa48c180d31435c723ae6f9978cd785128d788a074fafd，全部raw local-only清单、五张实际数据图及分析方法身份见autonomous_rl_centered_actor_r14.json。220个旧不可变授权文件及R13四raw字节未变，原模型及七旧checkpoint核验保持。累计保守预留655360step/36000s，余1344640step/50400s，stop=false，不退款/扩大。

前25条step5..125存储采样position/velocity/action与R11完全相同，仅支持这些采样的初态公平性。原_01审查抄录重复key在任何登记/预留前被严格JSON拒绝，invalid保留，真实GO正确_02用于登记；运行源码/测试身份未变。独立实际结果复审及最终同步以机器报告/本地OPS收据为准，未完成前不称PHASE_COMPLETE。

## 【8】问题分析

CONFIRMED：中心化本轮相对旧类R11保留更多短目标能力，但仍少于自身初态；原距离Val三次相同终止计数，没有严格学习提升。原Train有51个episode进入目标距离阈值，814个近目标采样中779个仍速度过大，只有35个距离/速度联合达标采样；这支持“到达不等于停稳”问题，不证明唯一根因。全部Train无符合motionless定义的episode，不是靠长期完全不动降低碰撞。

SUSPECTED：状态反馈形状/制动能力以及Gaussian探索可能仍妨碍停稳；中心化抑制常量偏置，但不自动创造速度反馈或绕行决策。UNVERIFIED：神经网络是否真正使用LiDAR、单变量因果能否跨seed复现、如何改善原距离学习。下一步先只读新R14权重与已冻结反事实输入，检查位置/速度/yaw响应，不能重复R12旧模型实验或直接盲目增加训练。

## 【9】局限性

单seed/6小Val每域、手写初始化与结构约束，不是现实矿井能力；无Test/PX4/FASTLIO/硬件。既有可选cv2/NumPy ABI stderr和历史warning继续披露，未升级依赖。R7畸形active_task诊断Minor仍在；EXACT/BOUNDARY恢复未实现，RESTART_REQUIRED。失败不重试本ID，不从中间checkpoint继续。

## 【10】后续

完成本阶段结果复审、历史audit、显式Commit、正常OPS sync及远端核验后，进入R15新R14固定输入反馈诊断（零optimizer/新物理episode/Test），以本次真实停稳证据决定下一单变量实验。先新身份、准入与独立GO，R14已有attempt不得重复或warm start。不在普通科学FAIL终止长期计划。

## 【11】给项目负责人的总结

这次真实训练完整完成，短目标验证还能成功5/6，比之前低学习率模型的0/6更好；但初始手写控制本就能6/6，所以不能说训练让它更聪明。原距离仍1/6成功，与自身初态一样，避障和停稳没有达到整体目标。下一步先检查新模型如何响应目标误差与速度，再决定只改一个变量的新实验；不会无意义重跑百万步，也不用你反复确认普通研发。剩余原授权1344640步/50400秒，未扩大预算，PX4/实机/Test仍不运行。
