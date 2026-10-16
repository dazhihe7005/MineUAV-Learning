# AUTORL-R13：Actor状态归零锚定，零更新接口预检

## 【1】阶段目标

针对R12固定输入中的累计均值偏置，验证独立Actor参数化是否兼容当前SB3训练链。不是训练、导航评价或真实安全认证。R12已完整推送79befa26f46f3b8bedd55320eed3bb3869dc5518，不重做旧模型诊断。

## 【2】技术背景

R11低LR首32更新KL降低，但短/原目标最终Val均0成功6碰撞；R12固定背景下其final零state速度指令范数约1.19m/s，yaw均值约2.108裁剪至+1。这支持累计偏置假设，未证明唯一根因。仅进一步降LR或std不直接约束常量偏置；本阶段先验证结构性候选，禁止据此声称学会停稳。

## 【3】系统架构

新StateCenteredPolicy继承原SB3 MultiInputActorCriticPolicy。一次共享CNN产生LiDAR64，拼接state7、heading2、age1得74D；Actor对真实特征和同背景零state特征各算一次，共享参数后求均值差。Critic仍读真实74D。Gaussian std可训练，动作clip及ENU速度尺度沿用原接口。没有新地图、未来状态或特权目标路径。

## 【4】代码实现

新research_regression/authorized_rl/state_centered_policy.py含CenteredMlp/CenteredHead、新固定工厂、原先验适配、实际拓扑/优化器引用核验及规范参数比较SHA。旧环境、actor_prior、make_ppo、CheckpointManager、Reward、物理、控制器和授权账本源码均不改。新centered_preflight.py/run.py固定phase、120s owned-group守护、唯一run目录、旧证据和当前准入核验；禁止任意脚本/模型路径/seed/训练步CLI，PPO.learn/train/Adam.step及环境reset/step均拒绝。

## 【5】算法原理

设x=[l64,s7,h2,a1]，anchor(x)=[l64,0,h2,a1]。mu(x)=head(Actor(x))−head(Actor(anchor(x)))。同一背景非零state响应仍可由LiDAR条件化；测试用FORCED_TEST_STATE明确演示，非学习结果。state7全零时均值归零及head常量bias抵消是有限计算的结构恒等式，不等于速度制动、稳定性或避障能力。该限制也可能抑制有用的偏置动作。CNN/Critic参数数量未增，Actor MLP计算两次；没有改成另一套PPO算法。

## 【6】实验协议

AUTORL_R13_CENTERED_ACTOR_PREFLIGHT_ONLY；CPU1/120s，唯一.research_local/autorl_r13_centered_preflight/run_01。训练/optimizer/物理/新扫描/episode/Validation/Test全0，预留预算0。R12真实旧扫描背景不改时间戳；62反事实state，新建中心化初态、重新新建旧类先验参考、重载本轮新初态各62条，共186条实际推理，非186新传感器帧。旧训练checkpoint不用于初始化。

REG1d32001d24bdcb0e06ae842f1bcc278a462796920ca9287a30a172ecf78d6887，SOURCEf968de2cb239660cf3d9bde370a7fa6f25e3566c00bbf82652e7a4224dc31c24（100文件），execution21af8a4e535a7f36cfd71df13d675eb2052a235aabbfe755c8953827f41a4d7b。当前LR3e-5、初态std.3及其他冻结超参数只用于构造/推理，不执行训练。Mem<1536MiB、磁盘<2GiB、stop/身份变化即停，不自动重试。

## 【7】实际结果

工程接口PASS；唯一run exit0、owned group已清理。result SHA1107e3fe8388b6b36910a469a1a35324c51efac2a6fabeaf2dd41eb923876c5d。网络170857参数；新完整policy SHA20f5c226abc762e5aa4a9046ba400b2b0f7012b0da6715d14204b097cae06001，与旧类6312e99c…不同；规范键名参数值SHA恰6312e99c87fd5a4dca066bd1ef64acd17d5ee44b563c152c59c133ee659d7f74。规范SHA只证明参数值相同，不把新函数冒充旧策略身份。

62真实历史背景反事实输入上初态raw均值最大差0、Critic最大差0；std/log_prob等价由独立单测核验。新初态保存重载62条完整输出相同、权重与空Adam状态一致。无非空Adam状态重载实测：NOT TESTED，不声称精确恢复；仍RESTART_REQUIRED。本轮保留local-only新初态ZIP，非正式训练模型。

完整选择回归144OPS+50Tensor+885历史=1079PASS，15排除、0测试optimizer；新4纯边界+12神经测试，非全仓PASS。先缺模块RED，再GREEN；早期结构检查Important通过Actor/Critic/CNN无参数激活损坏注入RED→GREEN。补齐真实62输入/log-prob、非零state bias抵消、工厂参数一致性。既有cv2/NumPy ABI stderr仍由SB3未用Atari导入捕获，CPU链成功；历史套件2既有warning，不升级依赖。

inner wall6.594716s/CPU5.774181s从imports后计起，含构造、保存重载和结束检查；launcher13.659884s含启动核验。whole-worker HWM487.266MiB含imports，模型前MemAvailable8302.387MiB仅单次采样，不是全程最低或联合仿真实时证明。221授权文件全字节不变，18账本事件、累计预留589824step/32400s，余1410176step/54000s、stop=false；无退款/新预留。四raw身份和两张实际网络输出图见JSON。

## 【8】问题分析

CONFIRMED：所有SB3 Actor调用统一使用差分均值，Critic仍用真实输入；参数对象/顺序未丢出优化器；新类实际保存重载正常；原初态有限panel等价。

SUSPECTED：约束常量偏置可能保留短目标反馈，为后续PPO改进创造条件。

UNVERIFIED：学习增益、物理闭环制动、障碍避让、泛化、长训练稳定性。不能把结构强制零输出作为达标奖励或成功率。

一次派生绘图脚本误将Path传给要求bytes的clean_png，实际预检已成功且未受影响。原脚本/部分图/错误说明独立归档后，只修后处理并重新生成派生图；不重跑实验、不修改冻结运行源码或raw。Matplotlib既有Axes3D导入warning不影响这两幅2D图。

## 【9】局限性与审查

预审实际GO，C0/I0/新Minor0；两Important及两Minor在实际run前解决，local收据保留。继承R7 active_task诊断上下文Minor不适用无环境R13，未声称消失。最终结果复审及Git发布状态以真实收据为准。只有一个历史LiDAR背景，随机合成背景单测明确不是扫描。差分可能限制动作表达，std仍导致随机动作；无非空Adam状态恢复、无新任务/Val/Test/PX4/FAST-LIO/实机验证。

## 【10】下一阶段

先完整结果复审、历史audit、Commit/OPS sync及远端SHA核验。然后独立R14设计：从新初态训练65536步/3600s，继承R11低LR/std/距离课程，以中心化Actor作为单一函数变量；新类不复用旧step0导航评价，计划36个新Val、Test0。新协议、当前回归、独立GO及实际预算检查通过后才预留和启动；不重试旧ID，不加载旧权重继续。

## 【11】给项目负责人的总结

这阶段把“没有位置/速度/航向误差时，策略仍持续发非零均值”作为可检验问题，做了一种能抵消该偏置的网络表达。代码、梯度和存档链通过，但还没有训练，因此算法导航性能没有已证实提升。下一步用与R11相同的小预算做真实对照，看看限制偏置能否帮助到达和停稳；也可能失败，届时保留证据继续分析。现有独立授权仍有效且余额未变，不需要重复授权普通研发；不会启动PX4或真实设备。
