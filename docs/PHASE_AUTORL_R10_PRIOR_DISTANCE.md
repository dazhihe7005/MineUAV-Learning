# AUTORL-R10：解析初始化与短目标课程PPO对照

## 【1】阶段目标

检验短任务解析先验在PPO中能否保留，以及是否有原目标的真正学习收益。相对R8仅改变Actor初始均值，保留原算法/物理/课程；不是重试R6/R8，也不是从旧checkpoint继续。

## 【2】技术背景

R9已推0bdcfa999ef75a0b4f3d34a6498d76267348b7d3，12/12短Train手写先验在原容差成功；R8已推2c73110，随机初始化距离课程65536步无成功。R3原目标随机/先验两臂、R8课程随机是既有历史对照；本候选补上先验×课程，不重做前三臂。

## 【3】系统架构

原独立grant→新schema10固定prepare/start→原supervisor/flock/独立lifetime guard→原worker→R8工厂/R7双身份adapter/R5距离scheduler→原PPO/CNN/PI/MuJoCo与raycast→rollout及优化后checkpoint→关闭Train后独立Validation。一个CPU训练worker/一个活跃MuJoCo，主管和轻量guard不算第二训练worker。

## 【4】代码实现

prior_distance_profile.py固定ID/预算/历史proof及当前准入；prior_distance_training.py复用std_training(PRIOR)初始化、严格R3 PRIOR step0复用、纯聚合GoalStopDiagnostics。budget/runtime/CLI仅新增schema10严格分支，原grant不变；worker新增初始化/评价选择和只读诊断聚合，不复制PPO。原环境/Reward/控制器/动力学和R5/R7 helper未改。新CLI没有checkpoint/seed/timesteps/任意脚本覆盖能力。

## 【5】算法原理

LiDAR[3,12,24]+state7+heading[sin,cos]+age1不变。解析先验只对Actor四条通路赋值，CNN/Critic保持随机seed0，初态全policy6312e99c…；不是BC学习。Gaussian std初始.3继续训练，采样动作在原环境裁剪缩放[1.5,1.5,1.0]m/s、1rad/s。PPO128rollout/64batch/2epochs/lr3e-4/gamma.99/GAE.95/clip.2/ent0/vf.5/maxgrad.5不变。

## 【6】实验协议

ARL_R10_PRIOR_DISTANCE_SEED0，65536step/3600s/seed0/CPU1；最终_02 REG69fcf055e72c0fee6986be3d6e4f7966e7308c9e958d9f6742bf2e5076bed302，SOURCE5170a41bb9cade75afa81f215755e5dfbfe15b7f1c9a39e736c07e5f8cca55b1，execution protocol5534e4ae5a2389e98d34b6d1aa6405f3159f1d53d85fa57084000c94365f7abf。12short .20m至32768，下一自然reset恢复原目标。500/100/25/10Hz、24000rays、NAV_V2_COLLISION_PREFLIGHT_V1、原7kg模型不变。

checkpoint0/32768/65536；Train关闭后short6×3和original6×2，共30新Validation，original0六条只读复用同6312…权重的R3 PRIOR摘要afbeac0d…，不是新episode。Final Checkpoint，Test0，不按Val择模。成功仍distance<.10m/speed<.15m/s连续5step、collision优先。四项预注册标准：短final≥4成功/≤1碰撞；短final成功不低于自身short0且碰撞不增；原final成功>历史R8 final0且碰撞≤R8 final2；原final成功>自身original0且碰撞不增。每项相关final均要求physical/sensor0，分开PASS/FAIL，不能用手写初态成功充学习，不能要求6/6天花板还严格提升。

运行前余额1541248step/61200s，如唯一预留后1475712/57600；全额保守计费不退款。Mem<1536MiB/磁盘<2GiB、stop/源码身份/NaN/未知终止/训练严重异常立即停止，保留attempt不重试，RESTART_REQUIRED。合法有限物理失败维持原PPO记账，与R9非学习严格中止模式区分。

## 【7】实验结果

工程执行PASS/主管账本COMPLETE，四项科学验收全部FAIL。唯一65536实际step、512rollout、1024PPO epoch update、2048Adam，1307707实际物理tick/合计2615.414s仿真；不是每步固定20tick外推。summary SHA c9fa8bcc9b14fe441903e1f733e7496393e39a36a72c0d2f8e907fc9778ba2ca，最终ZIP SHA90726f17fb15ef55fc464164fe1e9fbdbf6a77a110ddb8c2a71f33e05bbdec5e。23个原始文件local-only，3checkpoint重载权重/optimizer/确定性预测验证全部通过，Actor/Critic真实变化且有限。禁止重试/继续本ID。

425个结束Train episode=1success+318collision+106time_limit、physical/sensor0。SHORT165=1成功97碰撞67超时，ORIGINAL260=0成功221碰撞39超时。唯一训练成功在第一条episode/总step262，走2.20954m、10.48s到0.20m任务，原联合停稳满足；不能凭一次采样成功宣称学会导航。真实自然reset切换33100，最后137步未结束episode仍计预算。12base访问26–46次/426reset、全部24任务只来自冻结Train。

| checkpoint | short成功/碰撞/超时 | original成功/碰撞/超时 |
|---|---|---|
| 0 | 6 / 0 / 0（手写先验，不是学习） | 1 / 3 / 2（历史R3 PRIOR复用，非新执行） |
| 32768 | 0 / 1 / 5 | 0 / 6 / 0 |
| 65536 | 0 / 0 / 6 | 0 / 1 / 5 |

全部30新Validation真实结束/无probe_cutoff，Test0。原任务自身初态1成功，最终0；短任务先验6成功最终0。因此短候选、先验保留、相对R8初始化收益、原任务学习增益四项均FAIL，不改阈值。原final比R8少1碰撞但多1超时，成功仍0；不是避障改善。short32k已丢失先验，不能全部归因后半课程遗忘。短final平均路径5.512m、return−26.492，不能因0碰撞就说安全悬停。

SHORT只有8个episode采样进入目标球，总101个近目标采样其中96个速度不满足、5个joint均在唯一成功episode；其余164个短episode最长joint均0。原目标260episode从未在策略采样点进入目标球。此处新诊断只证明当前采样状态，不代表物理连续轨迹/唯一因果。318次碰撞中161条含floor接触，余157条障碍/边墙，不把所有collision混称障碍物撞击或physical_failure；分期接触对完整计数见JSON。

PPO全512更新：approx_kl均值.091482/最大.913452，clip_fraction均值.378281；value_loss均值16.1794/最大144.489，explained_variance均值−.000378/最小−5.27494/最后.015078，std最终.307476，raw动作元素裁剪均值.060429。图保留全序列，不从单个点推断全过程。前32更新(128..4096步)单独比较：R3 PRIOR/R8 RANDOM/R10 PRIOR的平均approx_kl=.073916/.003530/.071368、平均clip_fraction=.366577/.014038/.337402。这是限定前缀的事实，不是假称全程均值或唯一学习率因果。固定SB3源码ppo.py:184–294说明clip是比率目标剪切、不是KL硬界限；approx_kl日志为最后epoch minibatch估计均值，不能当成精确训练后全分布KL。

主管wall772.024s、worker770.768s；训练health末记录wall721.982s/CPU710.706s。worker HWM1778.152MiB含重载/延后Validation，训练观察HWM1698.922MiB；训练内部累计最低MemAvailable7093.902MiB，不认证含Val全程连续最低，也不把系统可用内存下降唯一归因训练泄漏。envstep P50/P95/P99/Max=1.189/12.380/13.124/19.407ms；scan6.429/6.956/7.506/12.189ms，encode4.206/4.564/4.876/7.661ms，均26004条新增训练帧样本、不含reset/Val；512PPO更新16.950/18.412/20.902/23.725ms。未观察OOM，不证明百万步内存稳定。

源码5170a41…运行前后不变；168个旧不可变授权文件及R9原17文件字节不变，只新增R10协议/run、事件15RESERVE/16COMPLETE及head。累计524288步/28800s保守预留，余额1475712步/57600s，stop=false，不退款。5张图来自真实raw、分箱只在右端可用，主代理已目视无裁切。纯分析契约5项RED→GREEN/0optimizer，分析器源码独立GO仅认证分析方法；实际结果与最终Git远端以真实审查/OPS收据为准，不预先认证。

新6纯契约/1真实hook测试RED→GREEN；首次真实hook测试使用错误的评价字段路径，先核真实overall数据后只修测试断言，没有改历史记录。_01完整124OPS/35Tensor/885历史通过后，独立预审发现short step0默认被误标为未训练随机网络的Minor，实际训练前新增标签测试RED→GREEN修正。最终_02新125OPS+35Tensor+885历史=1045项选择回归PASS/15排除/2既有warning/0optimizer，不用旧_01替代；新独立复审C0/I0、继承R7 Minor1条件GO后，新登记/recover/doctor/资源/169旧文件/余额stop及prepare实际SHA全部通过，才唯一执行。运行期SOURCE5170a41…冻结，R9/R8 proof未变。

## 【8】问题分析

CONFIRMED：手写先验初态6/6短Val成功，训练后0/6；本课程未保留控制先验、未学会原任务。PPO先验前缀和全程更新KL/比率clip明显偏大，Critic解释方差也较差。SUSPECTED：解析参数化下相同步长造成过大函数更新、破坏初始反馈，可能比单纯缺探索更值得先检验。UNVERIFIED：学习率是唯一原因、Critic因果或降低LR必然提升导航；不凭阴性结果擅自加步。

## 【9】局限性

单seed/小型固定Train/Val/noTest，不证明泛化或真实矿井安全。初态解析均值不看LiDAR。粗碰撞几何TEMPORARY_PLACEHOLDER、冻结整帧近似MID360不是真机时序/FASTLIO。R7任意active_task结构损坏可能遮蔽context的Minor保留。cv2/NumPy及Axes3D等旧warning未全局修复。R9独立runner的json_safe512节点Minor不等于本PPO故障捕获已发生该错误。

## 【10】下一阶段

先完成本次阴性结果独立复审/Commit/OPS sync远端。下一候选R11仅降低全局PPO学习率3e-4→3e-5，对照本R10；相同先验/课程/std/CNN/Reward/物理与65536步，先离线验证常量LR schedule、optimizer参数组、step0 checkpoint和重载身份均真实反映新LR，不改旧L6报告/旧源码/旧协议。新执行协议须明确基础L6元数据只是来源，不可伪称仍使用原LR；Actor/Critic共用学习率同时改变，不说成仅Actor LR。冻结导航与先验保留标准、早期更新幅度假设并独立审查，再在原余额唯一执行。若准入/安全/未知状态阻塞则停，不重试R10或直接百万步。

## 【11】给负责人的总结

这次完整训练跑完了，但没有保住原来能到点的手写反馈：短任务初态6个全成功，训练后全失败，原任务也无成功。训练中那1次成功不能代表学会导航。现有数据更值得检验“每次PPO更新是否对这套先验太猛”，而不是直接盲目加步。下一次候选只降学习率，先证明程序确实使用新配置，再审查和训练；不改本次结果，预算仍余约148万步/16小时保守wall，普通研究继续自主推进。
