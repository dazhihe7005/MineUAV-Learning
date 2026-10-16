# AUTORL-R17：空场景重放比例对照——工程完整，科学FAIL

## 1. 阶段目标

检验后半程原距离训练把empty抽样概率从25%提升至50%，是否保留基础到达停稳能力。R16已完整发布c8307c0f670787bb2a4897e14034452cbd57588f，未重跑。R17训练主管/账本COMPLETE；阶段最终交付仍需本报告独立结果复审及正常OPS sync远端核验，不能提前宣称PHASE_COMPLETE。

## 2. 技术背景

R16原域Validation初态1成功/3碰撞/2超时，32k为2/3/1，final为0/3/3；两个empty曾在32k成功、final超时。R17把课程混合分布作为唯一算法变量，不先认定是遗忘、Reward或学习率的唯一问题。既有三个Train empty Scripted可完成性证据SHA5f482d0711652f14c8d925a8d4d6f24f467686f1a10eb2787c65c66048c4c70c只读复用，21项旧runtime源码当前全匹配；没有重复基线或冒充LiDAR策略。

## 3. 系统架构

固定L6的12 Train→short课程原调度器→EmptyRehearsalAdapter（reset-only）→原DistanceTaskAccountingAdapter身份/目标/几何/失效锁存→真实MuJoCo/PI/24k raycast/编码→原中心化Actor PPO。原验证器只在Train环境关闭后独立执行新Val；Test始终封存。新适配器没有把layout、障碍物坐标或抽样权重送入神经网络。

## 4. 代码实现

新增`research_regression/authorized_rl/empty_rehearsal.py`、`empty_rehearsal_profile.py`、真实链测试`empty_rehearsal_tests.py`及纯协议测试`tests/research_ops_v2/test_empty_rehearsal_profile.py`；budget/runtime/CLI/worker仅显式增加schema17与固定prepare/start路径。CLI不能任意传seed、权重、checkpoint、脚本或预算。原Actor core、动力学、Reward、控制器、历史协议均不改。

SHORT不新增RNG调用。达到32768后当前episode不改target，下一自然reset才对3 empty各1/6、9 nonempty各1/18加权。显式Train覆盖测试不消耗抽样RNG；非法seed/身份继续锁存失败。单MuJoCo实例，实际reset清PI积分、qvel、LiDAR缓存。32k权重强制核为R16的33bf6699158a35842fc05c3d4ca97f391f778db60c759fdad2564a6e94c87f60，实际完全一致；运行期源码冻结无修改。

## 5. 算法原理

中心化均值为f(实际74D特征)−f(同当前背景、state7归零)，不是强制Actor保持初始反馈曲线。Gaussian初始std=.1且可训练，全局LR3e-5；CNN/Critic/170857参数、PPO128/64/2及其他超参数、NAV_V2_COLLISION_PREFLIGHT_V1 Reward、动作尺度不变。状态仍显式SIM_GROUND_TRUTH_STATE，500/100/25/10Hz与24000射线不变。抽样重加权改变经验分布，不等于网络学会绕障；因此Train总体return/失败率不能当作同分布导航提升。

## 6. 冻结协议与身份

唯一ARL_R17_EMPTY_REHEARSAL_SEED0，新seed0网络，禁止任何旧ZIP初始化/续训。65536实际step、3600s上限、CPU1worker、RESTART_REQUIRED。0/32768/65536三个checkpoint各6原域+6短域新Validation，共36；历史评价复用0、Test0，选择固定FinalCheckpoint。保留原成功标准：误差<.10m、速度<.15m/s、连续5策略step。

- REG：8242b52bdc21cebcf5a72461a20e961f1d17f6893d47801eb6159505bc6f6f9c。
- SOURCE：590c019185093d572ed27bde12b2da3d130f6ffdcd77c06d465cb0c4ee1081bf（116文件）。
- EXEC：3140266ae8034fc0114ff5334961fb8260254c100d8971eb0f3c46fb4b348f1b。
- 准入：242804790abc0dd1de6214b8adb44979d7f34d832ffa8e7b2de4bf24961c973d。
- 真实summary：950a588ea593f5eb0ece276748624e590efb231472a5c5aacb67098946da2f72。
- 初始policy650a5841…；32k policy33bf6699…；final policy72979b2166726dab2b19ae7dea50b53806ecb7625074a90c67b08df5bab82535。
- final ZIP e5f66c5efb24b8d158c33cdeaf5dac432559fc76791ef0e76c87d26373e4f4f5（local-only）。三checkpoint均有实际权重/预测/optimizer保存重载核验，两个训练checkpoint含非空Adam。

## 7. 真实结果与资源

65536step/512rollout/1024 PPO epoch计数/2048 Adam，1309913物理tick=2619.826仿真秒；提前终止使其不是65536×20tick。394个结束Train=243成功/81碰撞/70超时；物理/传感器失败0。SHORT251=223/18/10，与R16完全一致；ORIGINAL143=20/63/60。395 reset，实际切换32850，尾238步未结束episode仍计预算、不计结束统计。后半程144次加权reset中77次empty=53.47%，不是保证每次小样本恰好50%。24个实际task、12个原geometry均有访问记录。

| checkpoint | short成功/碰撞/超时 | original成功/碰撞/超时 | original两个empty成功/超时 |
|---|---|---|---|
| 0 | 6/0/0 | 1/3/2 | 0/2 |
| 32768 | 5/1/0 | 2/3/1 | 2/0 |
| 65536 | 5/1/0 | 0/3/3 | 0/2 |

final原域front_box/detour/channel仍碰撞，side_pair和两个empty超时。两个empty末误差.123842/.119343m，side_pair末误差.072179m；后者离目标近也不等于满足连续停稳，不能凭末误差推断真实速度原因。Val无完整逐step速度轨迹，不能编造振荡或速度证据。初态1成功来自side_pair而非empty。

五项冻结判据：short candidate **PASS**；short own-prior保留 **FAIL**；相对R16原final改善 **FAIL**；相对own原初态学习 **FAIL**；empty32k→final保留 **FAIL**。科学总**FAIL**，不得挑32k为最终模型。原Train72条episode有采样进入目标附近，1455近目标样本中1340仍高速、115 joint；20次成功达到连续5样本；无motionless episode。不是所有失败都“从未接近”。

512更新KL均值.000199376/max.002566617/final.000026578，clip fraction均值.000106812。explained variance均值.242108/final.956383，value loss均值11.679092/final.040619；这些是各rollout诊断，不能用最后Critic拟合较好推断导航成功。std最终.0998916；raw动作裁剪比例均值.034298，不是电机饱和。

主管844.741s，外层856.937s，whole reaped-child HWM1730.730MiB；训练采样minMem6644.664MiB，含Val连续最低**NOT AVAILABLE**。末训练累计CPU783.189s/wall803.021s=单核97.53%，不是全系统或含Val完整CPU。step耗时ms P50/P95/P99/Max=1.171/12.003/12.300/19.044；扫描6.336/6.768/6.997/13.758；编码4.077/4.226/4.363/6.389；PPO更新18.532/19.100/20.030/26.320。未观察OOM或资源保护故障。

8纯协议+5真实链测试RED→GREEN；当前174OPS+60Tensor+885历史=1119选择PASS、15明确排除、0测试optimizer；另5分析方法契约通过，不称全仓。独立预审发现实际overall额外字段接线Important，训练前以真实R16格式RED复现，显式计数投影修复并复审关闭；原方法/NO_GO/RED均保留。最终分析方法5e6960845520111c77acc14e3abdacb21e704f7acd4d464a4d6e4dc02fbc9752。预审C0/I0/新Minor0，继承R7失败上下文Minor1。

五张实际图见`mujoco/reports/autonomous_rl_empty_rehearsal_r17/figures/`的episodes、validation、ppo_updates、goals、resources.png；主执行者已目视核验。23份raw的路径/大小/SHA均在机器报告，不上传ZIP或长日志。272旧不可变授权文件、R13/R15七raw、七旧模型及原物理身份保持。累计预留786432step/43200s，余额1213568step/43200s、stop=false；24事件/12attempt，不退款、不重置旧账本。

## 8. 问题分析

**CONFIRMED**：32k相同权重、short完整经历一致，后半程真实empty权重变更正常；该单seed/65536步设置下没有保留原empty成功，final原Validation仍0/6。**SUSPECTED**：到达后停稳反馈及其随更新的改变仍可能重要。**UNVERIFIED**：唯一根因、Val完整动态速度原因、更长训练效果和真实泛化。不能由此宣称所有课程都无效，也不能无记录增加预算重试。

## 9. 局限性

六个固定小Val/域、单seed，不外推真实矿井/飞行。手写先验、结构零输出和Train成功不是端到端避障已学会。现有原始轨迹只对最初少量episode有完整稀疏记录；不能假称保留final Val的逐步速度轨迹。可选cv2/NumPyABI stderr及Matplotlib Axes3D警告仍在，CPU/2D路径实际通过，无依赖升级。无PX4/FAST-LIO/硬件/Test，无EXACT/BOUNDARY resume。

## 10. 下一安全阶段

先独立实际结果审查、历史audit、Commit/正常OPS sync/远端SHA。然后R18优先零optimizer的停稳动态诊断：明确冻结新诊断身份与有限Train/Validation权限，读取新32k/final冻结策略的反馈或采集必要逐step动态证据；不得冒充旧Val原始轨迹、重跑R17训练或warmstart。需先设计、测试、独立GO。保持预算账本，不因普通科学FAIL终止长期任务，也不直接盲加训练。

## 11. 给项目负责人的总结

这轮验证了一个具体想法：训练后半程多练空场景，能否避免基础能力下降。代码和受监督训练完整跑通，前半程与上轮完全一致，后半程确实增加了空场景比例；但最终原任务仍没有成功，不能说避障能力提升。困难不只是碰撞，也包括到达附近后未停稳。下一步先用不更新网络的诊断补足真实动作和运动反馈证据，再决定有针对性的新实验；不需要你为普通分析再次确认，也不会扩大原授权、运行PX4或访问Test。
