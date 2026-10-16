# AUTORL-R6：短距离到点课程PPO对照

## 【1】阶段目标

检验先学习0.20m目标、再恢复原目标，是否改善PPO基础到点与停稳。**本attempt结果FAIL，首个transition被旧身份白名单拒绝，0次optimizer更新；学习假设NOT TESTED。**工程执行和学习改善分开验收，不以Reward或手写控制器成功替代学习证据。

## 【2】技术背景

R5的12条短任务已真实物理成功并独立复审/推送d4971529c63c9f2f305c78c851cb8d94b20d6487；只证明任务可完成。R4指出有限固定背景中的训练后反馈偏置，但不是唯一因果。对照是已完成R3 RANDOM/std.3，旧实验不重放，旧L9始终FAIL。

## 【3】系统架构

原独立grant/追加预算账本→新固定schema6 prepare/start→原supervisor/flock/lifetime guard→原PPO worker（仅选择R5距离scheduler和短Validation）→原CNN/PI/MuJoCo/raycast→checkpoint→关闭Train→独立Validation→真实报告。一个CPU训练worker、一个活动MuJoCo环境；守护进程不算第二训练worker。

## 【4】代码实现

distance_profile.py固定新ID、预算、证据与准入；distance_training.py复用原std初始化、独立短Validation及旧step0证据。budget.py/runtime.py/__main__.py只新增schema6固定分支，不改immutable grant/旧profile。worker.py新增schema6环境/初始化/短Validation选择，保留原采样与优化算法及故障记录，episode_details记录base/task和阶段。原物理/Reward/控制器/网络文件不改。

## 【5】算法和观测

原LiDAR[3,12,24]+state7+heading[sin,cos]+age1，来源SIM_GROUND_TRUTH_STATE，动作[-1,1]^4在原环境缩放[1.5,1.5,1.0]m/s及1rad/s。PPO n_steps128/batch64/epochs2/lr3e-4/gamma.99/GAE.95/clip.2/ent0/vf.5/max_grad_norm.5不变。随机Actor/CNN/Critic初态与R3一致，std初始.3并继续可学；不读旧checkpoint初始化。

唯一变量为目标距离课程：0–32767均匀12短目标；≥32768的下一自然reset恢复原12目标。旧episode目标不在中途改变，所有场景障碍/起点/Yaw原样，切换实际step记录而不冒称恰好32768。短目标manifest与原L6基准几何身份分开绑定。

## 【6】冻结协议

ARL_R6_DISTANCE_SEED0，seed0/CPU1worker/65536真实步/3600s。REG79ad97c769fb046efd1908ad73db945d437fd1ab3c7945b52162bff56a174fb8；SOURCEca83fb16b419ddcd4401249d9fbcd56b4e5577ba96625f792fff18803fd4218c；执行协议9132142a2e8e295bf877a38e3d466988e3925772f88817240b498dce1e6bfbce。

R5 proof395fe389…及公开report f97b5266…固定哈希，实际R5完整验证器和源文件仍须一致。R3 RANDOM control summary7651327c…完整及原物理/Reward/CNN源不变。checkpoint0/32768/65536，保存发生于完整rollout/对应optimizer更新后，step0为无更新初始化例外。模型选择Final。

训练关闭后新短Validation18条、原Validation12条，共30条实际新episode。原step0同初态policy的6条既有评价只读复用，标记reused，不计新episode。Test0，不用Validation结果改本轮协议。事前分别判断：短final成功≥4/6、碰撞≤1/6、physical/sensor失败0；原目标成功>历史0且碰撞≤6；短final成功数>自身step0。距离<.10m、速度<.15m/s连续5step标准不变。

最低Mem1536MiB/磁盘2GiB；低资源、NaN/Inf、未知终止、场景/源码/证据不符立即停止。唯一attempt，不自动重试，RESTART_REQUIRED。全部采样/optimizer更新按原授权计费，无免费训练测试。原账本不initialize/退款，不使用L9/Smoke/Pilot续训。

## 【7】实际结果与资源

运行前证据（历史快照）：6项profile与2项实际无更新接口RED→GREEN，1011项唯一相关测试PASS=105 OPS+21 Tensor+885历史，15排除；非全仓。2026-10-16机器重启使此前历史回归未生成final，明确INCOMPLETE，没有训练进程或预算事件。保留reboot_audit_01；新historical_restart01.stdout补做885项完整通过（121.10s/2既有warning），没有借用旧PASS。

独立预审GO：C0/I0/Minor1。Minor为旧历史runner自身只记录上游身份，不内嵌R6源码。当前新admission033c5342…同时绑定确切新final SHA、REG/SOURCE与执行前后源码核验receipt fcc8da88…，附带重启后实际新日志；这不是放宽保护或改名旧结果。该门控增强未在本轮改生产源码（如未来增强需新注册测试），如实保留Minor。先登记/recover/doctor、余额/stop/资源及prepareSHA核对后，才由原supervisor唯一预留和启动；实际运行结果未完成前不宣称学习通过。

**最终实际运行FAIL。** 上述准入链全部通过后唯一prepare/start，协议SHA实际9132142a…匹配；supervisor预留65536步/3600s，worker保存初始policy db33ba4d…和step0 ZIP后，在第一次物理transition返回时被旧TrainingMetricsEnv拒绝：`non-Train, unregistered or nonfinite PPO transition`。具体不是非有限值或非Train，而是新task identity不在旧base identity白名单。原始失败记录明确RETURNED_TRANSITION，policy_step_attempt=1、last_completed_policy_step=0，物理20tick/0.040000s；reward/action/qpos/qvel均有限，终止标志全false、LiDAR FRESH。

没有完整rollout、没有optimizer更新、没有完成episode、没有新Validation/Test。不能将预算预留65536误称实际完成65536；已完成训练记账步数0、一次实际尝试物理transition。初始化和checkpoint存在不构成训练成功。没有summary/complete/ppo_updates，不补写这些原实验文件，不重试ARL_R6_DISTANCE_SEED0，也不从step0继续。

supervisor实际wall3.215235589s；失败现场current RSS1053.921875MiB、MemAvailable10126.0625MiB，均非峰值，peak RSS/CPU总时长NOT_AVAILABLE。保存的当前速度.039287822m/s、目标误差.199997008m，只是第一帧，不能外推响应。未发现OOM证据，不能声称测得完整训练资源曲线。五个注册图明确标示未到达的训练/Val/PPO阶段，资源只有一个样本，目标只有reset+一帧，不画虚构学习曲线。

独立账本第11条RESERVE、第12条FAIL永久保留；本次预留不退款。累计保守393216步/21600s，余额1,606,784步/64,800s，stop=false，五旧attempt COMPLETE、新R6 FAIL。实际0次optimizer不自动退还授权步数/墙钟。原始失败/ZIP/日志local-only，逐文件封存hash，运行时源码保持ca83fb16…，尚未修复的R6旧源不能重新获准训练。

## 【8】问题分析

CONFIRMED根因：原TrainingMetricsEnv在ppo_training_orchestration.py检查info.scenario_identity_sha256是否属于self.env.by_identity；R5 scheduler保留这个字典为12个base身份，而返回的新短目标有独立task身份。实际返回2cb40770…，对应base adeeeb0d…在白名单、task不在，故安全拒绝。R6新增测试只覆盖scheduler/reset、初始化及独立Validation，没有穿过完整TrainingMetricsEnv→FormalHealthMetricsEnv→实际step链；独立预审也遗漏该消费者契约。这是已确认的实现/测试缺口，不是算法训练失败或Reward/LR证据。旧安全检查正确fail-closed，不应删除。

SUSPECTED：短距离课程仍可能提高成功反馈密度，本轮无法检验。UNVERIFIED：任何学习改进或迁移；不能从此实现异常判断PPO学不会，也不能因1011相关测试PASS宣称完整链已通过。

## 【9】局限

单seed/12Train/6Validation规模小，短目标往往在障碍前。工程PASS或短任务成功不代表避障、陌生矿井泛化、PX4或实机安全。没有真实MID360S时序/FAST-LIO2；optional cv2/NumPy ABI stderr及Axes3D warning保留，不系统升级。中断无精确续训。

## 【10】下一阶段

先将R6按FAIL保存证据/独立结果复审/Commit/Push远端核验，不能改本轮运行源码或报告成成功。随后进入独立Level1修复阶段：保留原12base和新12task的真实身份，显式受审的Train任务白名单适配，不删除原过滤、不伪造身份，不放入Val/Test。先以完整TrainingMetricsEnv→FormalHealthMetricsEnv真实一步及跨阈值/异常身份/失败锁存/Checkpoint场景状态做RED→GREEN；0optimizer。修复发布和预审后，才可在原余额内预注册新的独立学习实验，不重试本ID、不覆盖旧protocol/attempt。当前R6运行时代码仍有Important缺陷，训练准入BLOCKED；离线分析/修复在Level1授权内继续。

## 【11】给项目负责人的总结

这次确实按审查后的新协议启动了，但第一帧被“未登记任务身份”安全门挡住，因此还没做任何学习更新。问题是新课程任务和旧统计器之间没有完整对接，不是无人机动力学坏了，也不是PPO效果差。原失败会永久保留；我们不会偷改身份或关闭门禁来过关，而是先在独立阶段修复完整链并测试，再考虑新实验。预算依保守规则已经扣预留，仍余约160万步/18小时；普通离线修复继续，不等你再次下指令，旧R6绝不重跑。
