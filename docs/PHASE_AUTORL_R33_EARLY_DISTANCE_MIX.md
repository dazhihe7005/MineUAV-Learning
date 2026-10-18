# AUTORL-R33：早期距离混合学习对照——重启中断

## 【1】阶段目标

检验更早加入原距离任务是否减少R28原距离行为遗忘。唯一训练跨启动中断：执行FAIL/INTERRUPTED，完整科学验收NOT TESTED。不是重跑R28，也不使用R30/R31动作修正。

## 【2】技术背景

R28纯短课程32k时原距离Validation已从手写先验4/6降为0/6，最终仍0/6。R30/R31动作诊断均0/2；R32只读审计已发布4e39632430730bbb4670770ca7ce7c2754e142d4，没有证明控制器Bug或唯一根因。本轮只改变课程距离分布。

## 【3】系统架构

独立distance RNG→自然reset选SHORT/ORIGINAL→冻结Train/heldgoal→原MuJoCo、LiDAR和CNN-PPO。计划训练完后，三个checkpoint各执行SHORT/ORIGINAL六例，共36新Validation。实际在评价之前中断。

恢复采用进程/锁/启动身份审计→独立处置审查→BudgetStore.fail追加事件。没有执行原COMPLETE分析器，没有在旧run目录补写完成文件。

## 【4】代码实现

新增early_distance_mix/env/profile/cli/tests：分布、reset级目标选择、固定协议、显式准入和无更新回归。budget/runtime/worker仅增加schema33分派及复用R28初始化/记账。显式组合调度器，无全局monkeypatch，至多一个活动场景。

9个标准库与4个Tensor新增测试真实RED→GREEN；旧worker误落随机初始化而非fresh650a的问题在启动前修复。启动后187项运行/测试源码未改。中断后的local-only审计、绘图和封口程序不是原运行时异常捕获；路径与SHA在JSON中。

## 【5】算法原理

前32768步每次自然reset独立draw<.5选ORIGINAL，否则SHORT；base均匀12项。distance RNG为default_rng(seed+330000)，不消耗base RNG。后半自然reset只选ORIGINAL，empty总概率.5，与R28一致；不在episode中途改目标。每reset各.5不等于每step各.5。

Reward保持：10Δd−.005||a||²−.5w(d)||v||²+10success−30collision/physical−10sensor−.01step−2timeout。成功仍为距离<.1m、速度<.15m/s连续五个策略样本。保留手写先验成功不等于学会避障。

## 【6】冻结实验协议

ID ARL_R33_EARLY_DISTANCE_MIX_SEED0，CPU1/seed0，唯一65536步/3600秒预留。fresh650a，不从ZIP初始化；CNN170857参数、NAV_V2_COLLISION_PREFLIGHT_V1、PPO LR3e-5/γ.99/λ.95/128-64-2、动作尺度、PI、7kg、500/100/25/10Hz、24000rays不变。

- REG：21591d609618e860d7f2ee001b0790a66af81332b477473584e586aacb001d17。
- 187SOURCE：0dae1070650a0ef14ebe4201fb43093c3468d7c561a0e24088c61819a15e7066。
- EXEC：f14fbcea537e4ae55cd73712bed3eb3f4b4a9f97d50253c4cc6a32f80388e686。
- held manifest：3a670bbbc1b841fc81efbc2b6f81f3a58e4d84583a5f5cebaa61820f0e8007b3。

六个原判据保留：相对R28 final改善、相对自身prior改善、original≥4成功且≤1碰撞、empty32k至少1且final不下降、short≥4且≤1碰撞、short保留prior；全部36例PF/SF0。本次六项均NOT TESTED，不降低阈值。

## 【7】实际结果、测试与资源

旧boot189bfa8c-5d0c-4857-82e2-b221364d91d7，新bootd160d566-c4a7-4304-989f-176dc983dc5a。669条精确旧boot日志含2026-10-17 19:43:37+08:00 System Reboot/Shutting down/Sending SIGTERM。具体worker信号、退出码、重启发起者与原因未知。旧进程/进程组/锁已消失。

| 证据 | 实际结果及边界 |
|---|---|
| 完整更新 | 385rollout；49280步；1540optimizer step；770epoch |
| 已结束Train | 343个，最后49384步；是已采样下界，不是精确停止步数 |
| SHORT | 135成功，0碰撞，0超时；已闭合10156策略步 |
| ORIGINAL | 96成功，84碰撞，28超时；已闭合39228策略步 |
| PF/SF | 已结束前缀均0；不据此排除未记录最后transition异常 |
| 新Validation/Test | 0/计划36；Test0 |
| checkpoint | 0/32768存在且CRC/SHA/元数据通过；65536最终ZIP缺失 |
| 完成证据 | summary/complete/主管终态均缺失，不补造 |

32768 sidecar的262个实际reset与261个结束episode核验；之后无最终调度器快照，不能声称完整reset序列已测得。所有343个结束任务的元数据与冻结清单另行核验。旧formal_progress49152比最后更新日志更旧。

启动前1258选择PASS＝280OPS+93无更新Tensor+885历史，15排除，不是全仓。重启后280OPS实际267PASS/1FAIL/12ERROR、零optimizer：旧正例读取实时账本，pending R33被正常门禁拒绝，一个断言预期了更晚的forged-balance原因。全部失败节点与异常保留在JSON。93Tensor/885历史未再次运行；不把旧PASS冒充重启后全绿。另4项纯记账测试真实RED→GREEN，拒绝序列矛盾、缺少更新及混淆精确最终步数。

合法追加34FAIL后又执行同一280项选择，实际276PASS/1FAIL/3ERROR：仅拒绝pending的检查恢复，其余四个旧正例仍错误地期待旧准入门接受新失败ID。两次失败证据分别保留，当前不能称全绿；下一阶段修测试快照而不是关闭安全检查。

最后资源采样wall1422.254s、CPU1317.690s、前缀RSS高水位1667.594MiB；累计采样最低MemAvailable8322.4453125MiB，磁盘约64.72GiB。完整wall/全程峰值/精确最终物理tick未知，不宣称排除全程OOM。可选cv2/NumPy ABI及Axes3D警告保留，没有升级依赖。

547项旧保护文件、16项本次raw不变；独立GO后仅追加34FAIL，保留65536步/3600秒全额收费。余额885888步/25200预留秒stopfalse。五图见mujoco/reports/autonomous_rl_early_distance_r33/figures：episodes、validation、ppo_updates、goals、resources；Validation图明确标缺测，不画虚构曲线。所有raw、journal和ZIP local-only，不等于远程备份。

## 【8】问题分析

CONFIRMED：跨boot重启、日志中断、无最终评价，旧正例耦合实时预算状态。SUSPECTED：系统关闭打断主管清理。UNVERIFIED：具体进程退出原因、最后transition、课程最终效果、完整reset分布。231个Train成功不能代替独立Validation，不能直接与完整R28终点比优劣。

## 【9】局限性

单seed/小型开发场景，不证明LiDAR依赖、真实矿井泛化或部署。RESTART_REQUIRED，不ZIP续训、不重试旧ID、不退款。SIGTERM处理不保证断电/系统重启时完成，没有无损恢复。原L9等失败记录与旧授权不变。训练源码身份和完整实验完成是不同验收。

## 【10】下一安全阶段

先独立结果复审、26文件白名单Commit、正常OPSsync并核远端。随后Level0/1处理旧测试的历史快照隔离，审计R33永久失败封口证明。新训练必须另有协议、测试、独立GO；不能直接重启R33或放宽旧门禁。原授权仍有效，但当前新训练准入NO；普通失败阶段发布后继续安全研发。

## 【11】给项目负责人的总结

这次在系统重启时被打断，不是已经跑完后证明算法失败。中间有231次到达停稳，但没有最终独立评价，不能说学会避障。原始证据和预算保留，不偷偷重跑。先处理干净中断与测试问题，再用原授权余额推进独立研究。目前没有后台训练。

最终独立只读复审 GO_REPORT_ONLY，C0/I0。仅同意如实发布中断FAIL，不放行训练、不豁免重启后回归失败；统计、源码及原raw未改。Commit与远端身份以实际OPS收据为准。
