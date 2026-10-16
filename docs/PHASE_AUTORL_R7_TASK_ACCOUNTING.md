# AUTORL-R7：距离课程任务身份记账修复

## 【1】目标

只修复新距离任务与旧训练统计器的身份接口，0optimizer更新。R6首个transition被拒绝的事实及失败源码已推送2f6e5c297c2a12f7e69a8328d90aee11eed97589，永久FAIL，不重试/续训。

## 【2】背景

R5的scheduler以原12个base身份抽样，却正确返回独立short task身份。旧TrainingMetricsEnv要求返回ID属于其env.by_identity，导致R6首个20tick物理transition被拒绝。旧过滤正确fail-closed；不能删掉或将新目标伪装成旧目标。

## 【3】架构

原NavigationEnv/真实MuJoCo→未修改R5距离scheduler（12base抽样）→新增DistanceTaskAccountingAdapter（24Train任务成员视图）→未修改TrainingMetricsEnv→未修改FormalHealthMetricsEnv→原Checkpoint。活跃MuJoCo仍最多一个。没有PX4/FASTLIO/实机/Test执行。

## 【4】代码

新增research_regression/authorized_rl/task_accounting.py、task_accounting_tests.py；设计见AUTORL_R7_ACCOUNTING_DESIGN.md。外层白名单12short+12original，只读且仅用于记账成员判断（值不是reset行表），内层by_identity仍原base行表。返回task/base/stage/target保持真实；实际active_task/plant目标/几何与冻结清单逐次核对。非法reset仅拒绝，不静默改目标，任何异常锁存。外层current_identity明确为当前task，base_current_identity为inner base；未reset或已close的task为None，故障与trace不会把新目标伪装成base。sidecar同样明确两个身份作用域。没有改worker或旧schema6，因此旧R6依旧失败；新helper不能自行启动训练。

## 【5】算法与安全语义

不修改CNN、PPO、Reward、动作尺度、控制器、传感器或500/100/25/10Hz。距离阶段从内层active_stage读取，不根据已经跨阈值的计数中途改episode目标。Checkpoint保留L6base manifest和额外24task白名单/短manifest/当前任务，既不混同几何与任务，也不声称EXACT_RESUME。异常适配器没有将transition返回给上层时，原记录器诚实写LAST_KNOWN_CONTEXT，未知info/reward为NOT_AVAILABLE；物理时间/state仍可取得，不捏造完整transition。

## 【6】协议及身份

本阶段Level1离线修复，训练预算不消耗，原grant剩1,606,784步/64,800s。最终候选_02 REG486b2730207cdc96ea231cd845a433ff1eea7b1cb44537e2aa0c1beefe1deb16；SOURCE1eaa6b1f515ab32bce7b5ca15d1f1a5889ceb9505317fa6da3752794962031fb。新OPS登记不改V1/V2历史零预算或独立grant。R6原目录10文件及整个授权目录执行前后SHA逐一比较，原模型与七历史checkpoint由L6/OPS核验。

## 【7】结果

最终_02选择回归1022项PASS=105 OPS+32 Tensor+885历史，0fail/error/optimizer；15历史排除/2既有warning，非全仓。首个真实物理完整链RED复现旧RuntimeError；新adapter后该步正常记账为1step/20tick，新增11项定向测试包括12任务真实step覆盖、同状态/同动作轨迹/观测/Reward与RNG对照、FORCED_TEST_COUNTER跨32768、Checkpoint零更新保存重载、错误身份/目标拒绝和锁存、五类终止/四种physical原始reason、故障真实task与close身份语义。强制状态不是自然训练现象。原始旧链仍拒绝且保留失败现场。

首次8项验证7通过/1测试比较数组抛ValueError，保留green_01原日志；修正测试断言为所需reward/terminal/breakdown与obs/qpos逐项比较，未为此改生产adapter。_01独立NO_GO I1指出故障只保留base，随后两项真实RED断言失败→GREEN，补充task/base外部契约及close语义。不借用_01的105/31/885 PASS准入新源码；_02完整105/32/885及复审重新执行。技术性旧Validation短probe不当作新学习效果，Test episode0。

## 【8】问题分析

CONFIRMED：R6错误来自base与task身份契约，新增显式边界解决有限真实step证据。SUSPECTED：此前测试漏过完整消费者是接口覆盖不足，不是测试数量不足。UNVERIFIED：PPO学习改善、长期稳定性，本阶段没有训练。未验证任何新训练入口，不将修复通过当作新attempt已执行。

## 【9】局限

成员视图不是原reset行表，调用者应按明确接口使用。独立审查记录一个非阻塞Minor：若任意内存破坏令内部active_task成为非dict/缺失identity，其只读current_identity可能抛错并遮蔽旧记录器；正常冻结scheduler没有产生该结构的路径，本阶段不宣称任意内存损坏证据恢复。旧运行协议不更改；无精确恢复，仍RESTART_REQUIRED。cv2/NumPy optional import stderr、历史warning保持，不升级系统依赖。测试只选定范围，不是全仓PASS；旧排除逐项保留。

## 【10】下一阶段

最终当前源码测试/独立GO/新登记/历史审计/报告/Commit/Push远端完成后，才冻结新的独立修复后课程学习实验（候选R8，新ID、新schema及精确R7发布证据）。原R6不重试、不加载step0继续，不改变Reward/算法来掩盖接口Bug。剩余授权足够并且无停止/资源/未知状态时自主继续，不按普通阶段结束长期任务。

## 【11】项目负责人中文总结

我们保留了安全检查并把“场景”和“目标任务”分别登记，不再用旧场景身份掩盖新的课程目标。有限真实物理测试已走过之前遗漏的完整统计链，Reward和飞行控制没有变化；这不是算法学会导航。本阶段不花训练预算，独立审查/完整回归通过并发布后才能开始另一个获授权的新实验。旧失败永远保留。
