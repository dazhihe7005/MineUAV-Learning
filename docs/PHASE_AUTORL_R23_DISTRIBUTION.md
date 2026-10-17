# AUTORL-R23：Train/Validation分布只读诊断

## 【1】阶段目标

解释“Train有成功、独立开发验证仍失败”可观测的分布差异。统计实现PASS，未训练、未检验因果，不宣称算法性能提升。

## 【2】技术背景

R22已完整发布55594504a1e906940c4f1b7c5668de79ee8291be。λ.99相对完整R17的五判据全部FAIL；final原域2成功4碰撞，两个empty均撞边界x正墙。原L9/R20的失败永久保留。不能把一次Val改善掩盖短域遗忘或碰撞增加。

## 【3】系统架构

固定R17/R22发布报告与raw→身份/完整性核验→标准库圆周角距及逐身份统计→真实数据图→报告。无模型/物理/训练入口，Test episode0。

## 【4】代码实现

distribution_audit.py核对split、SHA身份、结束episode与Val覆盖，拒绝短域混入原域或错误分母；distribution_audit_run.py使用既有flock/owned group/父守护/120s/资源门。唯一结果原子发布，344份授权文件含head及所有events/run前后逐字节不变，不新增训练预留。

## 【5】算法原理

Yaw最短圆周距离=|atan2(sin(a−b),cos(a−b))|，避免179°与−179°被误算358°。nearest same-layout只说明初始朝向覆盖，不能等价为训练全过程状态覆盖。结束episode比例与policy步曝光分母独立，未结束尾部不补造终止/Reward。随机且持续变化策略的Train统计不能与final deterministic Val当成同策略泛化差值。

## 【6】冻结协议

AUTORL_R23_SPLIT_DISTRIBUTION_ONLY；训练/optimizer/新物理/扫描/推理/Train/Val/Test均0，单CPU算术worker/120s。REG `7879e9400486c234ae5257147d7e64f51a6193ad767c1c1be457f8e6f7638390`，SOURCE `336f98c8dbc76b4fc812e3e75281f93564a3ad4d875d7a4084ca11920241792a`，EXEC `7c5b59ba1ff7133541440690a4c6c0230fe759fc7e32d58ad1fae36c95ca7586`。固定12Train与6Validation；未读取Test episode结果。未改物理、PI、Reward、CNN、PPO、动作/传感器、原清单或历史checkpoint。

## 【7】真实结果

| Validation任务 | 初始Yaw° | 离全部Train最近° | 离同layout Train最近° |
|---|---:|---:|---:|
|empty / fe85ac4|-160.981|57.577|93.594|
|front_box / 80f560c|173.335|40.256|40.256|
|side_pair / f07b571|34.159|4.500|91.349|
|detour / 6587c37|77.490|9.352|84.503|
|channel / 919918b|61.364|6.775|6.775|
|empty / 0936e96|-145.117|41.713|77.729|

|历史run/layout|结束原域Train episode|结束原域policy步|Train成功/碰撞/超时|final Val成功/碰撞/超时|
|---|---:|---:|---:|---:|
|R17/empty|76|27044|19/0/57|0/0/2|
|R17/front_box|29|2033|0/29/0|0/1/0|
|R17/side_pair|9|2003|1/5/3|0/0/1|
|R17/detour|16|731|0/16/0|0/1/0|
|R17/channel|13|637|0/13/0|0/1/0|
|R22/empty|78|24522|57/0/21|0/2/0|
|R22/front_box|28|4396|2/18/8|0/1/0|
|R22/side_pair|12|2145|1/8/3|1/0/0|
|R22/detour|15|606|0/15/0|1/0/0|
|R22/channel|16|900|0/16/0|0/1/0|

这两组Train/Val列不是公平同策略差值，只是来源明确的描述。原始数据包含各身份完整计数和两个empty Val接触对；未结束尾R17 238步、R22 85步单列。

6项新增真实RED→GREEN；217OPS+70无更新Tensor+885历史=1172选择PASS，15排除，非全仓。预审发现角距缺少rad字段、Val矛盾outcome未拒绝，新增断言实际2项RED后修复为GREEN；仍是原6项测试内扩展，不谎称8项新测试。首轮1172PASS收据保留但绑定旧源码，最终SOURCE02重新完整执行1172，不复用旧准入。新6项不执行优化器；旧可选cv2/NumPyABI与Axes3D警告保留，不系统升级。worker算术wall 0.190s、CPU 0.188s、HWM 64.043MiB、离散资源采样最低MemAvailable 9445.031MiB，不冒充连续系统峰值/最低。预算仍1082496步/36000秒预留wall、stop=false。

## 【8】问题分析

CONFIRMED：训练与开发验证初始Yaw存在真实差异，同layout empty覆盖尤其不同；实际终止按身份与layout如表。SUSPECTED：原训练Yaw覆盖不足可能导致背景/姿态条件过拟合。UNVERIFIED：Yaw是根因、碰撞前具体动作、完整trajectory/advantage、扩展Yaw是否改善。layout、位置、障碍变化与Yaw共变，不能靠相关性完成归因。

## 【9】局限性

单seed、小开发Val，未检验Test/真实矿井/噪声/PX4。无新物理轨迹，不能判断是否绕到侧面、真实LiDAR因果使用或停稳反馈机制。原样本中的初始Yaw不是visited heading分布。过去R7/R18/R20 Minor仍保留。

## 【10】下一阶段

先完成独立结果复审、真实Commit/Push远端核验及预算/stop检查，再冻结新独立训练候选：以完整R17为控制，只改变Train初始Yaw分布为预定均匀圆周覆盖，保持几何/start/target/Reward/CNN/λ.95等不变；不从R22权重继续，不把回到R17配置描述为相对R22单变量。方向独立于Test，不能精确抄Val角度。新manifest/测试/审查/余额核验通过后才允许新预留，不自动重试任何旧attempt。

## 【11】给项目负责人的总结

这轮没花训练预算。我们确认训练任务和验证任务的起始朝向并不一样，尤其空场景；这给了一个比继续增加步数更具体的假设。但现在不能说“就是Yaw造成的”，更不能说算法已经变好。下一步优先在原有控制实验基础上，仅扩大训练时的朝向覆盖，看看独立验证能否改善，同时保留成功、碰撞和超时三类结果。授权余额不增加，设备和PX4不运行。

2026-10-17恢复附记：R22/R23在前次关机前已经完整结束，本次未重跑。当前boot的recover/doctor及344授权文件、head28、余额1082496/36000、stop=false核验PASS；原实验日期仍为16日。最终独立只读结果审查C0/I0/新Minor0，真实OPS后置审计与开机恢复收据已追加JSON，统计未改；封口后的report/本Markdown SHA由实际staging收据重新读取绑定，旧被审SHA不冒充新文件身份。Commit/Push状态只以真实同步收据为准。
