# AUTORL-R27：任务目标航向与到点停稳的受限对照

## 【1】阶段目标

只检验“同时要求转向Yaw0是否影响基本位置任务”的可反驳假设。复用R26已有两个step0原距离超时对照，新增两条保持初始航向的诊断，不重复旧episode。工程PASS；子集任务对照PASS；学习提升NOT TESTED。

## 【2】技术背景

R26已发布4310738b5f7f9472f321242690457e4d2bff561d，先验短2S、32k短1S1T，原距离两模型各2T。原距离先验曾进入10cm但未满足速度与连续停稳。这不能证明Reward或控制器是唯一原因；先隔离航向任务，避免盲目继续训练。

## 【3】系统架构

固定R26已发布证据→可信同一step0只读模型→同几何/起点/目标/初姿态→HoldInitialHeading wrapper→原环境真实MuJoCo/PI/24k扫描→两新trace→与两旧trace数值对照。仅CPU一个worker、两例各≤375步，总180秒；禁止learn/train/Adam.step，Test0。

## 【4】代码实现

新增goal_heading_diagnostic.py、goal_heading_trace.py、goal_heading_run.py及两份测试；原150 SOURCE不改。wrapper在原reset完成后仅设合法任务target_yaw为当前初始Yaw、重算7D当前state、同步metadata；不改qpos/qvel/PI/传感器/时钟。网络、动作、Reward、终止逻辑仍旧代码。分析逐行用真实pre四元数重算wrap(goalYaw−preYaw)、sin/cos heading、位置误差和世界速度，并独立复核d/v/tick/年龄/Reward/stop计数与初态/权重指纹。不是把旧零航向状态换个标签。

## 【5】判断原理

原成功标准d<.1m、v<.15m/s、连续5个25Hz样本不变；碰撞优先级及15s超时不变。仅在两条新任务均成功时heading_contrast为PASS，否则FAIL；单例改善不降低阈值。目标航向本来不在success与Reward里，它只改变合法观测及策略动作，这是显式任务变量，不是飞行安全门认证。

## 【6】协议及身份

REG `88167a3f0b25436c353d7319b8a4f1b0053e4e4126be3c5ee31c602b5c478a7c`；155SOURCE `372e18c77a389658b49c5e911d8215d3cfcb1f6df81c5437e8e42957ceaa39b6`；独立L1协议 `caa53b1f3419c43d359eac1831ba383cb7cd3b456e26850ffcc5a801cf0c4d7c`。相同初始先验650a5841…，可信原R24 step0 ZIP只读加载env=None；旧API名WARM_START不表示训练初始化/续训。SIM_GROUND_TRUTH_STATE/世界ENU/freejoint，不声称SLAM/PX4/COM标定。500/100/25/10Hz、24000rays、原7kg模型及Reward不变。冻结新任务身份包含原场景与goalYaw，旧manifest与R26结果不改。

## 【7】真实结果

| 空场景 | 原身份前缀 | 已有R26控制 | 新R27任务 | 新策略步 | 最终误差m | 最小误差m | 最大连续joint |
|---|---|---|---|---|---|---|---|
|0|fe85ac443a0d|time_limit|success|193|0.061768|0.002102|5|
|1|0936e963da20|time_limit|success|193|0.063828|0.002373|5|

新2episode/386策略步/7720实际tick；对照2episode为原R26已有数据，不当新增运行。判据PASS，新成功2/2，旧控制0/2。并非原任务导航通过，更非PPO学会避障。完整1199选择回归PASS=238OPS+76无更新Tensor+885历史、15排除、optimizer0；新7契约及1真实wrapper reset/单step测试RED→GREEN，另3纯算术RED→GREEN及4继承发布/受控子进程契约GREEN，不冒充科学episode。

内层wall9.531s、CPU9.686s；launcher wall17.358s，workerHWM1527.695MiB、含reaped后代HWM1527.695MiB；guard采样最低MemAvailable9037.219MiB，不是导入/加载全程连续峰值。363授权文件、42额外历史与旧150源码字节不变，预算仍1016960步/32400预留秒、stopfalse，无训练预算事件或观测OOM。

## 【8】问题分析

CONFIRMED为表中同先验、改变任务航向的两例结果及相同初态/实际字段。SUSPECTED的Yaw旋转耦合或控制反馈机制不能仅凭本实验确认；没有证明控制器实现错误，也不据此修改PI/PID。若无改善，同样保存FAIL并排除“仅改目标航向就足够”的强假设，不无限重跑。

## 【9】局限性

目标任务已改变，不能把新成功计入原固定Yaw0任务。只两个empty Validation、单先验，没有障碍、多seed、训练提升、真实设备或封存Test。现有对照与新wrapper路径并非同一源码版本，但旧150 SOURCE、几何、初始化和参数均逐SHA保持；不声称严格bitwise跨版本同程序对照。原R24仍永久FAIL/科学NOT TESTED，RESTART_REQUIRED，不能续训旧ZIP。

## 【10】下一阶段

实际结果复审、显式19文件Commit、正常OPSsync核远端后，检查余额/stop/资源；依据新证据决定是否值得研究任务航向课程或继续反馈诊断。任何新PPO实验必须独立协议、失败闭环准入、测试及GO，旧训练gate仍closed，不能靠本报告放行。普通阶段结束后继续已授权的安全研究。

## 【11】给项目负责人的总结

这次没有训练，只问一个具体问题：无人机先不转头、保持出发时朝向去同一目标，是否更容易停稳？结果新2/2，原有对照0/2，冻结判据PASS。这只能说明该简单子集的任务差异，不能当成网络已经会导航。预算没花，模型和老失败实验都没有覆盖；后续依据证据选择一个新变量，不盲目调Reward或继续百万步。

最终独立结果复审GO，C0/I0/新增Minor0；只认证固定2个新改变任务航向诊断及2个已有对照和原始数值，不宣称原任务或学习成功、不补R24完整结论或授权训练/ZIP续训。JSON仅追加review/recovery/doctor与audit引用，统计和图不变。19文件显式白名单Commit/正常OPSsync远端状态以实际收据为准，不能由本段预先认定已推送。
