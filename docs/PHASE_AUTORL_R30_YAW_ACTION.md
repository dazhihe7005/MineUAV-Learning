# AUTORL-R30：Yaw动作单变量消融

## 【1】阶段目标

检验R28最终模型的Yaw动作耦合是否参与两个原距离empty任务不能停稳；独立非学习、改动作诊断，不是新策略训练。

## 【2】技术背景

R29已发布c3b7197ed715638db7c1d18c962fe28f7b41ba1a。静态函数probe未支持远距离指令塌缩；训练后存在跨轴/Yaw输出但闭环因果未知。旧R28final两个empty均375步超时、终点约.062/.066m，不能因进入位置球改判成功。这里只复用旧两条汇总为既有控制，不重复未修改模型Val，不倒造其逐步轨迹。

## 【3】系统架构

可信final ZIP只读load → 当前真实观测 → 原模型deterministic推理 → 仅将已clip动作index3设0 → 原PI及MuJoCo真实mj_step/LiDAR → 实际逐步记录与分析。新轨迹KINEMATIC_POSE_REPLAY=NO；不直接改qpos运动（仅原reset初始化），无PX4。

## 【4】代码实现

yaw_action_diagnostic.py冻结病例/控制与双成功判据；yaw_action_trace.py显式ZeroAppliedYaw代理，不改policy，复用原goal_heading_trace.collect；每step记录原mean、原clip、实际applied、速度指令、qpos/qvel、PI积分、LiDAR时间及Reward/终止。yaw_action_run.py核OPS、源码、协议、资源、flock、唯一attempt与生命周期守护，无训练入口。4标准库＋2Tensor实际RED→GREEN，单20tick物理接口不算2科学episode。首个GREEN01名字的候选实际上3ERROR（event字段误用），保留并说明；核真实weight_sha256与validation.policysha后GREEN02通过。

## 【5】原理与边界

单变量是实际YawRate动作置0，不是任务目标Yaw设0：目标仍保持初始ENU yaw，原PI reset以初始yaw起步；不改控制器PID、积分规则、Reward或动力学。同一个输入下原三维平移动作逐值相同；干预之后轨迹和后续观测可以不同，不能说两轨迹仍同输入。动作消融成功只支持Yaw动作在该对子中的作用，不代表原网络自动学好，也不证明唯一根因。

## 【6】冻结协议

REG `866768cd558c7036644392569cc9df654a68ecfea6f2ff1245dc35d979195090`；173SOURCE `302b82788b90fc538146f51f1367e43f5fcfcf58a64dad8e816b57f4545c07f9`；EXEC `e5f4cd89e0b3aae7b0885d8dca339ea5c37047454fa50e702f2fe8ef8c1f1d63`。可信final65536模型参数SHA `f7349c40158b1f064542bdf8180f22cf9b68f11aee9ab563f732e678bfc85475`，2个held原距离empty Validation，每例≤375step/内部180秒/CPU1；复用原final2控制，Train/optimizer/Test=0，无旧ZIP续训。500Hz物理/100HzPI/25Hz策略/10HzLiDAR/24k射线不变。成功必须距离<.10m、速度<.15m/s连续5step，无碰撞或物理/传感器失败；双例均成功才contrast PASS，不能事后放宽。

## 【7】实际结果

工程PASS；冻结Yaw动作对照判据 **FAIL**。两个新的显式改动作episode，合计750策略step/15000实际物理tick，optimizer/训练/Test=0；原R28科学FAIL保持。原控制0成功，两新例0成功。

|新例终止|策略步|实际tick|终点目标误差m|终点速度m/s|最长联合停稳样本|路径m|
|---|---|---|---|---|---|---|
|time_limit|375|7500|0.062229606337647|0.3451609888658413|0|6.70402444434704|
|time_limit|375|7500|0.06698574586618274|0.34950507717261786|0|6.905310552110893|

逐行独立复核：初始pose/yaw、pre/post状态连续性、state/heading观测、非末行无terminal且20tick、时间与.002s物理步长、实际步数、qpos目标距离、qvel速度、Reward分项与总项、原始mean→clip与实际仅Yaw清零、实际速度映射、LiDAR因果/新鲜度及完整连续停稳计数一致。case持久化与result一致，两case之和等于全局步数/tick。没有将旧控制轨迹画成真实已有trace。三张图来自这两条实际新轨迹。

独立预审曾发现本地后处理Important：初版验证遗漏上述多项绑定，因此未获准运行。用真实旧R27数值复制并显式改装unit-only动作metadata验证旧分析循环，13个篡改被错误接受，另2个本来已被拒绝；原始旧trace不改，这不是新物理实验。复用固定SHA的R26/R27纯算术验证器并补完整动作/状态/持久化与总量检查后，2个本地测试覆盖15种trace和4种bundle篡改均GREEN，另计、不混入1234。第一次legacy_RED01因unitfixture保留极小原Yaw命令而不满足新格式，失败收据保留，实际缺陷复现用RED02；全部修复限本地分析helper，173运行SOURCE与协议不变。最终准入以修复后独立GO为准。

worker墙钟17.157s、CPU17.318s、生命周期HWM1422.707MiB，逐策略guard采样最低可用内存8902.398MiB；outer launcher24.617s单列，180秒inner预算/190秒外部清理不混淆。无已观测OOM/超时/残组，不能证明无限长运行内存稳定。

1234选择PASS=262OPS+87无更新Tensor+885历史，15既定排除，非全仓；新增6项，测试optimizer0。483旧保护文件/原168SOURCE/账本head32不变，余额951424步/28800预留秒stopfalse。

## 【8】问题分析

CONFIRMED：真实动作只改Yaw，真实物理状态与终止可追溯，冻结对照结果FAIL。若FAIL，必须记录“仅清Yaw不足以满足双成功判据”的反证，不能泛化Yaw完全无影响。若PASS，也只能说该两病例修改动作改善，不是原learned policy完成任务。SUSPECTED：与平移反馈、近目标速度或其他耦合的关系待证。UNVERIFIED：唯一根因、其他场景、LiDAR依赖、跨seed及实机能力。

## 【9】局限性

单seed/两empty/明显的执行动作干预；不验收避障、无GNSS飞行或真实设备。旧控制无逐step trace，不根据末点倒造振荡/速度。旧模型参数及检查点不改，历史L9/R24永久FAIL不变，当前仍RESTART_REQUIRED、不声称精确恢复。caught cv2/NumPyABI和Axes3D警告保留，无系统升级或swap修改。

## 【10】下一阶段

先结果独立审查、18文件真实Git检查/Commit/正常OPSsync与远端，未齐不得phase_complete。再核budget/stop/resources，按实际对照及trace选下一可反驳假设，可能更直接检查近目标平移反馈；不重复该attempt/原R28Val、不读Test、不盲目加训练预算。用户长期原授权继续有效，但不扩大权限。

## 【11】给项目负责人的总结

这轮只把最终网络指令中的转向速度设0，其他动作继续来自网络，让真实MuJoCo验证两次能否停稳。实际对照结论FAIL，不是重新训练，也不能称原模型学会自主导航。没有花新的训练步数，旧负结果完整保留；下一步由真实轨迹决定，不用高Reward或接近目标的位置代替停稳成功。

最终独立只读结果复审GO：C0/I0/newMinor0。只认证真实两例改动作诊断与科学FAIL边界，不将原R28改判成功，不授权新实验。18文件真实Git检查/Commit/正常OPSsync及远端以实际收据为准。
