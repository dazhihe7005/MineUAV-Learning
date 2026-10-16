# AUTORL-R15：新中心化权重反馈诊断设计

## 目标与依据

R14已完整发布1ab46a3d0c857ded3c0f496db0e54261a49b3005。新Actor短Val为6→6→5成功，原目标始终1成功3碰撞2超时，科学总FAIL；原Train814个近目标采样有779个仍速度过大。需要区分固定输入反馈形状与物理执行，不能凭“中心化使零state归零”宣称已学会制动或导航。

这是复用现有feedback_spec/feedback_probe/CheckpointManager的有界扩展，不开发新扫描/训练系统。根据用户明确长期自主授权，不再索取普通阶段重复确认，但实际执行仍须独立GO及精确OPS新登记。

## 备选与选择

1. 推荐：先读新R14 32k/final模型，固定旧R4背景与62输入，124新推理；复用R13实际同新类初态62行，不重新推理旧面板。零optimizer，低成本区分常量偏置、位置/速度/Yaw响应。
2. 直接减小探索std：会耗训练预算，暂无法确定确定性均值反馈是否合理，因此不先执行。
3. 直接加制动先验：同样需先核对当前速度反馈，避免凭直觉增加另一个变量。

本轮仅选1。独立实验ID AUTORL_R15_CENTERED_FEEDBACK_ONLY，CPU1、120s。两个完整R14 checkpoint只读推理，不作为训练初始化/续训。禁止参数、optimizer、历史raw、预算账本变化；Test/PX4/FASTLIO/硬件/新物理均0。

## 冻结接口与判据

复用原62个OFFLINE_COUNTERFACTUAL_INPUT，历史LiDAR/heading/age完全固定。输入是反事实状态，不是新轨迹或新的传感器采样。R13 new_policy_panel为同一初态函数参考；R14初态policy SHA20f5c226…与其一致。R14实际发布报告、两个ZIP/sidecar/summary、R13报告/result/背景、真实审查收据全部逐项校验，不将缺失或损坏历史重新当baseline。

输出raw mean/std/value/clip后动作及声明ENU速度尺度，权重/optimizer/RNG每次panel前后相同。每panel仅使用当前R14真实中心化类，verify_centered_policy核网络/对象/LR；PPO.learn/train及Adam.step硬拒绝。

先验保留诊断沿用既定R12数值：zero-state速度范数≤0.05m/s，整个panel相对初态normalized action最大差≤0.05，三轴非零误差零速度样本方向全部正确，正负速度响应差相对初态0的绝对误差≤0.05m/s，Yaw ±0.5rad响应差严格正。原先验没有负速度反馈，不能错误要求严格负才能保留。单独报告速度响应符号/幅值，不把先验保留PASS等同学习/停稳PASS；结构零state不算科学成绩。

## 实现与验证顺序

1. 新纯profile/引用门控与两项历史数组测试先RED；明确来源、检查缺失、历史变异、未知预算attempt、CLI任意参数拒绝。
2. 最小新centered_feedback.py/centered_feedback_run.py/测试实现，旧runtime/worker/Actor/Reward/模型不改。
3. 当前完整OPS+Tensor+885历史无optimizer回归，15既有排除逐项沿用；所有收据绑定新SOURCE/REG。
4. 独立只读预审精确GO→新OPS register/recover/doctor→全授权文件/停止/资源核验→新独立owned process group唯一run。attempt存在永久拒绝重放。
5. 真实结果与独立反事实分析、实际三轴/速度/Yaw图、中文阶段MD/JSON与五长期文档；结果独立复审→显式Commit→正常OPS sync远端核验。
6. 根据结果决定下一单变量候选；若诊断不足如实UNVERIFIED，不为消耗预算而训练。普通阶段完成不结束长期研究。

## 风险与边界

单一历史背景不能证明LiDAR依赖性、动态稳定性、泛化或真实电机/飞控行为。固定点响应不同也不单独证明Bug。当前预算余额1344640step/50400s，R15不得新增预留或修改任何授权文件；L9永久FAIL、旧OPS零维护预算及L9R false保持历史身份。
