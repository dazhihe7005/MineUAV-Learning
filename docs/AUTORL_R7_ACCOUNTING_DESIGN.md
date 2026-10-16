# R7：有界身份记账修复设计

基线2f6e5c297c2a12f7e69a8328d90aee11eed97589；R6唯一attempt永久FAIL，原目录/protocol/SOURCE/账本只读。剩余1,606,784步/64,800s，不initialize、不退款，本阶段0optimizer/训练/Validation/Test。

## 意图与选择

已确认新short task ID不在旧12base白名单，第一物理transition被拒绝。修复只针对消费者接口，不改变R5任务、原NavigationEnv、Reward、PI、动力学、CNN/PPO、安全过滤。备选一是把task伪装base（拒绝：错误身份）；备选二是重写TrainingMetricsEnv（拒绝：扩大历史源码变化）；采用新增显式Gym适配器。

内层仍是已验证R5 scheduler，均匀抽样原12base，不同时保留多个MuJoCo；外层给旧记账器暴露冻结24个Train任务（12short+12original）只读白名单。reset只允许原base覆盖身份；返回保持真实双身份，校验current base、active task、stage、目标、障碍hash与冻结清单/实际plant一致。不准入Validation/Test/unknown。异常锁存，原故障记录器诚实保留可取得的context，不伪造没有返回的info/reward。

## 验证顺序

1. 真实Navigation→scheduler→TrainingMetricsEnv→FormalHealthMetricsEnv第一step测试，原实现先RED复现；learn/train/Adam.step tripwire。
2. 新adapter GREEN；与同seed无包装scheduler同动作逐项比较obs/reward/终止/state，确认只变身份接入。
3. 完整12场景一步覆盖、强制计数跨32768（FORCED_TEST_COUNTER，非已训练）只在reset换目标、PI/cache重置和RNG不变。
4. 五类终止及四种physical原始reason用FORCED_TEST_STATE真实MuJoCo测试；unknown/错base/错stage/错goal/Val/Test拒绝和锁存；不把注入计作自然训练。
5. Checkpoint保存/重载零更新，sidecar同时保留旧L6base范围和新task白名单/active identity，RESTART_REQUIRED不变。
6. 新版本登记、完整105+新增OPS/21+新增Tensor/885历史无更新回归，原始R6/历史hash及预算字节核验，独立只读GO、中文报告/JSON/进度/Commit/OPS Push核验。

不修改旧schema6以重试R6；本阶段不开放新训练入口。R7发布后，另行冻结新ID/协议和审查才可做独立学习实验；算法变量仍只有距离课程，不能把实现修复说成性能提升。
