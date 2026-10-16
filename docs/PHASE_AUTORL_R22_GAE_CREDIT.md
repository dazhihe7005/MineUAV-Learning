# AUTORL-R22：GAE时间信用单变量对照

## 【1】阶段目标

相对完整R17，仅改变GAE λ .95→.99；γ保持.99。实现PASS、科学FAIL，不是R20重试。

## 【2】技术背景

R19显示原域empty实际步数已83.47%，观察成功的折扣Reward高于失败，不能仅归咎曝光不足或Reward排序；R18 final从未进入目标区。R20γ.995被关机中断，永久FAIL/科学NOT TESTED，不作成功对照、不继续旧ZIP。

## 【3】系统架构

固定CLI→已发布失败封口证明→原预算/主管/worker→中心化Actor与SB3 PPO→关闭Train后独立Validation→计数及图表。新gate不改旧eligible_budget，也不退款或修改历史拒绝字段。

## 【4】代码实现

新增closed_failure_proof、gae_credit_profile/training/cli/tests及标准库测试；budget仅增schema22校验，runtime/worker增固定22分支。R21锁存SIGTERM不变；原物理、PI、Reward、CNN、旧模型均不变。实际worker的无更新测试在learn方法边界停止，防止适配函数正确但入口漏分派。

## 【5】算法原理

GAE A_t=δ_t+γλA_(t+1)，δ=r+γV_next−V。λ增大改变优势及Critic λ-return，不改变γ的折扣目标。核γλ从.9405变为.9801，半衰期约.452s→1.379s，不是硬窗口或因果证明。保持原Reward `10Δd−.005||a||²−.5w(d)||v||²+10success−30collision/physical−10sensor−.01step−2timeout`。

## 【6】实验协议

ARL_R22_GAE_CREDIT_SEED0，fresh seed0手写先验/std.1可训练，非旧ZIP；65536实际步/3600秒全额预留/CPU1。137文件SOURCE `5b4013569e334db25b0154ad5c88a3d6d6ba781bb75e988ca1e19606234887e8`、REG `4564cd3419bbb607fe3ee29355aa7def17e600048cf5b89e327a36e05e2ceba2`、EXEC `7d24fa9a6ff3dc45497ff6715af3d65e3e930942d96e9986e2bccad51f68bd6c`。中心化Actor/同CNN/170857参数/LR3e-5；n_steps128,batch64,epochs2,γ=.99,λ=.99,clip.2,其余不变。原12Train+short .20m课程，32768后下个自然reset才切原域，empty reset概率.5。500/100/25/10Hz/24000rays不变。Final Checkpoint，36新Val、Test0；五项原判据在设计中预先冻结，没有改阈值。

## 【7】实际结果

65536真实environment steps/1309969实际physics ticks、512rollout/2048Adam/1024epoch，Actor/Critic均更新且模型/梯度有限，三个checkpoint实际重载optimizer/预测一致。terminal可能在一个策略步内部提前结束物理推进，因此不把65536×20当作实际值；计数已用各结束episode的真实仿真时间/.002和未结束尾步核对。短域结束165个：{'collision': 17, 'success': 135, 'time_limit': 13}；原域结束149个：{'collision': 57, 'success': 60, 'time_limit': 32}。尾episode 85步只计真实预算，不补造Reward/终止。真实切换步32882。

| checkpoint实际步 | short成功/碰撞/超时 | 原域成功/碰撞/超时 |
| --- | --- | --- |
| 0 | 6/0/0 | 1/3/2 |
| 32768 | 3/1/2 | 0/4/2 |
| 65536 | 2/2/2 | 2/4/0 |

五判据：{'empty_retention': 'FAIL', 'gae_credit_benefit_over_R17': 'FAIL', 'original_learning_over_prior': 'FAIL', 'short_candidate': 'FAIL', 'short_prior_retention': 'FAIL'}；科学FAIL。32k不是模型选择结果，报告final。余额1082496步/36000秒预留wall，旧349不可变文件不变；仅新增RESERVE27和COMPLETE28。训练前1166选择回归PASS=211OPS+70无更新Tensor+885历史/15排除，新增9+4真实RED→GREEN；5项分析契约及2项finalizer来源/实际物理计数契约另列，非全仓。旧可选cv2/NumPyABI与Matplotlib3D警告保留，不升级依赖。完整CLI wrapper墙钟911.098s、主管child监控段892.686s（不含之后核验）、worker进程至summary含延后Val的HWM1764.305MiB、训练累计采样最低MemAvailable7993.465MiB。末个训练resource前缀HWM另为1652.055MiB（step65536），不冒充全体进程合计瞬时内存峰值，后者NOT AVAILABLE。

开发阶段finalizer契约_01因测试自己的AST `.values()` 拼写错误而ERROR，记录保留；修正测试后_02/_03来源检查PASS。独立审查又指出硬编码physics ticks错误，真实R17计数触发生产断言RED；修复后_05与最终_06对真实R16/R17及非法计数GREEN，_06同时准确区分worker含Val的动态HWM。没有修改算法或执行额外训练，不把失败/旧版证据当最终准入。

## 【8】问题分析

CONFIRMED：完成固定实验及实际模型更新、Train/Val隔离和记账；结果仅来自真实raw。五判据如上逐项如实报告。SUSPECTED：lambda较长迹可能改变停点与课程后保持，或增加方差；仅凭一次对照不能证明梯度因果。UNVERIFIED：逐transition advantage、未见矿井泛化、真实传感器和PX4迁移。不得把手写初态能力当作学习，不把Reward高当成功率高。

## 【9】局限性

单seed/小规模开发Validation，Test仍封存。没有实机、PX4、FAST-LIO2或新传感器噪声验证。R21只best-effort SIGTERM，不保证SIGKILL/掉电或精确续训；RESTART_REQUIRED不变。继承R7/R18/R20历史审查问题保留。

## 【10】下一阶段

先独立结果复审、正常OPS Commit/Push并核验远端，再核预算/stop/资源，根据实际证据选择新独立研究。不能重试R20/R22、不能追挑32k模型或改本次协议。算法不改善则记录FAIL后提出单变量新假设，不盲目百万步。

## 【11】给项目负责人的总结

这轮只检验“优势估计看得稍远一些”是否有助于到达停稳，没有改无人机或奖励。真实Validation表与五判据给出答案：FAIL；不能因此宣称可靠避障或真实矿井可部署。预算仍按完整预留扣除，历史失败没有被覆盖；继续研发须先完成本轮发布，再使用已授权余额，不需要重复批准普通研究步骤。
