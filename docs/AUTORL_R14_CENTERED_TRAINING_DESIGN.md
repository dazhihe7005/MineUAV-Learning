# AUTORL-R14：中心化Actor真实PPO单变量对照

## 背景和假设

R13接口已通过，成果经R13P正常OPS sync发布于76bd8065056fbb392e45d8f79b9e7c6aa5b394a5（远端SHA核验）。R13旧实际聚合SOURCE与后修管理SOURCE不同已明确披露；模型模块未改。本阶段不是重复R13/R11，不从其ZIP继续。假设：同背景zero-state均值锚定能减轻累计偏置、保留短目标能力，并有机会改善原目标；可能因为限制表达/探索/部分可观测性而失败。不认为R12已证明唯一根因。

## 唯一算法变量与冻结项

相对历史R11，唯一算法变量是Actor均值f(x)−f(anchor(x))。复用已提交state_centered_policy，原CNN/两层Actor及Critic参数值初态相同、170857参数，Gaussian初始std=.3且可训练。全局LR3e-5；n_steps128/batch64/n_epochs2/gamma.99/GAE.95/clip.2/ent0/vf.5/maxgrad.5，原NAV_V2_COLLISION_PREFLIGHT_V1 Reward、动作[1.5,1.5,1]m/s与1rad/s、500/100/25/10Hz、24k射线、7kg动力学和PI全部冻结。手写先验不是学习，不做免费optimizer测试。

## 身份、数据与预算

新schema14/ARL_R14_CENTERED_ACTOR_SEED0，65536实际训练step/3600s保守预留，CPU1/seed0。使用现有repaired distance scheduler：12冻结Train、先0.20m短目标，达到32768后下一自然reset切原目标。不改场景清单或几何，不读取Test。新完整初态policy20f5c226abc762e5aa4a9046ba400b2b0f7012b0da6715d14204b097cae06001，规范参数值SHA6312e99c…只比较值。旧L6/L7网络hash仅base provenance，新执行身份绑定实际policy类/源码/函数版本。保留旧协议和账本，不initialize/退款。

若实际预留，账本从589824step/32400s变655360step/36000s，余额1344640step/50400s；不是当前已消费声明。已用目录/attempt永久拒绝，不重试R6/L9或任何旧run；RESTART_REQUIRED。停止请求、MemAvailable<1536MiB、磁盘<2GiB、NaN/Inf/来源或身份异常立即停。

## Checkpoint和评价

步骤0/32768/65536各保存、重载并核实际新policy结构、LR、optimizer状态、确定性输出；真实训练checkpoint非空Adam重载是本轮要验证的证据。沿用原CheckpointManager，sidecar旧网络身份是base provenance，新execution hash及事件结构核验补足实际函数身份，不伪造旧hash。

训练实例关闭后，各checkpoint在6原Validation+6短Validation实际评价，共36新episode，历史原step0不复用。historical R11固定报告/summary仅作非并行对照：final原Val0成功6碰撞，short初态6成功0碰撞，最终short0成功6碰撞。所有评价确定性、不改权重/optimizer/训练RNG。最终checkpoint为选择规则，Test0。

预注册四个科学判据分别报告，不降低原任务标准：

1. 短任务候选：final short≥4成功、≤1碰撞、无physical/sensor_failure。
2. 先验保留：final short成功不少于own step0，碰撞不高于own step0。
3. 原任务学习：final original成功严格高于own step0，碰撞不高于own step0。
4. 相对R11函数改进：final original成功>历史R11 final0，碰撞≤其6，且无physical/sensor_failure；只是单seed有限历史对照，不泛化认证。

结构性zero-state均值0不作为学习PASS判据；首次128步前后的历史动作一致性只可作公平性诊断。报告五种互斥终止、minimum/final goal error、停稳联合样本/连续数、真实接触对、PPO KL/clip/entropy/EV/value loss、std/动作幅度与资源，不仅Reward。

## 实施计划（native，TDD）

- [ ] 先新增纯profile/CLI/累计账本/拒绝覆盖及缺证明准入RED，再固定centered_training_profile.py并仅扩展budget/runtime/__main__显式schema14入口。无任意参数或脚本。
- [ ] 新centered_training.py复用已验证Actor初始化/原scheduler/save/load，补真实1step无更新hook及36新Val计划/标签测试RED→GREEN。不改原导航环境/Reward/终止。
- [ ] 原worker只加schema14分支，仍原PPO算法与监督/失败记录；新类不走旧初态复用/旧fullpolicy注释。更新前后与checkpoint重载核实际policy结构/LR。
- [ ] 完整当前OPS/Tensor/885历史无更新准入，逐项15排除，独立只读预审，版本register/recover/doctor、全部历史/独立grant/预算stop/资源/实际源码、精确prepareSHA→唯一次start。运行期冻结所有生产和测试源码。
- [ ] 实际完成后封存raw，纯分析/真实图/中文11节报告/JSON/长期文档、独立结果复审、Commit/正常OPS sync及远端SHA；科学FAIL也保留并分析，不自动重试该attempt。

## 风险与自审

检查policy各SB3路径一致、梯度与非空Adam参数顺序、无旧initializationZIP输入、无初态评价身份冒用、36新Val准确、账本累计不刷新。中心化可限制避免障碍的常量偏移，零state缺角速度等信息仍不保证制动；原观测部分可观测性、std探索、Critic拟合仍可造成失败。当前单变量对照不是部署/安全控制器，不启动PX4/FASTLIO/硬件。普通阶段结束继续原授权内下一安全研究。
