# AUTORL-R2：Actor初始化单变量实验设计与实施计划

## 研究问题

R1均匀和课程最终均0/6成功；R2前置证明相同原网络经解析均值初始化可在3条empty Train按原标准停稳。只问：这种基本引导能否改善PPO训练，而不是继续重复课程？只改变Actor初始均值权重，保留CNN、Critic、log_std=0（std=1）、Reward、PPO、500/100/25/10Hz、24000射线及原12/6/6清单。

## 独立冻结协议

执行profile `ANALYTIC_PRIOR_R2_V1`，实验`ARL_R2_PRIOR_SEED0`，seed0、CPU单worker、65536实际训练步、wall上限3600s，32,768／65,536完整optimizer后保存。全新随机seed0权重须与R1精确一致，再调用未改变SHA的解析初始化，不读旧checkpoint。沿用原累计授权账本，追加预留和完成事件；不修改旧grant／R1事件／L9R denial。

对照为**已经冻结的R1 UNIFORM历史控制**，不用已完成ID再次训练，也不声称是新同时执行的对照；同seed／预算／底层算法但存在跨阶段实现版本差异。新增管理代码限于初始化与step0快照／证据门控，需独立审查。保留此局限，不以单seed声称统计显著。

增加step0初始化checkpoint：一次真实训练env reset，保留对应_last_obs及episode-start标志；不额外抽场景、不采样动作、不调用optimizer。三份冻结checkpoint（0／32768／65536）均在关闭Train后分别执行6Validation，Test=0；读取step0仅做评价，永不用于恢复／Warm Start。Validation不影响本次训练随机序列。选择固定Final，不择优。

## 事前评价口径

1. 初始化对照效果：R2最终Validation成功数>R1最终0，且碰撞数≤R1最终6。只能称此单seed控制的初始化效果，不能据此认定PPO学习。
2. 学习超越手写先验：R2最终Validation成功数>自身step0，且碰撞不增加；若相同或更差，不能说PPO改善先验。
3. 基础导航候选准入：最终至少4/6成功、最多1/6碰撞、物理／传感器失败0。即使达到，也不是LiDAR依赖、多seed或泛化证明；还需要独立感知消融与复现。

同时报告完整Train成功／碰撞／超时／物理／传感器失败，目标最小／末端距离、路径、动作范数／裁剪、PPO全rollout指标、资源。失败不改Reward，不延长或自动重试本attempt。

## 实施计划（native，按已批准自主研究范围）

- [x] TDD：新增固定schema2 profile及旧账本兼容、禁止改预算／超预算／旧ID重试／未知初始化；先RED后GREEN。
- [x] `prior_profile.py`限定新执行参数与已发布预检报告／原始证据SHA；核验初始化函数和原物理源码未变、原R1对照摘要身份正确。
- [x] 复用原`runtime.supervise`，仅增加schema2有界证明分支；旧schema1语义保持。CLI显式prepare-prior/start-prior，不接受任意脚本／路径／checkpoint。
- [x] 复用原worker与CheckpointManager；解析赋值前后精确SHA，step0快照保存／重载及SB3_setup_learn因果关系无optimizer测试。拒绝重复reset或状态串漏。
- [x] 新OPS登记／源码冻结／独立预审、标准库与真实SB3无更新检查、相关历史回归；无Critical/Important后预留65536／3600并完整执行唯一一次。
- [ ] 三份checkpoint独立Validation、真实结果审计／图／MD／JSON／长期交接、独立结果复审、显式Commit/Push及远端核验。

## 审查重点

step0不是optimizer里程碑，不能扰动训练初始观察或多抽Train RNG；旧历史不能被schema2重释；跨阶段对照局限和初始能力必须披露；任何额外PPO更新都进入同一不可重置累计账本；若实际源码改变，原预检不能冒充修改后的验证。

最终执行前另有固定`autorl_r2_checks.json`准入收据：绑定当前登记／源码及完整OPS、9项Tensor、885项历史回归的真实文件SHA。prepare和start在创建／预留训练前核验，缺失或不完整拒绝。该收据不是新训练授权，预算仍只能来自原不可重置用户账本。
