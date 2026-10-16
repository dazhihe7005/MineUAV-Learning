# AUTORL-R6：距离课程PPO对照设计与实施计划

> Native连续执行、TDD、独立运行前/结果只读审查。R5已推送并核验d4971529c63c9f2f305c78c851cb8d94b20d6487，12短Train物理可完成，不能把这个当作学习。原授权余额1,672,320步/68,400s、无stop；不重初始化账本。

## 假设、单变量与固定协议

假设：先训练0.20m任务能提高到点停稳的成功反馈密度，再迁移原目标。对照使用已完成R3 RANDOM/std.3的固定summary7651327c99f27940ce047d4b03b57b40f4ab59aff37632757ae70350445d36d1，绝不重跑旧对照。唯一变量为reset时的目标距离课程：前32768步均匀12短目标，下次自然reset恢复原12目标；不在episode中途改目标。所有几何、起点、Yaw、Reward、CNN、PPO超参数、动作/频率、24000射线不变。初始化仍seed0随机均值/std.3，std继续可学习；不使用旧checkpoint初始化。

新固定schema6/ID ARL_R6_DISTANCE_SEED0，65536步/3600s/CPU单worker。保留旧schema1/2/3的字节协议和schema4拒绝，不改变immutable grant。新入口必须用现有budget/supervisor/worker，不复制PPO算法。R5公开reportSHA f97b5266ecec75e9e8067ad9611026a3fd7d8e0db1bcb6743062f81919809ee7及proof395fe3891efc2b7917d087a939922d945bd9d3c1aace4e632efa95b366fd0a94固定；新任务manifest f93b6dce91abeadbde8ebaf619a3c5839cbb9312cca2402f2868472345d5faf3。R5实际验证过的short_goal/env源码需仍与已发布身份一致。

## Checkpoint与评价（训练前冻结）

保存0/32768/65536，完整rollout对应optimizer更新后保存。关闭Train后顺序评价，单环境常驻。每checkpoint独立运行6个派生短Validation，共18；原6Validation在32768/final再运行12episode。原step0评价复用R3 RANDOM相同权重的既有6条结果，不重跑、不计为新episode。实际新增Validation总30，Test始终0。新短Validation仅用于诊断，目标近且障碍可能不在路径上，不冒充避障能力。

三个标准分别判断，Final选择不择优：短任务final成功≥4/6、碰撞≤1/6、physical/sensor failure0；原目标final成功>历史0且碰撞≤历史6；短任务final成功数>自身step0。保留各类失败/return/最终及最小距离/路径/动作/停滞/KL/critic/资源，不因结果差修改协议或扩步。单seed/少量Validation不能证明泛化。

## Task1：固定新profile与预算准入

文件distance_profile.py、tests/research_ops_v2/test_distance_profile.py，修改budget.py/runtime.py/__main__.py只加显式schema6分支。

- [ ] RED：固定预算累计不刷新；任何seed/步数/std/初始化/manifest/评价篡改拒绝；缺失/改变R5 proof或历史控制拒绝；未完成新测试准入拒绝；CLI不接受任意checkpoint/脚本；旧五个attempt账本回放不变。GREEN后整套OPS测试。
- [ ] 实现distance_experiment/validate_distance_profile/verify_distance_proof/verify_distance_checks；prepare-distance/start-distance仅AUTORL-R6-DISTANCE注册可执行，prepare不训练；start仍由原supervise计费及资源保护。

## Task2：复用训练器与独立短Validation

文件distance_training.py、distance_training_tests.py；worker.py仅schema6选择原R5 scheduler、原std初始化helper及新短Validation，原schema执行不变。

- [ ] RED→GREEN真实无更新测试：新随机/std.3初态等于R3已验证policy SHA；唯一首次reset是新短目标、没有额外Train RNG，checkpoint0重载一致。
- [ ] RED→GREEN：新ShortValidationEvaluator只用派生Validation6，真实每个1step受限接口检查；模型/optimizer/训练步数/Train RNG不变；不访问Test；历史step0原评价只读复用验证权重身份，不作为新增episode。
- [ ] 实际课程训练复用原Measured/FailureEvidenceRecorder/rollout/optimizer包裹、记录双身份与阶段；summary追加实际新Validation计数和短目标清单hash。所有optimizer更新计预算，测试本身0更新。

## Task3：冻结、唯一训练与交付

- [ ] 新源码/协议/注册/当前全部无更新回归绑定；独立预审无C/I，OPS登记/recover/doctor、R5proof/历史控制/余额及stop核验。
- [ ] prepare后核对协议SHA，预留65536步/3600s后唯一start；全过程不修改运行代码、不自动重试，故障按RESTART_REQUIRED并保留attempt。
- [ ] 实际完成后核对65536、512rollout/2048Adam调用/finite weights+grad、checkpoint原子身份及optimizer/预测重载、12Train几何来源/短任务阶段/自然切换、全部30新Validation+0Test、历史hash不变。
- [ ] 原始数据/模型local-only，真实图/中文11节MD/JSON/六份长期文档，独立结果复审/Commit/Push远端核验。若改善FAIL，保存真实结论、分析下一可反驳假设，在剩余预算和安全条件下继续下一独立阶段，不当作终点。

## Review Focus

旧授权账本不能重置；旧profile/hash回放须一致；source与proof不匹配不可预留；reset而非中途改变目标；新/原task身份明确；step0旧评价不能误计新episode；模型选择Final；Validation不改变训练状态；无更新测试不能暗含optimizer。资源门槛MemAvailable1536MiB/磁盘2GiB，不放宽。
