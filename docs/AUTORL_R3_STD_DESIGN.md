# AUTORL-R3：初始Gaussian探索尺度的受控研究与实施计划

> 执行方式：native，使用executing-plans／TDD；用户已授权预算内自主研究，不重复索取普通阶段确认。最终及运行前均独立只读审查。

## 研究意图与方案选择

R1随机Actor/std1与R2解析Actor/std1最终均0/6成功。R2未保住自身step0的1/6成功，全部512rollout平均raw动作元素越界57.37%，最终std1.0054；但解析网络参数尺度也改变了优化敏感性，不能认定噪声是唯一根因。选用只改变初始std的补充两臂，避免同时改Reward、学习率或网络：

- RANDOM：原随机seed0均值，初始std=.3；历史对照R1 UNIFORM/std1。
- PRIOR：原解析seed0均值，初始std=.3；历史对照R2/std1。

两组各65536实际步／3600秒cap、顺序CPU单worker，共131072步／7200秒保守预留。不重跑任何旧实验。初始std=.3后仍参与原PPO优化，不固定std、不更改ent_coef。对μ=0的一维Gaussian，越界概率由std1的0.31731变为std.3的约0.000858；这只是解析概率，不是导航表现保证，非零均值需实际记录。

候选“降低学习率”改变更新强度但不能分离初始探索；“BC训练”需要额外监督预算计量设计。因此本阶段不同时做它们。

## 冻结协议与结论边界

新schema3固定profile `INITIAL_STD_R3_V1`，ID `ARL_R3_RANDOM_STD03_SEED0`／`ARL_R3_PRIOR_STD03_SEED0`。原schema1/2、旧grant／账本事件／L9R denial字节不改。原12/6/6清单、Reward、CNN、PPO超参数、4D动作尺度、500/100/25/10Hz及24000射线不变；Test=0，RESTART_REQUIRED，禁止Warm Start。

两组都保存step0／32768／65536，Train关闭后对每份执行6Validation。Final选择，不挑最好checkpoint。第一层标准：每组最终成功数高于各自冻结std1历史对照，碰撞不增加；第二层：最终成功数高于自己的step0且碰撞不增加，才支持学习增益。候选基本导航门保持至少4/6成功、最多1/6碰撞、物理／传感器失败0。不同初始std两组之间不能混为单变量对照；历史对照有跨阶段管理代码差异，不做统计显著／多seed／LiDAR依赖／矿井泛化声明。

记录完整Train类别、raw reason、最小／末端误差、动作clip／std、全部rollout PPO、资源、checkpoint重载、初始化张量差分。保留step0先验与学习的区别；失败不延长、不重试同ID、不调当前协议。

## 模块边界

复用原supervisor／guard／worker／CheckpointManager／Train调度／Val。新`std_profile.py`负责固定协议、历史控制／物理来源和当前测试准入；`std_training.py`只在全新未reset模型上变更初始log_std，必要时先调用未改变的解析Actor，再正常一次reset及step0快照。`budget.py`／CLI／runtime／worker仅增加schema3显式分支，旧分支语义不变；不开放任意脚本／checkpoint参数。

## 全局约束

同一不可重置200万步／86400s授权；当前余额1803392步／75600s，预留全额不退款。单CPU worker，MemAvailable>=1536MiB、磁盘>=2GiB；无PX4／FAST-LIO2／硬件／Test／历史修改。先完成源码身份、无更新测试及独立GO才允许预留和启动。

## Review Focus

1. 新schema不能重新解释原R1/R2预算事件，也不能刷新grant。
2. 初始化只改变log_std及声明的先验Actor；CNN、Critic、随机数／Train序列不得隐式改变。
3. step0只做评估，不能变成Warm Start；重载必须保留optimizer及预测身份。
4. 缺失／过期准入、源码变化、旧目录／消费ID必须在训练前拒绝。
5. 两个历史对照与自身step0的科学判断分开，不能选择中间最优或混用动作clip统计单位。

## Task 1：固定profile及预算兼容

Files：新增`std_profile.py`、`tests/research_ops_v2/test_std_profile.py`；修改`budget.py`。

- [x] 先写测试：两固定臂能够通过同一账本，未知arm／改步数／std／seed／初始化文件均拒绝；旧R1/R2事件字节不变，重复ID拒绝；缺当前准入拒绝。
- [x] 运行定向标准库测试，观察缺实现RED；实现固定schema3分支，再GREEN。

## Task 2：真实初始化与显式入口

Files：新增`std_training.py`／`std_training_tests.py`；修改CLI／runtime／worker。

- [x] 无optimizer真实SB3测试先RED：全新随机模型std.3只改log_std；先验模型先与R2精确相同再只改log_std；deterministic动作、CNN／Critic、RNG不变。
- [x] 实现`initialize_std_training(model,scheduler,manager,*,identity,seed,variant)`；step0重载与_setup_learn无额外reset，错误初态拒绝且不赋值。
- [x] CLI显式prepare-std/start-std --variant RANDOM/PRIOR；prepare/start核验profile／当前准入／冻结历史control和未改物理来源，不受理checkpoint。
- [x] 只运行无更新Tensor／OPS／历史选择回归，所有身份绑定同一候选；独立预审及OPS新版本登记。

## Task 3：唯一两臂执行与结果发布

- [x] 独立预审无Critical/Important、准入真实完整后，在原账本顺序预留／执行两个独立实验，各65536；不重复旧运行或临时修改源码。
- [x] 原始结果local-only；验证每臂实际步数／2048 Adam／512rollout、3checkpoint、18Validation、Train身份／历史hash及预算。
- [ ] 真实图／中文11节报告／JSON／长期文档、独立结果复审、显式Commit/Push及远端SHA核验；阴性结果同样保存。

接口预检：Task1固定protocol供Task2显式入口及Task3；Task2初始化只接收固定variant，不新增任意执行能力。当前旧schema3拒绝测试会继续拒绝“把R2协议schema改3”的非法对象，不应删除旧测试。测试禁止PPO.learn/train及Adam.step，实际训练更新只能由计费运行产生。
