# AUTORL-R22：固定折扣目标下的 GAE 时间信用对照

## 研究问题与证据

完整R17增加empty任务reset概率后，实际原域policy步83.47%来自empty，最终原距离Validation仍0成功/3碰撞/3超时。R18观测到一个empty Validation的两个final策略从未在采样时刻进入0.1m目标区，不能归因为“已抵达后刹不住”。R19三条轨迹的成功折扣回报高于所观察失败，反驳该局部Reward排序错误解释，但真实逐transition advantages未保存，因果仍未知。

R20把gamma改为.995的独立实验被系统关机中断，永久FAIL/INTERRUPTED、科学NOT TESTED；不重跑、不加载其中间ZIP。本阶段不把R20部分Train记录当作成功对照。相对完整R17，仅把GAE lambda从.95改为.99，gamma保持.99；研究较慢的优势迹衰减是否减少原域课程后的停点偏差。这与R20改变折扣目标的问题不同，不是换ID重试其协议。

gamma*lambda从.9405改为.9801，几何半衰期约.452s改为1.379s（25Hz）。这些只是核衰减尺度，不是硬截断、真实梯度归因或已证明的导航提升。较长迹可能增大方差、使Critic或Actor恶化；失败也完整保留。备选为原域empty-only隔离，但已有83%实际曝光，因此本次优先只检验lambda，不同时改Reward或采样概率。

## 固定实验与算法

新ARL_R22_GAE_CREDIT_SEED0，唯一新随机seed0手写先验初始化；禁止任何旧ZIP初始化。65536 actual steps，3600秒全额永久预留，CPU1worker；用既有独立grant，绝不重置/退款/扩权。旧余额1148032/39600，若执行预留后应为1082496/36000。运行前再次核验stop、锁、所有attempt、当前SOURCE/REG/依赖/历史/资源、剩余额度和独立GO。

仍采用R13原中心化Actor及同CNN/参数对象/共享特征、初始std.1且可训练、LR3e-5、gamma.99、n_steps128/batch64/epochs2/clip.2、其余PPO参数不变。Reward NAV_V2_COLLISION_PREFLIGHT_V1、7kg动力学、PI、500/100/25/10Hz、24000rays、原12Train及short清单不变。初态完整policy SHA必须650a5841c96a054bf43db9a54a6c5c6da6af8b4b362e4c934a2121d0a16849a4。短距到32768后下个自然reset切原域，原域empty reset概率.5；不会要求与R17的32k权重相同。

## 执行路径与失败封口准入

独立gae_credit_cli只提供固定prepare/start与显式协议SHA，经原supervise/worker执行，不能接受shell、任意种子、路径或checkpoint。configure在第一次采样前同时改model及rollout_buffer的gae_lambda，verify在每次更新前后、加载checkpoint及summary检查两者与gamma/其余配置，算法仍由SB3实现。

新closed_failure_proof不改旧eligible_budget或BudgetStore事务；只验证已发布R20 FAIL报告、两个实际独立审查、全部16原raw及posthoc、严格第26事件SHA/理由、真实累计账本匹配，以及R21已发布修复/收据/guard和9测试源码。未知FAIL、pending、stop、假预算、缺证据仍拒绝；R6仍要求旧冻结修复链。旧R20身份永久消耗、旧gate仍拒绝；仅允许新、另经协议和审查的研究。R21_SIGTERM为best-effort检查点语义；SIGKILL/断电/预留后I/O窗口仍不可保证收据，不实现resume。所有新版本仍遵守RESTART_REQUIRED。

## 冻结评价

三个checkpoint在step0/32768/65536保存并验证optimizer及预测重载；全部Train环境关闭后，每个checkpoint独立跑6short+6原距离Validation，共36个新Val；Test=0，Final Checkpoint选择，不选最佳中间模型。

五项预注册判据：32k short≥4成功/≤1碰撞/无物理或传感器失败；final short成功不少于自身step0且碰撞不增；final原域成功高于自身step0且碰撞不增；final原域成功高于完整R17的0、碰撞≤3；原域empty final成功不少于自身32k且32k至少1成功。全部满足才科学PASS，缺证据NOT TESTED，完整观察不满足FAIL，不改阈值。单seed/小Validation不能证明真实矿井泛化或LiDAR因果使用。

## 测试、审查与取证

9项新标准库契约及4项Tensor测试：profile单变量/授权不变/未知失败拒绝/旧gate仍拒绝；真实SB3 GAE与TD递推；参数预测不变；零更新checkpoint/受控step；真实train_worker分派直到learn方法边界立刻拦截，不执行任何采样或优化器。9+3先missing-module RED；另真实worker .95分派RED，修复后GREEN；中间开发验证对GO_DISPOSITION_ONLY历史收据类型的精确适配不会修改原收据。正式准入必须绑定最终源码的全套无更新测试及独立GO，不用开发结果替代。

冻结运行中不改SOURCE、Reward、网络或场景；资源MemAvailable<1536MiB/磁盘<2GiB、NaN/Inf、身份/来源/记账异常立即停止、保留唯一attempt和完整预留，不自动重试。真实完整结果、PPO健康/动作/目标误差/Train访问/Val和资源图、MD/JSON、独立复审、正常OPS同步与远端核验后，再决定下一项研究。当前设计不是已执行结果，也不是训练完成证明。
