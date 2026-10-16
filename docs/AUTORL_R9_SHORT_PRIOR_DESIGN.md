# AUTORL-R9：短目标解析Actor可完成性（设计与执行清单）

## 问题与范围

R8完整65536步但三科学标准FAIL，短Train21条episode进入目标球却没有联合停稳成功；不能据此唯一归因Reward/速度/保持时间。R9只回答：已有R2手写解析Actor均值，在R5冻结的12个0.20m短Train任务中能否按原标准成功？这不是重做R5 Scripted，也不加载任何旧模型。属于既有流程的有界扩展；用户长期授权要求自主推进，不另索普通阶段确认。

备选是直接再训练（缺非学习可完成性证据）或修改Reward（同时引入新变量）；本轮选择先验证同结构神经网络反馈，不变CNN/Reward/PPO/PI/物理/场景/24000射线。

## 冻结契约

实验ID AUTORL_R9_SHORT_PRIOR_FEASIBILITY_ONLY；唯一空目录`.research_local/autorl_r9_short_prior/run_01`；12个Train逐一覆盖，Validation/Test=0。新seed0网络，复用initialize_actor_prior，log_std设log(.3)以匹配既有R3先验身份6312e99c87fd5a4dca066bd1ef64acd17d5ee44b563c152c59c133ee659d7f74；deterministic predict，eval/inference_mode。CNN/Critic保持随机初始化，无optimizer、无learn/train，预算文件不变。手写state-only反馈不是学得的避障策略，不证明LiDAR依赖。

复用R8实际scheduler工厂（R7双身份adapter/R5短任务），每次显式base身份reset，没有初始化额外reset；完整TrainingMetrics/Health消费者仍经过真实step。375步原时间上限，每次原终止即记录；普通科学失败collision/time_limit可继续剩余冻结覆盖，physical_failure/sensor_failure以及任何未知终止、身份/非有限/安全/资源异常立即停止不重试。成功保持distance<.10m、speed<.15m/s连续5策略步，collision优先，不变原terminal/reward。整轮180s guard、170s主动停止、MemAvailable>=1536MiB/磁盘>=2GiB、单MuJoCo CPU。source及grant每步核验，旧R6已封存FAIL不构成未知活跃attempt，其他非COMPLETE一律拒绝。

记录真实task/base/位置/速度/动作/reward/ticks/LiDAR源时间与age/接触/规范及原始终止；每条episode的逐步trace在终止后一次性原子写入不可覆盖JSON，不宣称每step已落盘。普通异常保存已有内存trace；SIGKILL/断电可能丢失当前未结束episode，只保留attempt和已经落盘证据，必须按INCOMPLETE审计，不能补造缺失transition。唯一attempt/complete/result分别原子写入。解析先验只允许新随机且0更新网络，不接受checkpoint参数。评价all12成功为可完成性PASS；否则FAIL，不能改标准。0训练步/0优化器与实际物理测试步分别记录。

## 实施与验证

- [ ] 先增加纯协议/证据校验测试及真实模型初始化/一次step接口测试，RED后最小实现。新模块short_prior.py（固定spec、初始化、成功证据验证）、short_prior_runner.py（唯一安全入口）、short_prior_tests.py（真实无更新测试）和纯tests。
- [ ] 110项既有OPS、33项Tensor、885项历史选择回归及新增测试重新完整执行；历史15排除保持逐项依据，不运行裸pytest/额外optimizer。源码绑定，固定持久化上游，不读取封存Test结果。
- [ ] 新OPS登记/独立只读GO/recover/doctor、grant预算/stop/历史attempt与资源核验，再执行唯一非学习run。R8源码和旧证据保持历史身份。
- [ ] 原始trace数值检查、图表来自实际物理、独立结果复审、中文11节MD/JSON/六长期文档/README、显式Commit/OPS sync/远端SHA。随后依据结果决定新独立训练协议，不把R9当训练或先验成功当学习增益。

## 审查重点

真实task而非base冒充；合法失败可统计；不使用旧权重；不调用optimizer；伪成功或截短trace不得通过；全12覆盖不能因首个普通失败而省略；源/预算/停止检查；scope只Train。当前R7任意内部active_task结构损坏可能遮蔽context的Minor继续披露，不声称任意内存破坏可恢复。预审指出新增Minor：json_safe总512节点上限可能只保存异常trace前缀并截掉后置last_returned_transition，故不保证完整现场/最后transition必有；异常不能据此PASS，本阶段不扩展完整快照恢复能力。
