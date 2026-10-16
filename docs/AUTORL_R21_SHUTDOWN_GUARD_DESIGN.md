# AUTORL-R21：主管 SIGTERM 事务边界设计

本阶段为用户长期自主研发授权内的 Level0/1 离线修复，不执行任何学习或训练后模型评价。上阶段 R20 已以 FAIL/INTERRUPTED 发布（5e95e3f9373753f408b6e2b92beb2670193e00cd），不重跑；剩余独立授权 1148032 步/39600 秒预留墙钟，不退款，不修改历史 grant/预算/attempt。

## 已确认问题与方案选择

R20 原 boot 的 kernel 已记录系统关机和向 remaining processes 发 SIGTERM，不能确认该 worker 最终信号或精确步数。当前主管默认 SIGTERM 处理直接退出，不能进入 Python except 清理。相比异步 raise（可能撕裂 budget event/head 两步写入）、关机抑制或后台服务（不在授权范围），选择最小的延迟处理：handler 只锁存第一条信号及 boot/两种时钟，主管在预留前、启动前、监控 probe、worker 退出后、完成事务前检查。

启动检查点已经观测到信号时，不启动新 worker；检查点之后或 Popen 创建期间才到达的信号仍可能短暂创建 child，随后清理自身独立进程组，不能保证信号到达后零额外执行。按既有路径写 supervisor_failure 并追加 FAIL，保留全额预留。事务内到达的信号不会立即抛异常。如果完成证据已经核验并越过最后完成检查点，则完成 COMPLETE event/head 事务，不追加矛盾 FAIL；即使信号恰好在 event 发表前到达也遵守这个边界，而非声称信号和提交存在绝对线性化。这个边界是“已经完整完成”的记账语义，不是忽略仍在运行的训练。

## 数据与边界

新增 shutdown_guard.py，不导入 Torch/SB3/MuJoCo；runtime.py 只接入上下文和检查点，并在 failure 中记录 termination_request、exact_final_timesteps=NOT_VERIFIED。原 handler 在退出时恢复；重复 SIGTERM 不覆盖第一条观测。不改 budget.py、worker.py、lifetime.py、PPO、Reward、物理、扫描器、旧运行数据或旧协议。

不保证 SIGKILL、掉电、不可中断 I/O 或系统关机宽限时间不足时完成清理；没有实现 EXACT_RESUME/BOUNDARY_RESUME。预留后、try 前目录/permit 写入 I/O 异常的旧窗口以及 guard 的强制 SIGKILL 缺少独立原因收据，仍是已知限制，后续只能独立审计/设计，不能宣称本次解决全生命周期。

## 验证

9 项标准库测试：第一信号锁存/重复/无信号/异常恢复 handler；隔离主管真实 SIGTERM、预算 reserve event 已写但 head 未写时的 SIGTERM、核验后但最后完成检查点前的 SIGTERM、COMPLETE event/head 中的 SIGTERM，以及启动检查点之后/Popen之前信号的真实短暂 child 清理。集成测试只替换准入和实际训练 worker（为短命标准库进程），使用临时真实 BudgetStore、原子 JSON、真实信号和 stop_group；不会启动 PPO 或写真实授权账本。前 8 项 RED 已复现 4 个隔离主管 -15 和 4 个缺失处理断言；第9项为审查新增窗口行为刻画，不伪称额外 RED。最终证据必须绑定登记源码和完整 GREEN 回归。

独立只读审查后才登记/发布。测试 PASS 仅表示这些离线信号场景，不是修复后真实训练或实际关机验证。新算法训练仍需新协议、新准入及审查，R20 永久不重试，Test=0。
