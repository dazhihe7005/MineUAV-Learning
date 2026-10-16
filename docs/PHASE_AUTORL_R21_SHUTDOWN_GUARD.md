# AUTORL-R21：SIGTERM 事务边界离线修复

## 【1】阶段目标

修复主管收到普通 SIGTERM 就直接退出、无法进入清理路径的可复现问题。不执行学习、不重跑 R20、不改变预算或算法。实现 PASS；导航科学效果 NOT TESTED。

## 【2】技术背景

R20 已发布为 FAIL/INTERRUPTED，提交 5e95e3f9373753f408b6e2b92beb2670193e00cd。原 boot 的 kernel 有关机和群发 SIGTERM 证据，但 worker 最终信号及精确步数未知；40448 只是最后已验证完整更新。不能把该次中断当作 gamma 科学失败，也不能读取中间 ZIP 续训。

## 【3】系统架构

固定 supervise 入口 → DeferredTermination 锁存 → 既有校验/预算事务 → 检查点 → 子进程监控 → 完整完成或既有失败清理。handler 不做文件 I/O、预算操作或异步 raise，不新增后台服务，不抑制关机。

## 【4】代码实现

新增 `research_regression/authorized_rl/shutdown_guard.py`；修改 `runtime.py` 接入上下文，在 reserve 前、启动前、monitor probe、退出后和最后完成前检查。失败 JSON 新增原信号观测及 exact_final_timesteps=NOT_VERIFIED。新增 `tests/research_ops_v2/test_shutdown_guard.py`。未改 budget.py、worker.py、lifetime.py、Reward、CNN、PI、物理或扫描器。

## 【5】原理与边界

异步抛异常可能卡在 event 已写而 head 未写之间，破坏哈希链，所以先锁存、再在事务边界抛既有 OpsError。完整完成已经越过最后检查点后，COMPLETE 事务优先，不再追加矛盾 FAIL。启动检查点之后才到达的信号可能短暂创建 child，再清理；并非信号与 Popen/提交绝对线性化，也不保证零额外执行。

## 【6】实验协议

离线协议 SHA256 `e1d678d6dc1cd6e82ed0c63691812d841b9ce4f76940d4dd1a8a1a5332e8e4fa`。REG `22006cf50fa7a917045e78beda98f5b3fcb4da31469b1d13088a26e184260ce3`；SOURCE `42c73b28580f5eaf722009eaa42226e0f197bcc705918263d1a7f8d055916b24`（131文件）。新增9项信号测试只使用临时真实 BudgetStore/原子 I/O/真实 Linux 信号和短命标准库进程；准入和训练 worker 由测试替换，模拟 worker 的 summary/optimizer 数是受控夹具，不是真实训练结果。其余66项Tensor和885项历史回归可以构建fresh模型、临时checkpoint及受控MuJoCo reset/step，并非全体测试都是标准库或零物理；不评估R20训练模型。PPO optimizer=0、训练步数=0、Test=0、PX4/FAST-LIO2/硬件=0。

## 【7】实验结果

最终 1153 项选择范围 PASS：202 OPS（含9项新增）、66 无更新 Tensor、885 历史；15项排除，非全仓。前8项 RED：4缺失handler断言、4隔离主管直接SIGTERM退出(-15)；修复后 GREEN。第9项是审查新增启动窗口刻画，不伪称额外 RED。reserve和complete测试在真实 event 发表后/head 更新前注入 SIGTERM，链仍可读取，分别仅 FAIL 或 COMPLETE；live child 的真实进程组清理通过。

第一轮 _01 的个别套件结束，但途中新增第9测试改变 SOURCE，最终绑定断言失败，整体 INCOMPLETE；全部原证据保留。_02 已重新跑完整套件并绑定当前源码，不偷用旧PASS。资源详见 JSON：OPS墙钟 18.007s/主管峰值RSS 28.195MiB，Tensor墙钟 58.415s；不能外推真实训练/系统关机的清理性能。339份 grant/head/events/run/旧raw全部不变；剩余1148032步/39600秒预留墙钟，stop=false。

既有 SB3 可选 Atari 导入中的 cv2/NumPy ABI 警告仍被捕获并保留在 stderr；本次所选 CPU 路径不调用 OpenCV，完整退出码为0，不把警告隐瞒成无警告环境，也不升级系统依赖。

## 【8】问题分析

CONFIRMED：默认SIGTERM会绕过清理，离线真实信号已复现并修复；独立预审发现绝对“零启动”声明过强，现已收紧并补测。SUSPECTED：R20 worker 中止与已证实关机时间强相关。UNVERIFIED：该worker确切最终信号、停止步数、旧boot是否OOM、关机发起者/原因，本阶段没有补造这些信息。

## 【9】局限性

这是 best-effort SIGTERM 处理，不是 SIGKILL/掉电/不可中断I/O保证；未实际关机，也没有运行修复后真实 PPO。原预留后 try 前目录/permit I/O 异常窗口、guard 强杀独立原因收据缺失仍存在。真实训练新运行时 NOT TESTED。RESTART_REQUIRED不变，无 EXACT_RESUME/BOUNDARY_RESUME。旧 eligible_budget 对R20 FAIL仍拒绝，不能直接绕过来训练。继承R7/R18/R20审查历史不抹除。

## 【10】下一阶段

正常 Commit/Push/远端核验后，独立离线研究已关闭失败的版本化证据准入，并设计与完整R17对照的 GAE 单变量假设；不重放已中断R20。任何新训练必须重新冻结协议、检查累计预算及独立GO，不能把本修复当作算法效果或新执行批准。

## 【11】给项目负责人的总结

这次解决的是“收到正常终止信号时尽可能先保存失败记录和清理自己启动的进程”，没有声称电脑断电后还能工作。没有消耗训练预算，也没有提升导航成功率。R20失败历史和中间模型全部原样保留。接下来先处理失败证据如何被新协议正确识别，再选择独立算法实验；已有长期授权仍有效，不需要为普通离线研发重复确认。
