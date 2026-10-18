# AUTORL-R34：历史预算测试夹具修复

## 【1】阶段目标

修复五项旧回归因实时账本增长产生的失败；保留真实运行门禁的拒绝能力。工程PASS，导航性能NOT TESTED，训练与optimizer更新0。

## 【2】技术背景

R33重启中断已永久FAIL并公开于34d46f722cad0ed61c765ed98a5083fcb8f17911。闭环处置后原280项OPS实际为276PASS/1FAIL/3ERROR：三个历史正例遇到新增R33FAIL，另一伪造余额负例被未知失败分支先行遮蔽。原生产门禁拒绝R33是正确行为，不能修改它来让旧正例通过。

## 【3】系统架构

历史单测：固定R28报告SHA→完整budget_after→显式单home reader替身→真实proof文件及原gate→历史断言。集成负例：退出context→真实head34→原三个入口拒绝R33。两条路径分开，不把单测快照当当前授权。

## 【4】代码实现

只改六个测试文件：tests/research_ops_v2下新增_historical_budget_fixture.py、test_historical_budget_fixture.py，调整test_closed_yaw_failure.py、test_goal_heading_diagnostic.py、test_interrupted_yaw_diagnostic.py；research_regression/authorized_rl/interrupted_yaw_diagnostic_tests.py的旧只读loader测试也显式复用该reader。固定完整R28快照而不是删掉当前账本的失败记录。每次读取深拷贝、只匹配实际home、正常及异常退出自动恢复BudgetStore.status；其他home仍调用原reader。真实checkpoint/proof检查与optimizer禁用保留，loader测试退出再核实际账本。生产源码、原预算、模型、Reward、PI和安全门禁均未修改。

## 【5】验证原理

历史正例必须有固定依赖，真实当前准入必须读实时证据。两者混用会把正常的fail-closed当回归故障。负例分别断言unknown/pending、stop latched、actual immutable三个准确分支，防止“任意抛异常”掩盖未覆盖的安全逻辑。context在单测试进程作用，不支持并发生产使用，也不提供任何训练授权。

## 【6】协议及身份

独立AUTORL_R34_HISTORICAL_FIXTURE_ONLY，Level1；REG `42b046d6513e9b0fe3fd498fa8905233302f8584c5ae84c65ac95388935a711d`，189源码身份 `1e12eef096b1478797604f95955997e5e1e82ee8abbff2830cafe5a9db515c45`，执行协议 `cfa73522c28a00654dc168bb4bab81bd8dad086794ddee089c940f2d3279b0e2`。固定R28报告 `54af04e2001f6612fea6a20d6af1837243a3a7b23c728ddf3d61d3b0921b35fc`；该历史快照16attempt/951424步/28800秒，**不是当前余额**。真实账本head34仍885888步/25200预留秒、stopfalse，原grant不变。单CPU测试进程，MemAvailable底线1.5GiB，磁盘底线2GiB。

## 【7】实际结果及资源

| 证据 | 实际结果 |
|---|---|
| 旧280项闭环回归 | 276PASS、1FAIL、3ERROR，保留原收据 |
| 新增7项缺接口RED | 7FAIL |
| 首次修复后中间收据 | 6PASS、1ERROR：其他home的实际错误是FileNotFoundError，测试原错误预期OpsError；文件名含GREEN但不是GREEN |
| 修正实际异常契约 | 7PASS |
| 候选01完整回归 | OPS287/历史885通过，但Tensor92PASS/1ERROR；独立NO_GO，未激活 |
| 候选02修复 | 旧R26 loader也使用同一显式历史夹具，旧候选及失败收据完整保留 |
| 候选03格式修复 | 02全部测试通过后，cached diff发现新测试末尾空行；只移除一字节LF，AST完全一致；02所有文件归档，重冻03并重跑全部回归 |
| 最终OPS | 287PASS |
| 继承无更新Tensor | 93PASS |
| 继承历史选择回归 | 885PASS、15DESELECTED |

合计1265选择PASS、0FAIL/ERROR/SKIP，非全仓。7新增测试包含快照完整性、深拷贝、异常还原、单home限定、缺失/篡改拒绝、真实当前三个入口拒绝、预算分支与文件不变。受限历史物理/模型接口测试不等于新导航实验；optimizer与科学Validation/Test均0。

572保护文件、183项非改动旧源码、全部授权目录文件含head字节一致。没有新增训练reserve/protocol/attempt/checkpoint；R33的16raw和FAIL不变。

OPS wall 19.992s，主进程峰值RSS 39.320MiB，子进程HWM 39.320MiB，CPU 7.299s，端点最小MemAvailable 9559.930MiB；Tensor wall 139.145s。不是整个阶段瞬时峰值或训练吞吐。已有可选cv2/NumPy ABI警告继续保留，不升级环境。

## 【8】问题分析

CONFIRMED：旧测试实时依赖与错误分支遮蔽；实际RED→GREEN及外部真实门禁负例支持修复。CONFIRMED：R33原失败证据和剩余预算未改。UNVERIFIED：新的导航能力、百万步稳定性、下一实验效果。没有新科学图，不复用旧扫描伪装本阶段数据。

## 【9】局限性

历史文件仍是proof测试的必要本地依赖；本次不是完全脱离历史数据的便携套件。没有EXACT_RESUME/BOUNDARY_RESUME。测试通过不能释放R33重试权限；旧gate仍关闭，需要新独立关闭证明和新的研究准入。

## 【10】下一阶段

正常发布、核验远端及预算/停止/资源后，先独立只读审计R33关闭证明与现存中间证据。判断是否值得对保存的32768步checkpoint做新、有限且明确标记的Validation诊断；不得补作R33最终评价或训练续跑。新训练须独立预注册、审查和实际计入剩余原授权。

## 【11】给项目负责人的总结

这次修好的是历史测试，不是让失败实验“变成功”。原训练因为电脑重启中断，不能接着旧ZIP无损续训；预算仍885888步/25200秒。本轮零训练，相关测试恢复全绿，同时确认真实安全门仍然拒绝旧入口。接下来依据已有中间证据选择有价值的新研究，不重跑失败attempt。您看到的两段动图是另外保存的真实历史状态回放，不是本轮新训练或PPO已经学会导航的证明。

最终独立只读复审GO，C0/I0；仅准许发布测试夹具修复，不构成新训练准入。科学数据/预算不变；最终Commit及远端以真实OPS同步收据为准。
