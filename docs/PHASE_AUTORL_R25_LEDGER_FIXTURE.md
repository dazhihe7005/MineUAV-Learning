# AUTORL-R25：历史预算fixture隔离修复

## 【1】阶段目标

修复R24重启后两项旧历史正例被当前新增FAIL状态误伤的问题，同时保证真实当前训练门禁继续fail-closed。工程PASS；导航性能NOT TESTED，optimizer0。

## 【2】技术背景

R24中断已公开996d2b10ab39e4303476450f6b510142a35d31a1，科学NOT TESTED、phase_complete=false。重启后222OPS有220PASS/2ERROR，旧正例把会增长的真实账本当历史输入；原gate拒绝新增R24FAIL是正确行为，不能删门或把R24改COMPLETE。

## 【3】系统架构

历史unit路径：固定公开R22报告→完整budget_after→显式受限reader替身→原真实proof/原gate→精确断言。真实集成路径：退出context→原BudgetStore读真实head30→原gate拒绝R24FAIL。二者分开，不将历史通过当当前准入。

## 【4】代码实现

唯一源码改动tests/research_ops_v2/test_gae_credit.py；contextmanager用unittest.mock.patch.object显式限定BudgetStore.status读取，每次deepcopy固定快照，退出自动恢复。两个历史正例和一个stop/forged历史负例使用它；真实proof文件与生产函数不patch。新增两个测试覆盖context结束后reader恢复/实际账本不变、真实R24FAIL拒绝；其余145项运行/测试身份记录全部原样。不是隐藏monkeypatch，不改授权或运行时门禁。

## 【5】算法及验证原理

本阶段没有新PPO算法。可变外部状态不能作为历史单测的未声明输入，否则新失败会遮蔽本来要测的分支。unknown/pending、stop、forged分别断言unknown-failure、stop-latched、caller-budget-digest错误；不能只看到任意OpsError就算分支已测到。历史快照不是真实预算，不能由它启动研究任务。

## 【6】协议与身份

AUTORL_R25_LEGACY_FIXTURE_ONLY，Level1，无训练/新科学episode/Validation/Test；单CPU测试进程，受控物理/推理回归另列。REG02 `eb2b52eabb9a99f55c3df18e432dc3acfb80ec1de7458fdde1ddbe54cfadfb8d`，146SOURCE02 `3bd5f4408876611b08234b4fdb1f3eed7d52846d70cbbe5e7192dd0910682418`，L1协议02 `42091390b36aff092a4fbd2cb6511d0eab08f5f3a146dbdfb4f143186026bfdb`。固定R22报告 `ea1f49e4df91af36b158f318022137e8fa76e0d39ec14ba10b76a219509ba8b5`，14历史attempt/1082496步/36000秒；实际head30/1016960步/32400秒不可被它替代。原7kg动力学、PI、CNN、Reward、PPO参数、扫描模式及旧checkpoint均未改。

## 【7】实际结果与资源

初始定向4例真实RED=1FAIL/2ERROR/1PASS→修复后11模块GREEN。独立早审Important1指出forged负例被R24状态遮蔽：增强原因断言后实际1FAIL RED，再隔离历史负例→11GREEN02。首候选整套1183已完成且保留，但测试SOURCE变化后不用旧收据准入；重冻02并完整重跑1183PASS=224OPS+74无更新Tensor+885历史，15排除，optimizer0，非全仓。

363授权文件含head字节未变，11额外R24证据未变，145项非本测试源码身份未变；没有新训练protocol/attempt/checkpoint/预算事件。真实current gate仍拒绝R24FAIL。原R24的220PASS/2ERROR收据、16raw、中间ZIP、失败MD/JSON均未覆盖。

OPS回归wall15.546s、主进程HWM30.324MiB、子进程HWM30.324MiB、端点最小MemAvailable10187.113MiB；Tensor回归wall101.974s。这些不是训练吞吐或完整阶段瞬时资源峰值。已知可选cv2/NumPy及Axes3D警告保留，无全局升级。没有新科学点云/轨迹图，不复用旧图冒充新实验。

## 【8】问题分析

CONFIRMED：可变账本导致旧正例误报；任意异常断言会遮蔽防伪造分支。真实RED与精确分支GREEN已证明修复。UNVERIFIED：导航能力提升、R24最终效果、真实关机清理保证；本阶段不研究这些。初次prepare日志的syncfiles计数字段显示了authfiles数，实际注册清单始终11项；最终以JSON而非该显示字段为准。

## 【9】局限性

历史reader替身不是真实当前训练准入；真实proof文件仍须本地保留，未声称全套完全无历史依赖。训练门禁继续拒绝R24，未实现新训练放行。EXACT/BOUNDARY_RESUME未实现，原L9/R20/R24失败保留，不从旧ZIP续训。

## 【10】下一阶段

完成独立复审/正常Commit/OPSsync远端后，核预算/stop/资源，优先设计一个独立无训练的中间checkpoint诊断，判断中断前学习是否已破坏短距离停稳；必须另冻结输入/checkpointSHA/范围并审查，不把它回填成R24完整结果。若需要新训练，必须先有独立新协议和严格已关闭失败证明，不能靠本次单测PASS放行。

## 【11】给项目负责人的总结

这次修的是“历史测试如何正确读历史”，没有重训，也没有改奖励或飞控。两项旧ERROR不再因为账本增长而误报；反过来，当前失败实验仍挡住旧训练入口，防伪造余额的分支也真正被测到。预算仍1016960步/32400预留秒。本阶段无算法成绩提升结论，后续先做有限诊断，而不是盲目把中断训练重来。

最终独立只读复审GO，C0/I0/新增Minor0；只认证本阶段测试隔离修复交付，不认证新诊断/训练、R24重试或ZIP续训。JSON仅追加真实review/recovery/doctor与audit引用，统计不变；封口前后SHA分别记录。11文件显式白名单Commit/正常OPS sync远端状态以实际收据为准，不能从本段预先认定已推送。
