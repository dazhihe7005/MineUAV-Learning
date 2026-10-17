# AUTORL-R28-PUB：测试格式与发布身份修正

## 【1】阶段目标

修复真实Git whitespace发布阻断，保持已执行训练身份可追溯。本子阶段只做发布修复，不训练、不新增导航episode；Commit/Push尚待实际命令，未标PHASE_COMPLETE。

## 【2】技术背景

R28唯一65536步与36Val已完整完成、科学FAIL、独立结果复审GO。旧seal已正常追加审查metadata和显式stage28文件，但实际cached diff检查exit2：test_held_goal_entry.py:41有额外EOF空行。因此没有Commit或Push；不能关闭检查、静默删后冒称旧SOURCE仍一致，或把工程阻断隐藏。

## 【3】系统架构

只读保留旧训练/OPS archive → 精确测试格式差异 → 新publication-only REG → 新身份选择回归 → 独立GO → 正常register/recover/doctor/audit → 31文件真实working/cached检查 → 单独Commit/正常OPSsync/远端SHA。旧R28报告和五图作为不可变实验附件一起发布，不重复seal或科学分析。

## 【4】代码实现

唯一源码修改是测试末尾移除1个额外LF：3198→3197字节。old SHA `9ad406b9a1b85de9ae438bd52758f8429ab39bfff40764ac4f6c247df457cb1c`，new SHA `d4286b34601d3c68b5a4a4b0e341f3d125ba8e677b1e817812922750c1f5d41c`。包含行号的完整AST dump相同，另外163个SOURCE记录完全不变。没有改worker、预算门禁、模型、Reward、PI、传感器或动力学。

## 【5】原理与边界

Git检查针对具体字节，OPS源码身份也针对字节；语义不变不能冒充SHA不变。保存旧字节与已执行SOURCE，再登记新版本，才能同时保持实验可重复与发布检查有效。第一次本地准备误把no-index差异退出码1当应为0，未写新注册；随后改用独立临时GIT_INDEX_FILE，在真实diff --cached --check下得到exit0，不修改用户真实索引。旧actual staged RED exit2保留；31文件最终检查仍须真实执行，绝不忽略blank-at-eof。

## 【6】版本化协议

原训练REG `cc5a35abd5d6a965945a42c7fe0584946bc1df37a98941c4c61e82ded9a604dc`，as-run SOURCE `94daaf6b68f5db9977ede2c5db14660a067549f1e28d41b8ee48fb29aada3fc3`，EXEC `bc2f81ffa7e7fb67ec46a6afe31255300f6ff86ce782168edc3f648e043db3f7`，永久保持。新发布REG `53431c954f89f8caa0f94b3aefd23c6cb9d3e0b3c231fab71ca8c5e99c2d2ccc`、SOURCE `e3e61d7d7e8cebfbf80498402d2a571e8c80ff0362376431f112f4e8e575e132`，新阶段AUTORL-R28-PUBLICATION-FIX，仍零OPS训练预算。真实独立授权余额951424步/28800预留秒stopfalse、head32，完全不变；不是重新授权。RESTART_REQUIRED不改。

## 【7】实际验证结果

新SOURCE完整1221选择PASS=253OPS+83无更新Tensor+885历史，15排除；未中断、optimizer0，非全仓。EOF实际RED→GREEN与AST相等证据在JSON；新注册31白名单文件，462历史/授权/已完成R28证据不变。原R28科学结果仍3FAIL/2PASS：held原距离4/6成功→0/6→0/6，short始终6/6。初始4个成功来自手写先验，不是学习成果。本发布子阶段无新增模型、训练、科学评价或资源性能实验，不复用旧证据冒充新学习。

## 【8】问题分析

CONFIRMED：发布失败来自新增测试EOF格式，不是训练/物理失败；未绕过身份或whitespace门禁。原执行代码完整归档，当前测试格式版本明确不同。SUSPECTED：无新增算法根因结论；长距离能力下降机制待R29单独诊断。UNVERIFIED：真实Commit/Push/远端，须收据；不得预先宣称完成。

## 【9】局限性

当前整个SOURCE不与原164字节快照完全相同，仅测试格式差异有明确证明。训练运行时代码未变，因此不把测试LF修正谎称为训练后运行时代码修复，也不重训。旧R28 OPS状态不手工改已同步；新版本正常发布后才可从最新权威状态交接。科学FAIL没有因发布PASS而变成功。

## 【10】下一阶段

实际发布并核远端后，核budget/stop/resource，正常登记R29无optimizer同输入策略函数诊断。重用可信checkpoint与已存观测，反事实状态probe要明确标注，不当作闭环轨迹；不重放已完成36Val、不重复R28、不用Test调参。

## 【11】给项目负责人的总结

训练已经跑完，但仍没有把短距离能力迁移到更远目标。发布时又抓到测试文件一个多余空行；这次没有掩盖或强推，而是保留旧执行版本，用新版本登记和完整测试修复发布。没有多花训练预算、没有动历史模型，不需要重复批准普通开发。下一步仍是先查反馈映射，而不是盲目重训；最终GitHub是否同步以真实远端SHA为准。

最终独立发布复审GO，C0/I0/新增Minor0；仅认证测试格式差异、新版本/完整1221回归和正常OPS审计，不将原R28科学FAIL改为PASS。统计/raw/图/协议与旧注册保留。31文件实际whitespace检查、Commit、正常OPSsync和远端SHA以真实后续收据为准，本段不提前认证它们。
