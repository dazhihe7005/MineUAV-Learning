# AUTORL-R13P：已跟踪白名单制品的Git发布修复

## 原因与范围

R13实际网络预检/结果复审已通过，local commit8492143保留；OPS sync在git add阶段遇历史.gitignore规则mujoco/reports/*_pre*/，即使两张PNG已显式跟踪，普通git add精确路径仍返回1。因此R13远端尚未同步，不能称阶段完整。问题由真实Git dry-run及隔离bare远端测试复现，不是训练失败。不给旧OPS状态手填成功、不改.gitignore、不伪造旧SOURCE；使用显式新OPS版本处理。

## 最小修复

只改research_ops_v2/publication.py的暂存步骤：literal pathspec核对已跟踪文件，将白名单内的已跟踪项用git add -u；新项仍普通add，不能自动force-add忽略raw。所有原来路径/大小/敏感信息/PNG CRC/历史/测试/审查/提交范围/远端SHA门禁保留。训练预算账本与模型/物理/训练代码均不改。

## 验证与发布顺序

隔离真实Git仓库测试已跟踪ignore父目录的白名单小文件能更新、Commit、Push及远端核对，未登记私有文件保持untracked；原测试保留拒绝私自提交/敏感数据/失败push等行为。先RED旧git add异常→GREEN修复，再当前145OPS、50Tensor、885历史无更新选择回归。新R13P登记清单显式包含R13未发布提交的18文件及本次新管理源码/单测/文档/JSON/公共快照，发布锚点仍79befa26而不是未发布HEAD。独立预审后register/recover/doctor/audit，结果复审，Commit及正常OPS sync远端。无训练/optimizer/PX4/Test。

## 身份与限制

R13旧注册/协议/报告/raw/actual sourcef968…原样保留。修复发生在唯一预检结束后，最终组合SOURCE与R13旧聚合身份不同：最终组合版本重复R13实际预检=NOT TESTED，不重跑旧唯一attempt；R13模型模块及全部实际模型路径源码字节未变，新管理逻辑由当前回归与真实发布验证，不冒称旧运行认证了新管理源码。R13远端发布通过R13P审计后的功能分支包含原提交，必要时额外发布原R13功能分支精确旧commit并核远端，不修改旧OPS历史状态。完成同步后才进入R14新训练设计。
