# AUTORL-R13P：已跟踪证据图的OPS发布修复

## 【1】阶段目标

修复R13提交后、推送前出现的真实Git路径阻塞，不改实验、不重跑网络诊断，不开始训练。R13 commit8492143保留，原审查和实际结果不变。

## 【2】技术背景

历史.gitignore含mujoco/reports/*_pre*/。两张已明确登记、审查、跟踪的小型PNG位于autonomous_rl_centered_preflight_r13目录。Git对已跟踪文件在被忽略父目录下执行普通git add仍返回1；git add -u同一精确路径返回0。原OPS sync因此在暂存阶段退出，未到push/远端核验；绝不将其称已同步。

## 【3】系统架构

新R13P版本登记→独立审查/当前回归→原历史保护→正常OPS register/recover/doctor/audit→原白名单/制品/提交历史核验→修正暂存→Commit/Push→远端SHA。新版本保留旧R13整个协议/源码归档/状态，不回写旧零预算，也不重新初始化独立训练grant。

## 【4】代码实现

只改research_ops_v2/publication.py：literal ls-files取得已跟踪集合，按注册的精确文件划分existing/fresh；existing用literal git add -u，fresh仍普通literal add。没有force-add新忽略文件，没有改.gitignore、main、远端地址或任一旧保护逻辑。tests/research_ops_v2/test_publication.py新增真实临时Git+bare远端回归，私有未登记文件不入库。

## 【5】原理

这是Git已跟踪文件的更新语义修复，不是算法改进。-u只能更新索引中已有条目；代码只传注册白名单内的existing路径，不使用全仓-u。fresh仍受ignore规则控制。literal pathspec避免文件名被解释成模式。原路径/大小/凭据/PNG CRC/历史/审查/待推提交/远端SHA门禁全部保留。

## 【6】协议与身份

阶段AUTORL-R13P-PUBLICATION-REPAIR，功能分支fix/research-publication-tracked-artifacts，基线8492143，原可靠发布锚点79befa26而不是未发布HEAD。24个显式sync路径覆盖R13原18文件及本次6文件。REGf6befae4779f6458f3c4a800373a6d5bbc7aa0859e22cca9f5c667619173de02，SOURCEc1aceb59783ff1eb1cd922f53f87b1b951e6100303cf0b8779f835be9a0f1280。与R13聚合身份仅两源码文件变化（管理发布实现/对应单测）。

无训练预算预留，optimizer/训练/新研究episode/PX4/Test为0。相关历史回归中的受控物理单测与本阶段无新研究实验明确区分，不能称整个测试进程没有MuJoCo工作。

## 【7】结果与证据

真实Git dry-run确认plain add失败、-u成功。隔离实发布单测RED旧git add OpsError→GREEN；9项publication定向测试全部通过。完整当前145OPS+50Tensor+885历史=1080选择回归PASS、15排除、0测试optimizer，非全仓测试。独立预审C0/I0/新Minor0且精确绑定当前SOURCE/REG/checks660d754e…；register/recover/doctor已实际PASS，历史audit及最终结果复审/实际同步以收据为准。新测试检查忽略父目录下已跟踪白名单文件确实提交/推送，额外未登记私有文件不被提交。

OPS单测进程wall11.894693s/CPU3.439921s、HWM27.840MiB、起止端点MemAvailable最低8521.914MiB；它们不是整个Torch/MuJoCo回归的峰值或连续内存最低。本阶段未采集统一整套进程HWM，明确NOT AVAILABLE；原有cv2/NumPy ABI stderr和两项历史warning继续披露，不改环境。全221授权文件与R13四个raw字节不变，保守预算余1410176step/54000s、stop=false，无新预留/退款。

R13实际原始result1107e3fe…/initialZIP/attempt/complete及报告cece507b…只读，未重跑。R13源中的模型/推理代码保持原SHA；本阶段当前最终聚合管理版本没有重新执行R13唯一实际预检，明确NOT TESTED，不能把旧运行冒充新管理代码验证。

## 【8】问题分析

CONFIRMED：根因是Git对ignore父目录下已跟踪精确路径的普通add行为，不是模型、optimizer、显存或安全停机故障。SUSPECTED：无额外算法假设。UNVERIFIED：后续训练效果及新管理版本下的R13实际预检重做（禁止且无必要）。

## 【9】局限性

只支持已通过全部原门禁且已跟踪的有界制品；新忽略原始数据不会自动force加入。真正远端Push尚须实际核验，失败仍保留SYNC_PENDING。R13结构性零均值仍非学会导航，RESTART_REQUIRED不变。

## 【10】下一阶段

本次完整测试/独立审查/结果交付/正常OPS sync远端完成后，进入新R14中心化Actor65536步单变量对照设计与准入。旧R13实际run不可重复；不从其诊断ZIP或R11 checkpoint继续训练。预算、stop和当前源码重新核验。

## 【11】给项目负责人的总结

网络预检本身没有失败；卡住的是Git发布工具对历史忽略目录的处理。现在用最小改动修复“已审查且已跟踪的图无法再同步”这一问题，没有放开raw/模型上传权限，也没有多跑训练。修复发布完成后继续真正的算法对照，不扩张成新调度平台，不消耗训练预算。
