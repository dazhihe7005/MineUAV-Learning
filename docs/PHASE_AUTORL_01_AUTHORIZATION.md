# AUTORL-01：版本化训练授权与预算执行

## 【1】阶段目标

落实AUDIT-01之后项目负责人的新授权：累计2,000,000训练步、24小时，CPU单worker；先建立可核验入口，再研究基础导航。旧协议不重写，训练不会因一句公开PASS自动获准。

## 【2】技术背景

OPS-V1/V2只支持零预算维护，L9R旧driver仍明确拒绝学习。直接更改这些常量会破坏历史；本次新增独立grant和训练模块，旧L9FAIL、L9Rdenial、L9S与全部原数据保留。

## 【3】系统架构

`用户明确授权 → 新grant SHA → append-only reservation → 共享OPS flock → 固定CPU worker → 原SB3 PPO/真实MuJoCo → 原Checkpoint/Validation → 新证据与账本`。

主管和轻量fork守护进程不是额外训练worker。超时、停止、资源或父进程死亡撤销整个进程组；进程锁覆盖清理。源码/场景/历史身份均有检查，不接受任意脚本、shell或checkpoint初始化。

## 【4】代码实现

`research_regression/authorized_rl/`分离预算、runtime主管、lifetime进程组守护、curriculum调度、worker、baseline、无优化器preflight及CLI。CLI用`python3 -m research_regression.authorized_rl`。仍复用原make_ppo、MultiScenarioTrainingEnv、FormalHealthMetricsEnv、FailureEvidenceRecorder、CheckpointManager和LidarPPOValidationEvaluator，不改历史环境、控制器或算法。

## 【5】管理和算法语义

预算不是episode数。训练步在开始前按完整128步rollout预留；每128采样对应2个PPO epoch、4次Adam调用。唯一实验只能预留一次，换ID也不刷新累计上限。中断或失败不返还；本实现成功也全额保守计费wall，实测时间另列，不凭自报秒数退款。

状态记录和事件原子写、独占创建；hash链与head检查丢失／孤立文件。未决attempt阻断后续调度。未实现EXACT_RESUME/BOUNDARY_RESUME，继续RESTART_REQUIRED。

## 【6】实验协议

本阶段只进行标准库并发／进程故障测试、受限无学习MuJoCo／forward／checkpoint测试及受保护历史回归；实际PPO optimizer=0。用户grant SHA、登记SHA、源码SHA与最终测试结果由机器报告提供。未使用Test。下一阶段两臂各65536步只改变场景调度，细则见设计文档；本阶段不把提案当实际训练。

## 【7】实际结果

已完成：71项V2／授权标准库测试（含25项新增）、49项V1回归、885项相关历史回归，合计1005项PASS；15项明确DESELECTED、2项继承warning，不是全仓。另有6项真实无优化器物理／推理／checkpoint检查PASS。授权机制独立源码／注册复审GO，Critical/Important均0。独立新账本已实际初始化，预留0步，余额2,000,000步／86400秒；本阶段训练optimizer为0。

历史回归峰值RSS1956.10MiB，采样最低MemAvailable9451.66MiB，无OOM记录。cv2/NumPy ABI及已有Matplotlib/Gymnasium警告保留；未修改系统Python／依赖。阶段远端发布以真实同步收据为准。

见`mujoco/reports/autonomous_rl_authorization_v1.json`的原始证据绑定与独立复审。新预算测试覆盖跨重启、2M累计、wall上限、锁竞争、重复／未决attempt、停止、损坏拒绝；真实子进程测试覆盖截止期限、主进程先退出的后代清理和独立守护。

无学习预检真实执行两次策略step（.08秒物理时间），验证课程下次reset与缓存、真实SB3 checkpoint保存／加载、optimizer状态和预测一致性。强制阈值计数仅标记FORCED_TEST_STATE，绝不冒充已训练32768步。

## 【8】问题分析

CONFIRMED：预审发现非阻塞预算读锁可误停，现改为最多1秒的短事务等待；真实并发读／stop已回归。CONFIRMED：旧清理只看worker存活会漏孙进程，改为检查整个进程组并回归。checkpoint暂态train闭包不应被SB3序列化，保存时移除、之后恢复，重载核验真实参数、optimizer、预测及步数。

SUSPECTED/UNVERIFIED：百万步内存稳定性、课程能否提高成功率、真实矿井泛化均未由此阶段证明。

## 【9】局限

当前固定入口只实现PPO两臂新初始化，非任意算法平台；监督更新预算折算及精确恢复未实现。全额wall预留可能比实测消耗保守，不自动退款。请求停止不自动解锁；不保证异常断电后的训练可续。

## 【10】下一阶段

机制通过实际测试、独立GO和Git同步后，初始化独立账本，冻结下阶段登记，运行三个empty Train的state-only Scripted基线。全部达到原停稳标准才预留／启动课程对照。异常不重试，先分析证据，普通研发不重复申请已批准预算。

## 【11】给项目负责人的中文总结

本阶段解决“批准的训练如何防止越预算、重启重复运行、丢失失败证据”，不是提升神经网络性能。授权是新的独立版本，不会改掉原百万步失败历史。完成与远端同步以CLI收据为准；训练步、剩余预算和是否实际学会导航会分别报告。
