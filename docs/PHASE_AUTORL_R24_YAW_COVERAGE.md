# AUTORL-R24：训练起始Yaw覆盖实验中断报告

## 【1】阶段目标

检验只扩展Train起始Yaw能否改善开发Validation停稳/避障。实验执行FAIL/INTERRUPTED；科学五判据均NOT TESTED，不把中断当Yaw假设被证伪，也不伪造完成。

## 【2】技术背景

R23描述性发现empty Val与同类Train初始朝向覆盖差异。控制为完整R17，非λ=.99的R22；保持γ=.99/λ=.95，只改Train起始Yaw。R22/R23均已发布，不重跑。

## 【3】系统架构

固定CLI→独立累计预算/独占主管→原PPO worker+reset级Yaw调度→原7kg MuJoCo/PI/24000rays→原计划关闭Train后独立Val。本次系统重启导致主管终态缺失，改走外部只读posthoc→独立处置审查→BudgetStore.fail，不执行原COMPLETE分析或finalizer。

## 【4】代码实现

新增yaw_coverage纯120任务契约、env/profile/cli/无更新tests；budget/runtime/worker仅固定schema24分支。12几何/start/target不变，每个五个绝对ENU朝向[原Yaw,0,+π/2,π,−π/2]各1/5，非原Yaw加offset。Yaw用独立Random(seed+0x52323400)，不消耗base NumPy RNG；仅reset设初态，正常step真实mj_step。新任务/base/distance/目标/实际Yaw分开记账，不冒充原任务。旧schema17前缀一致检查仍仅作用17。

## 【5】算法原理

同中心化CNN/Actor-Critic170857参数、手写先验/std.1、LR3e-5、PPO n_steps128/batch64/epochs2。Reward不改：10Δd−.005||a||²−.5w(d)||v||²+10success−30collision/physical−10sensor−.01step−2timeout；成功仍d<.1m、speed<.15m/s连续5个25Hz样本。新初态覆盖不等于网络获得场景真值，不向其输入任务标签、地图或未来轨迹。

## 【6】冻结实验协议

ARL_R24_TRAIN_YAW_COVERAGE_SEED0，fresh seed0/CPU1/65536步/3600秒全额预留。REG `f10adedc274d5e12baa9d69fd4d3b0fe2f5084fd4764e8940fe0c68e741f6df3`；146SOURCE `3b2b6d7ae0bb668c22c8eab1c0db7151d7afc020b37035a0047a9af04a5042cf`；EXEC `3595c1be7190fa6288439547ab2a472fd2c1cc8f2f783654ac25ad23d56b038a`；120任务manifest `d8da5814caccebb22373fea6ca4f8b66a0fe58c03c9cc928ffb47e1571de1010`。500/100/25/10Hz、24000rays/原动作尺度不变。short至32768后的自然reset才转原距离，base empty抽样.5；Final Checkpoint，计划36新Val、Test0。

## 【7】实际结果、测试与资源

旧boot `75a72268-5a3c-471f-aa58-e35f9f4b0f6a`，新boot `189bfa8c-5d0c-4857-82e2-b221364d91d7`；精确_BOOT_ID的Oct17 02:36:32 UTC日志确认System Reboot/Shutting down/Sending SIGTERM。最后更新mtime02:36:30.530 UTC，时间接近不证明worker具体信号。已无原worker/主管/进程组/锁；summary、训练complete、主管终态、final ZIP均缺。393条完整更新到50304步/1572Adam/786epoch；较旧progress50176不能冒充最终进度。243个结束episode到50213步，精确退出步数/物理tick/退出码未知。

short170={'collision': 19, 'success': 131, 'time_limit': 20}，原距离73={'collision': 22, 'success': 22, 'time_limit': 29}，自然切换32874。成功共153、碰撞41、超时49；仅中断训练前缀，不是final策略成绩。两个0/32768 ZIP只读CRC/SHA/元数据PASS，不在中断后加载模型、执行新Val或续训。32k sidecar的170个实际reset与169个结束episode绑定已核；之后缺最终reset快照，不能伪称全程测得Yaw逐项完整。新Val0/计划36、Test0；五判据均NOT TESTED。

训练前1181选择PASS=222OPS+74无更新Tensor+885历史，15排除，不是全仓。开发step0 wrapper接口ERROR与预审分析器Important均真实保留并启动前修复；新增5+4 RED→GREEN，分析3/finalizer2 GREEN02另列。重启后222OPS实际220PASS/2ERROR：两个旧GAE正例直接读取变化的真实账本，再断言旧closedfailure门应接受，新增R24被正确拒绝；不隐藏，不删检查，不把旧PASS冒充新boot全绿。独立74无更新Tensor全部PASS；未重跑885历史。所有测试optimizer0。旧cv2/NumPy ABI可选导入警告在02:19已出现，此后有393完整更新，不据此归因重启；MatplotlibAxes3D警告保留，2D出图。

最后前缀资源wall1043.174s/CPU987.717s、记录HWM1643.777MiB、累计采样最低MemAvailable8627.852MiB；完整wall/全程峰值/旧boot OOM排除/全体并发RSS未知。384旧文件及16原raw不变；仅29RESERVE/30FAIL，保留65536步/3600秒全额收费，余额1016960步/32400预留秒stopfalse，未退款。

五图均由实际前缀生成；Validation图明确缺测而非虚构曲线。原raw、journal及ZIP local-only并记录路径/大小/SHA。本报告是POST_HOC，不伪称运行时捕获到了不可用的traceback、最后transition或reward。

## 【8】问题分析

CONFIRMED：跨boot系统重启与日志终止，独占进程消失，完成证据缺失；两旧测试正例依赖可变账本。SUSPECTED：重启生命周期提前终止主管清理；具体worker信号/杀死顺序无收据。UNVERIFIED：重启发起者/原因、精确最终训练步数、全程OOM排除。R21只best-effort SIGTERM，非系统关机/SIGKILL保证，不能因已加handler宣称关机可靠完成。

## 【9】局限性

Yaw科学对照未完成，不能与R17/R22final作同条件优劣结论。手写初态不是学习成绩；小开发Val非Test/真实矿井泛化。RESTART_REQUIRED，不精确续训、不退款、不旧ID重试；原L9/R20/R24失败永久保留。重启后测试问题尚未修复，新训练NO_GO，未关闭安全门。

## 【10】下一安全阶段

先完成中断证据独立结果复审、Commit/正常OPS sync核远端。随后仅Level0/1修历史正例的冻结fixture/可变账本耦合及闭合R24处置证明，新增拒绝未知失败回归和独立审查，再核授权/资源。不得重放R24、从32k ZIP继续或直接开始另一训练；不运行Test/PX4/FAST-LIO2/真实设备。

## 【11】给项目负责人的总结

这次训练不是算法跑完失败，而是在约5万步时遇到系统重启，完整结果没产生。已有153个训练成功episode不能证明最终学会避障；没有最终Validation就不下科学结论。原始文件和预算如实保留，训练不会偷偷重来。下一步先修两项受账本变化影响的旧测试并审查失败封口，仍在已有开发权限内；真正新训练需要新的独立协议和准入，不增加总授权。

最终独立只读复审：GO_REPORT_ONLY，C0/I0/新增Minor0，继承R7/R18/R20 Minor保留。仅同意如实发布FAIL/INTERRUPTED报告，不豁免重启后2ERROR、不放行新训练、不标实验或phase_complete成功。真实postboot recover/doctor/OPS文件审计收据已追加JSON；统计/五图/运行源码/原raw未改。资源图mem_available_mib为记录的累计采样最低值，不是每次瞬时可用内存。封口前后SHA分开记录，Commit/远端身份只以实际OPS同步收据为准。
