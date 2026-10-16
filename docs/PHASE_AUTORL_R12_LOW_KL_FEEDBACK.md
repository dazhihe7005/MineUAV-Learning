# AUTORL-R12：低KL训练后的固定反馈诊断

## 【1】阶段目标

R11降低学习率后仍未保留短目标能力。本阶段不训练，只读R10/R11新冻结模型，检查零误差偏置、目标方向和速度敏感性，区分较小单次PPO KL与累计反馈变化。不将有限输入推理当导航性能。

## 【2】技术背景

R11已推bd9095ff2f1b16a1bcd89a8ae699f4df47d1a05a，65536真实步、四导航标准FAIL/机制PASS。其前32KL.002472较R10.071368显著下降，但最终两组Val全碰撞。不能据此继续盲加步或确定唯一根因。R4旧11模型已完成，本轮只分析4份后来产生的新32k/final模型，不重做R4旧实验。

## 【3】系统架构

新V2登记/源码/独立GO→独立零训练runner及flock/parent guard→读取R4封存NPZ→固定62反事实state→4个只读checkpoint→原infer_panel→本地证据/纯分析/图。没有环境实例、新raycast、物理step、rollout、optimizer、Validation或Test。

## 【4】代码实现

feedback_followup.py固定R10/R11/R4 report/actual review/all raw、summary/complete/sidecar/ZIP/policy和实际原背景身份；load_background不生成替代数据。feedback_followup_run.py只允许固定phase/协议/准入及唯一目录，所有grant文件前后全SHA相同，原PPO.learn/train/Adam.step tripwire拒绝更新。保留原infer_panel对权重/optimizer/Torch RNG的逐panel不变性检查，不复制推理实现。旧OPS/grant/原环境/CNN/Reward/PPO/PI/动力学完全不改。

## 【5】算法原理

同R4：LiDAR[3,12,24]+state7+heading2+age1；背景来自真实历史reset的24000射线，原采样时间0不重写，不假装新观测。62分组输入/59不同state，仅替换明确OFFLINE_COUNTERFACTUAL_INPUT。输出均值/std/裁剪动作及Critic；换算ENU[1.5,1.5,1]m/s和1rad/s只声明接口单位，不发送控制。

指标：零state速度指令范数；X/Y/Z非零误差且零速度的同向比例，分母6/6/4；+.1m误差下正负速度输入的同轴指令差；yaw±.5差/1rad；有限panel的raw均值触边比例、相对Scripted最大归一化动作差。不是物理漂移/减速度/安全间隙或成功率。Critic没有配对return真值，不认证拟合。

## 【6】实验协议

AUTORL_R12_LOW_KL_FEEDBACK_ONLY，CPU1/guard120s，唯一.research_local/autorl_r12_feedback/run_01，0训练预算预留，既有余额1410176step/54000s必须字节不变。模型仅R10/R11的32768/65536，4×62=248新响应；R4 R3 PRIOR step0的62条是历史参考，不重复加载推理，不混成310新响应。所有模型身份、源及预算快照在新协议冻结，stop/Mem<1536MiB/磁盘<2GiB/源码或旧证据变化即停止，不重试attempt。唯一允许的已封存FAIL是R6，不将未决/未知状态放行，不修改旧R4的限制。

最终_02 REGb54713cf6ec095b728787a35c4e1af59a439d7f8fa13c87f7f811425f829219a、SOURCEc40fa23285cb672db4587cb0a67d3fb3739d28de063b00e54a643aae9d1fe8c3、execution2b19c1224893334325e3769f28cba00a7699a1495e2c1bffdfaa369c621146ca。有限panel先验保留要求：零state速度范数≤.05m/s、最大归一化动作差≤.05、三轴固定误差全同向、速度差相对原零敏感性的绝对误差≤.05m/s、yaw差>0。分别PASS/FAIL，不等同导航；真正避障策略可以合理偏离Scripted。

## 【7】实验结果

唯一run工程PASS，result SHAd4578edbd556c10e34189376c84b5554b46799318536e15d8e02c583906b771b，真实attempt/result/complete与launcher exit0、进程组清理均核验。实际4新checkpoint×62输入=248新响应；62旧初态响应明确复用，59个不同state，没有新物理/扫描/episode/Validation/Test或optimizer。所有4panel的权重/optimizer/Torch RNG不变；4个原checkpoint/sidecar/全policy和95个SOURCE文件执行前后均匹配。

| 模型 | step | 零state速度指令范数 m/s | X/Y/Z目标同向比例 | yaw差/1rad | panel raw触边元素比例 | 有限panel保留 |
|---|---:|---:|---|---:|---:|---|
| 初态先验（历史复用） | 0 | 0 | 1 / 1 / 1 | .995856 | 0% | PASS |
| R10原LR | 32768 | .381546 | .8333 / .6667 / .5 | .068759 | 24.60% | FAIL |
| R10原LR | 65536 | .739462 | .6667 / .6667 / .5 | .806882 | .4032% | FAIL |
| R11低LR | 32768 | .294207 | .8333 / .6667 / .75 | 1.015547 | 0% | FAIL |
| R11低LR | 65536 | 1.190841 | .6667 / .5 / .75 | 0 | 26.21% | FAIL |

全部是固定历史背景的反事实指令，不是无人机实际漂移、角速度或导航成功率。四新模型最大归一化动作偏差相对Scripted分别1.431241/.851896/.235562/1.5。R11在32k比R10更接近原函数，但本panel final反而偏离较大，不能外推全场景优劣或唯一根因。R11 final零state原始均值[.361208,−.697365,−.174141,2.107650]；yaw裁剪为+1，五个±.5内yaw输入均裁剪到+1，所以响应差为0。这是策略输出饱和，不是电机饱和或PX4控制分配故障。

三轴正速度−负速度的同轴指令差：原初态均0；R10@32768=[+.007160,−.183818,−.058607]、final=[−.313035,−.162312,−.047306]；R11@32768=[−.047160,−.062817,−.009690]、final=[−.039404,−.039095,−.010444]m/s。它们只是输入敏感性；即使符号有减速倾向，仍有大偏置和错误目标方向，不能单凭该符号证明物理制动或安全。

inner wall6.90997s/CPU6.09171s从imports之后计起，含加载、推理和结束核验；完整launcher wall14.02837s还含启动前身份检查。whole-worker HWM486.340MiB含imports及模型加载；四次模型前资源探针最低MemAvailable8279.930MiB，不是全程连续最低。62输入batch的均值/Critic/predict耗时4.321/3.027/2.811/2.841ms，不当单策略step或端到端25Hz预算。未观察OOM；没有新测扫描/物理性能，原R4更高RSS不当作本轮改算法节省内存的证明。

全部221个授权目录文件（含head/18events/全部旧run）SHA前后逐字节不变，无新reservation/退款/初始化；累计589824step/32400s保守预留，仍余1410176step/54000s、stop=false。52个R10/R11/R4已发布raw清单及实际预审/结果审查引用均核验。3个新raw local-only，console/exit另有SHA，四张图来自实际新模型输出，主代理目视轴/图例无裁切；没有手工造响应点。

新9纯契约/2数组来源RED→GREEN，_02完整140OPS+38Tensor+885历史选择回归=1063项PASS、15逐项排除/2既有warning/0测试optimizer，不称全仓。3纯分析方法契约GREEN02，复算旧R4基准数值与原分析一致，不计新实验结果。_02独立预审C0/I0/新Minor0/继承R7 Minor1 GO、方法独立GO之后才新登记/recover/doctor、预算/stop/resource/protection检查和唯一执行，SOURCEc40fa232…全程不变。实际结果复审与Git远端认证仍以真实独立收据和OPS收据为准。

_01候选及其139/38/885完整收据、旧独立GO均保留，不用于_02。主代理新增分析器基准自校验发现概念Important：原Actor只取位置误差与yaw误差（features64/65/66/70），无速度输入；Scripted也是误差型指令。旧R4原先验三速度差恰为0，初始候选却要求严格负，故基准自己不能通过。分析green01文件实际有1失败，名称不作PASS依据；已留原始失败。真实执行前明确修订判据/新身份，新增pure RED→GREEN，独立审查确认旧GO遗漏问题，不借旧GO运行新代码。所有旧候选源码/设计/公开草案复制到本地只读archive，旧实验结果不变。该修复是方法校准，不是看新模型结果后降低阈值。

## 【8】问题分析

CONFIRMED：低LR显著降低单次PPO近似KL，但新冻结模型在相同panel中仍有明显零误差偏置和方向响应偏离；这与R11实际Val失败一致。原先验没有速度输入，判据错误已事前校准。SUSPECTED：无约束的可训练Actor偏置/背景通路可积累成过大的常量指令，可能是比继续盲调LR更具体的干预点。UNVERIFIED：这是唯一根因、与实际物理漂移的一一因果、任何网络约束能提升导航或真实避障；偏离先验本身也不必然是Bug。

## 【9】局限性

一个历史LiDAR背景、有限反事实输入，不是连续运动、通用稳定性或真实场景泛化。历史初态只作基准，load模型可能消耗进程局部RNG，只有各infer_panel内部的Torch RNG保持不变；无训练环境存在，不宣称精准续训。继承R7任意active_task结构损坏诊断Minor与本无环境推理路径不相交。cv2/NumPy可选ABI及Axes3D警告照实保留，不改系统依赖。封存Test、PX4/FASTLIO/实机仍不触碰。

## 【10】下一阶段

先实际结果独立复审/历史audit/Commit/OPS sync远端，不重放本诊断。下一优先R13离线Actor输入状态归零锚定方案预检：同一当前LiDAR/heading/age下构造内部零state参考，研究mean=f(obs)−f(same-current-background,zero-state)以抑制常量偏置，不加障碍真值/未来信息、不动CNN/Reward/动力学。先证明新初态等效、Gaussian/梯度/序列化与原控制接口正确，0optimizer；若可行，再另立新协议对照R11做PPO，而不是直接复用旧模型继续训练。锚定零指令是结构约束，不是学习成绩，真正导航增益仍须独立物理Validation。安全/未知状态/停止/预算/会话限制先交接。

## 【11】给负责人的总结

这次没花训练预算，固定同一份旧雷达背景检查新模型，发现即使每次PPO变化很小，训练后仍可能在“已到目标且不运动”的输入下输出很大的速度指令，最后偏航输出也饱和了。我们确认了更具体的症状，没有证明唯一根因、更没有学会避障。测试还先纠正了自己的判据错误，旧失败记录保留。下一步先离线验证“把零状态输出作为参考相减”的网络表达，先测可靠性，再考虑预算内独立PPO对照；不直接重跑或增加预算。
