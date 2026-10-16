# AUTORL-R15：新中心化模型固定输入诊断（工程PASS，先验保留FAIL）

## 【1】阶段目标

只读R14新32k/final权重，核对位置、速度、Yaw反馈，避免把“零state结构归零”与学会停稳混淆。

## 【2】技术背景

R14已完整推1ab46a3d0c857ded3c0f496db0e54261a49b3005；原距离三次Val均1/3/2，短距离6→6→5成功。原Train多数近目标采样速度仍过大，证据支持检查反馈，而非盲增训练预算。

## 【3】系统架构

固定R4真实历史LiDAR/heading/age+62反事实state→CheckpointManager只读重载R14中心化policy→infer_panel→均值/std/value/声明动作→纯数值分析。不推进MuJoCo、不发送控制。

## 【4】代码实现

centered_feedback.py冻结来源/协议/准入；centered_feedback_run.py唯一输出、owned group/flock/父守护/资源检查；新7纯及2数组测试。复用原feedback_spec、feedback_probe和centeredpolicy核验，旧worker/Reward/物理/PI/权重不改。

## 【5】算法原理

确定性均值动作经clip[-1,1]后按[1.5,1.5,1,1]解释为ENU速度及Yaw rate。只是离线声明值，非物理执行。速度响应差可诊断反馈形状，初始手写先验该项恰为0，不能要求严格负才能“保留”。零state均值0为中心化恒等式，不能作为新学习成绩。

## 【6】冻结实验协议

AUTORL_R15_CENTERED_FEEDBACK_ONLY，两个新模型×62=124新推理，R13同类初态62历史复用；CPU1/120s，训练/optimizer/物理/新扫描/episode/Val/Test全0。REG3e4595837231c8bef9085a45c7f723e21983ffcd197a1d9e5c4e5d96ff022969、SOURCE87f8ad7efdb0d4a817208c20299535569efbba653b1b1152bfccb8ab460b512d（108文件）、executionfbd8951ff158c1a7e9daf1bc959b14de6ff335878bfa2a6e9cb78d139f966c2e。固定门槛见prereg/design，结果后不改。所有247授权文件及R13四raw字节前后不变，余额1344640step/50400s、stop=false，不新增预留。

## 【7】结果与测试

新7纯+2数组及4纯分析契约RED→GREEN；完整159OPS+53Tensor+885历史=1097选择PASS/15排除/0测试optimizer，非全仓。checks SHA f1f6dbe60a39ba7cbc67b77005db789246d2c0f75da0bd4bb277a638ef69f62c；独立C0/I0/新Minor0 GO后，新register/recover/doctor及即时预算/历史/资源核验实际通过。唯一run exit0/groupgone，无launcher timeout，complete绑定result SHA85d36387eff30e4ee02e0c4ee8c9bcd398c72e3f2886469ee65e9406dbeac846。

| 固定输入指标 | R13初态历史复用 | 新R14@32768 | 新R14@65536 |
|---|---:|---:|---:|
| 零state声明速度范数(m/s) | 0 | 0 | 0 |
| 三轴目标方向正确率 | 各1 | 各1 | 各1 |
| X正速度减负速度指令差(m/s) | 0 | -0.285470 | -0.432698 |
| Y正速度减负速度指令差(m/s) | 0 | -0.410382 | -0.345414 |
| Z正速度减负速度指令差(m/s) | 0 | -0.075135 | -0.087956 |
| Yaw±0.5rad响应差(rad/s) | 0.995856 | 0.948869 | 0.946739 |
| 相对初态normalized动作最大差 | 0 | 0.215104 | 0.207756 |
| 固定panel先验保留 | PASS | FAIL | FAIL |

速度差是在目标该轴误差+0.1m、其余state分量0的固定面板中计算，X/Y速度±1m/s、Z速度±0.5m/s，不能把不同分母的差当相同导数。表中显示六位小数，精确值在JSON；零state归零是结构。两个新panel均无raw mean触及动作±1边界（占比0），不是训练动作clip统计或电机饱和率。

124条新推理与复用初态62条没有混算；每panel参数/Adam状态/Torch RNG不变，加载本身可能消耗进程局部RNG，边界已明确。只读两个checkpoint ZIP SHA61a200948d9b9f41fb3f1af749ded5e05ae86dcdc926a3a9e4d6cfde71bc0b28、9c4bf63a6dea2cab494c429ea5c658b6280c20358fdd456b8d1a4926312a29bf，不用于训练初始化。

inner wall6.841s、CPU6.014s，launcher14.091s，worker/child-tree HWM486.871MiB；模型前两次Mem采样最小8065.727MiB，不是全程连续最低。62输入批推理分别4.562/3.141ms，仅本次批次，不作为单样本P99。无观察资源保护触发。三图均来自实际推理行，原始3文件local-only，完整路径/大小/SHA在机器报告。实际结果独立复审、Commit和OPS同步以真实收据为准，未通过前不标阶段完整。

## 【8】问题分析

CONFIRMED：在这一背景/面板上，训练后形成初态不存在的负速度响应，位置/Yaw方向保持正确；不像R11旧类final的大常量偏置和Yaw同向饱和。两个panel先验保留FAIL，是因为新反馈与初态不同，不能由此判定Bug或退化。保留旧函数本身不是学习目标；R14物理Val的原目标成功仍1/6、不胜自身初态，这一事实不被离线响应覆盖。

REFUTED_WITHIN_PANEL：“完全没有学到速度反馈”。SUSPECTED：在std约0.3的随机探索下，近目标高速采样影响停稳经验；现有负速度响应未能转化为更好的最终导航。UNVERIFIED：降低探索噪声能否提高真实停稳/避障学习、跨背景LiDAR利用程度、唯一根因。单panel不能证明真实闭环制动稳定性。

## 【9】局限性

单一历史背景和反事实state不代表轨迹、跨背景感知、泛化、FASTLIO或真实飞行能力。既有可选cv2/NumPy ABI和Matplotlib Axes3D警告保留，不升级依赖；旧R7诊断Minor不隐藏。

## 【10】下一阶段

先完成本阶段实际结果独立复审、Commit、正常OPS sync远端；然后新R16预检仅改变中心化策略初始Gaussian std0.3→0.1（仍可训练），其余CNN/Actor函数/先验均值/Critic/Reward/LR3e-5/PPO/课程/物理/24k射线不变。先独立冻结新profile/初态身份和验证初始确定性均值一致、测试与审查，再按原累计预算最多65536step/3600s的新attempt执行。不是把R14 final改std继续训练，不能旧ZIP warmstart；若前置失败先修复，不自动训练。该选择是待证伪假设，不保证改善。

## 【11】给项目负责人的总结

这一步没花训练预算，发现新模型确实开始按速度反向修正指令，所以“完全没学到反馈”不是准确说法。但它没有在原距离任务上获得更多成功，不能报学会避障。下一步最值得单独检验的是减小初始随机探索，让靠近目标时更容易积累停稳经验；这是新实验假设，不是现在就宣布有效。先完成审查和GitHub同步，再在原预算里继续，不用你逐阶段重新确认，不启动PX4/实机/Test。
