# AUTORL-R9：短目标解析Actor反馈可完成性

## 【1】目标

在12个冻结0.20m短Train任务中检验解析Actor均值是否能满足原到点停稳标准。不是训练、不是复用R8权重、不是证明学会LiDAR避障。

## 【2】技术背景

R8已完整发布2c731100f0c8888ffc7b0bf30f0a16c6d923895a，65536步/2048Adam但三科学标准FAIL，短21条episode进过目标球但0联合成功。不能唯一归因Reward/学习率或后半课程遗忘。R5 Scripted成功不等于同结构神经网络反馈已经验证。

## 【3】架构

独立OPS登记与当前source→锁/停止/资源/历史检查→唯一180s保护的非学习进程→R8工厂/R7 adapter/R5短任务→完整Health/TrainingMetrics→新随机CNN/Actor/Critic→解析Actor均值赋值→deterministic predict→原PI/MuJoCo/raycast→实际trace及终止。无训练permit、无预算预留、无Validation/Test。

## 【4】代码实现

short_prior.py负责固定spec、严格新随机初始化、成功trace证据契约和当前回归准入；short_prior_runner.py只提供固定独立目录一次性入口；short_prior_tests.py验证真实模型/实际step与双身份记账；纯tests验证伪成功/不全覆盖/非法时间等拒绝。不修改原环境/Reward/控制器/动力学/R5/R7/R8历史执行源。

## 【5】算法原理

复用既有ANALYTIC_STATE_MEAN_TANH_V1，四条通路实现10*tanh(.5*tanh(.2*y))，y=[ex/4.5,ey/4.5,ez/3,yaw_error]，输出由SB3 deterministic predict裁剪[-1,1]。log_std初始log(.3)仅保持同R3初始化身份，确定性推理不采Gaussian。原CNN/Critic未训练；此初始均值不使用LiDAR，不是感知避障网络。速度缩放仍只在原环境发生。

## 【6】实验协议

AUTORL_R9_SHORT_PRIOR_FEASIBILITY_ONLY，seed0/CPU1/12Train/375步每episode/180s总保护；原distance<.10m、speed<.15m/s连续5step，全部12成功才可完成性PASS，collision/time_limit仍继续固定覆盖，physical/sensor/未知终止立即停止不重试。500/100/25/10Hz/24000rays不变。最终REGa958ac5fc2fe3bcbc5c64b5267bf4a598f61697ec8d7e9a82dfa17ed48510532，SOURCE35a4efcf224a1a0e9b7c4c878e1948d139b174007c5e93798585ac4e3742772b，protocol2bf3f221a6a4a9f9ec3e481b8aac8e2ce4968df97f0a9616eae24ceac5439766。授权余额1,541,248步/61,200s，本阶段不消耗优化器预算；运行前后整个授权目录169文件SHA不变。

## 【7】实验结果

实际可完成性PASS：12/12真实MuJoCo短Train全部success，无collision/time_limit/physical/sensor。58–59策略步/2.32–2.36s每episode，701实际测试策略步/14020物理tick/合计28.04s仿真；训练步/optimizer/Validation/Test均0。初始与最终全policy身份6312e99c…一致，实际weight hash每步检查；不是旧checkpoint初始化。原始result SHA3abcb783342e3f4429904a1a2188a0741e297df3eece98315af17589f61915d1，trace与complete/attempt/health封存local-only。

最终误差0.078588–0.083641m、速度0.097659–0.109470m/s，末5个真实采样均满足原联合条件；不是目标中心精确命中、速度归零或长期悬停证明。每条轨迹的源目标、位置/速度、动作/奖励、真实ticks及LiDAR年龄可重算；12reset/288frame，其中276条独立新增scan/encode性能样本不含reset，不把701个policy step误称701帧。所有年龄<=.08s，无未来观测。三图为实际trace，离散时序用阶梯图，目视无裁切。

worker wall15.256s/CPU15.365s/HWM1669.391MiB；runner每step前置采样最低MemAvailable8704.117MiB，Health内部更密集采样的累计最低值在512/640步记录为8608.676MiB。两个口径不同，不能把前者称全采样最低，也不声称未采样连续最低值或整个Codex会话HWM。环境step P50/P95/P99/Max=1.383/12.993/14.335/14.962ms，scan6.764/7.124/7.365/7.571ms，encode4.346/4.527/4.791/4.939ms。未观察OOM；与R5单独Scripted耗时非严格性能对照，R9含网络、完整记账、每步身份与权重验证。

测试：最初新纯6/真实1先RED；两次GREEN尝试分别暴露测试夹具缺独立诊断目录、错误假设4D action范数应小于.1（遗漏原任务大Yaw误差），仅修夹具和逐轴解析断言，不改任务Yaw或动作尺度。首次独立预审NO_GO I1发现物理/传感器失败未停止剩余覆盖；新2项RED→GREEN修复后，_03完整118OPS+34Tensor+885历史=1037项选择PASS/15排除/2既有warning/0optimizer。旧_02收据仅保留历史，不能替代当前源码准入。最终预审C0/I0、新Minor1/继承Minor1条件GO后完成登记/recover/doctor/资源/169历史身份核验，才执行唯一run。运行前后源码35a4ef…未变。独立结果复审和Git远端以真实机器报告/OPS收据为准，不提前认证。

## 【8】问题分析

CONFIRMED：R8科学FAIL且已完整执行；R9手写控制先验在同网络/同短目标中12/12可完成，表示任务/接口存在可成功控制反馈，不代表随机探索或学习已解决。SUSPECTED：短任务的成功反馈可能有助于保留初始先验，相对R3原目标直接训练的先验丢失值得检验。UNVERIFIED：初始化能否改善PPO或泛化，必须另立新训练协议及对照评价；不唯一归因Reward、学习率或探索噪声。

## 【9】局限性

单seed/小型Train/非学习先验，不能外推真实矿井导航或证明LiDAR被使用。粗机体接触几何TEMPORARY_PLACEHOLDER。无PX4/FASTLIO/实机/逐点真实时序。每episode终止后才原子落盘逐步trace；普通异常尽量保存内存trace，但SIGKILL/断电可能丢失当前未结束episode，必须INCOMPLETE，不承诺完整快照。新增Minor：既有json_safe的512节点总限制可能只保留trace前缀、截掉后置last_returned_transition，不能保证最后transition必有；这种异常不可能据此PASS。R7已知Minor继续披露，不认证任意内部状态结构破坏的故障恢复。

## 【10】下一阶段

完整结果、独立复审、Commit/Push/远端核验后，设计R10独立课程PPO：相对R8只改Actor解析初始化，保留同65536步/3600s预算、std.3/32768短目标后自然reset换原任务/原算法和Reward。R3已有原目标随机/先验两臂、R8短课程随机，本候选补上短课程先验交互项，不重跑前三个ID。必须区分短目标先验保留、相对R8收益和原目标相对自身初态的真实学习增益，不能要求天花板6/6再严格提升，也不能把手写先验当学习。先新冻结/测试/独立GO/预算核验，再唯一执行，不直接开训。

## 【11】给负责人的总结

这一步没有训练：同一个网络中的手写到点反馈，在12个短任务都通过了原到点停稳容差。它的终点仍有约8cm误差和0.10m/s速度，不能理解为完美静止，也不是PPO学会避障。下一步检验这种初始能力在短课程训练中能否保留并产生原任务的真实学习收益；目前只是候选，先审查和冻结协议再用已有预算。预算不变，旧失败和模型不动，普通研究继续自主推进。
