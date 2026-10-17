# AUTORL-R26：中断模型的独立只读停稳诊断

## 【1】阶段目标

对R24真实step0与32768 checkpoint，在固定两个短距离及两个原距离empty Validation上分别产生新诊断，判断该短距离子集是否出现停稳遗忘。工程PASS，科学结论仅限本子集；不是R24最终结果，不重训、不续训。

## 【2】技术背景

R24已因系统重启永久FAIL/INTERRUPTED，最后可靠更新50304而非最终步数；未完成65536和原36次Validation，五判据仍NOT TESTED。R25仅修历史测试fixture，不能解锁训练。本阶段新8episode不回填旧实验。

## 【3】系统架构

已发表失败证据与真实FAIL账本→可信ZIP/sidecar核验→CPU只读反序列化→原stopping_trace→真实MuJoCo/PI/24k射线→JSONL→独立数值复核与图。两个模型串行、单worker、每case≤375步、总600秒；learn/train/Adam.step直接拒绝，权重/optimizer/num_timesteps前后相同。

## 【4】代码实现

新增interrupted_yaw_diagnostic.py、_run.py、_tests.py及OPS契约测试；旧146SOURCE不变。固定两ZIP与四个empty Validation身份，禁止任意模型/Train/Test或shell。复用旧CheckpointManager加载器env=None；其API只支持名为WARM_START的模式，这里仅执行经验证反序列化，不是训练初始化。逐step保存真实状态/动作/时间/Reward/PI；分析器从qpos、qvel独立重算距离、速度、动作尺度、物理tick、LiDAR因果关系及5连续样本，原始文件local-only。

## 【5】算法及判断原理

原成功条件d<.1m、v<.15m/s且连续5个25Hz策略样本不变。先验短距离两例都成功且32k成功更少，才支持本子集遗忘；先验未全部成功则NOT_TESTED_PRIOR_NOT_ESTABLISHED。本阶段无梯度、没有新的学习提升或泛化估计，未用完整验证集或Test挑模型。

## 【6】冻结协议

REG `102956dd023d973e64f53193b0d3bd9094f352510659e8d72b93328f8f582783`；150SOURCE `c5cf2d1300709154261a8bb68c1d8bc4b30541f2413c4d60708507fb6f5c346d`；独立L1协议 `2ea8327ef1ee0b52a70c1c6e5c4c8fe97e054077b2ac7b688d9cbd938d20aaad`。两个step固定0/32768，原模型γ.99/λ.95/CNN/Reward/PI/500/100/25/10Hz/24000rays不变。SIM_GROUND_TRUTH_STATE，freejoint世界ENU/姿态wxyz；不是FAST-LIO2。余额1016960步/32400预留秒不变、无新增训练事件，旧门禁仍拒绝R24FAIL。停止使用独立`python3 -m research_regression.authorized_rl request-stop`，不会清除锁存；通用researchctl-v2停止仅管维护任务，不能声称已统一接管本诊断。

## 【7】真实结果

| checkpoint | 距离阶段 | Validation身份前缀 | 终止 | 策略步 | 最终误差m | 最小误差m | 最大连续joint |
|---|---|---|---|---|---|---|---|
|0|SHORT|13daf599df67|success|59|0.083887|0.083887|5|
|0|SHORT|0a0b6aa72e84|success|59|0.084049|0.084049|5|
|0|ORIGINAL|fe85ac443a0d|time_limit|375|0.10501|0.026125|0|
|0|ORIGINAL|0936e963da20|time_limit|375|0.098859|0.023759|0|
|32768|SHORT|13daf599df67|time_limit|375|0.123745|0.063234|0|
|32768|SHORT|0a0b6aa72e84|success|375|0.087369|0.046307|5|
|32768|ORIGINAL|fe85ac443a0d|time_limit|375|0.260376|0.115437|0|
|32768|ORIGINAL|0936e963da20|time_limit|375|0.205325|0.081529|0|

新诊断8episode、2368策略步、47360实际500Hz tick。物理终止可能在最后策略周期中途发生，不将步数×20冒充实测。短距离遗忘判定：`SUPPORTED_IN_THIS_SUBSET`。计数{'0_ORIGINAL': {'collision': 0, 'success': 0, 'time_limit': 2}, '0_SHORT': {'collision': 0, 'success': 2, 'time_limit': 0}, '32768_ORIGINAL': {'collision': 0, 'success': 0, 'time_limit': 2}, '32768_SHORT': {'collision': 0, 'success': 1, 'time_limit': 1}}。Test0/训练0/optimizer0，原R24仍NOT TESTED。

完整1191选择回归PASS=231OPS+75无更新Tensor+885历史，15排除，非全仓；8项新增注册测试，另4项分析器人工算术契约RED→GREEN不冒充真实episode。早期6契约缺接口RED、计数/布尔负例另RED→GREEN、只读loader RED→GREEN已保留。模型测试人工输入forward不是科学episode。独立预审发现3个Important：审查元数据缺SOURCE、PNG额外chunk、launcher残留进程/kill竞态；4项真实发布/受控子进程契约先3FAIL/1ERROR，再修为4GREEN02，未运行模型或实际诊断。首GREEN01名字仅是候选收据，实际仍有1ERROR，保留而不当通过。

内层wall48.038s、CPU48.163s，launcher wall55.601s；worker HWM1453.457MiB，含reaped后代HWM1453.457MiB；guard采样最小MemAvailable9059.918MiB，不是导入全程连续峰值。363授权文件、15额外历史证据字节不变，正常进程组清理，无观测OOM。

## 【8】问题分析

CONFIRMED仅为表中实际新trace及对子集判据的结果。启动前另发现分析器误认running为字符串，实际共享classify_terminal返回None；按真实契约修合成fixture后1ERROR RED，修分析器后4GREEN，再进行02精确准入。01GO及已正常注册/doctor/audit保留，但01prelaunch不启动；新的02review/prelaunch重新绑定helper，不重复register或诊断。SUSPECTED的算法原因不能由两个任务推出；R24后半程最终效果、完整Validation和重启发起者仍UNVERIFIED。无成功不能证明永远学不会，局部成功也不能证明避障。不得拿Train非平稳成功与本次固定模型数据作同策略公平比较。

## 【9】局限性

只覆盖empty/两个朝向与两种距离，没有障碍泛化、完整6场景、最终模型、多seed、真实传感器或PX4验证。只读载入中间权重不具备EXACT/BOUNDARY_RESUME；旧协议与失败状态不改。已知可选cv2/NumPyABI警告非本链路调用，无全局升级。

## 【10】下一安全阶段

实际独立复审/Commit/正常OPSsync远端核验后，检查余额、stop与资源，再从新证据选择单一可反驳假设；若需要训练，必须另有新协议、失败闭环准入和独立GO，不重试原R24或ZIP续训。普通阶段不是长期任务结束。

## 【11】给项目负责人的总结

这次没有重训，而是把重启前保存的两个可信模型放在完全相同的四个简单任务上，实际观察如何运动和停稳。结果见上表；当前结论是`SUPPORTED_IN_THIS_SUBSET`，只适用于这两个短任务。工程链路与数据完整性通过，不代表已经学会陌生矿井避障；预算未消耗，后续依证据再决定一个新实验。

最终独立结果复审GO，C0/I0/新增Minor0；只认证固定8个新只读诊断及原始数值，不补R24完整结论或授权训练/ZIP续训。JSON仅追加review/recovery/doctor与audit引用，统计和图不变。18文件显式白名单Commit/正常OPSsync远端状态以实际收据为准，不能由本段预先认定已推送。
