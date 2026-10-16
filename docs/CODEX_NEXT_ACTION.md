## AUTORL-R13 最新：Actor归零锚定接口PASS，尚无学习增益证据（2026-10-16）

R12已推79befa26f46f3b8bedd55320eed3bb3869dc5518、OPS远端核验phase_complete=true，不重复。R13唯一AUTORL_R13_CENTERED_ACTOR_PREFLIGHT_ONLY exit0/进程组清理，result SHA1107e3fe8388b6b36910a469a1a35324c51efac2a6fabeaf2dd41eb923876c5d。mean=f(当前特征)−f(同背景zero-state)，不改CNN/Critic/Reward/物理/PI；170857参数、Actor MLP两遍/CNN一遍。新policy20f5c226…，规范参数值SHA6312e99c…只用于比较，不冒充同一函数身份。

62固定反事实输入初态mean/value差均0，旧类新建参考+新类+新初态重载共186实际推理；只复用R4历史真实扫描背景，无新扫描/物理/episode/Val/Test/训练/optimizer。零state输出零是结构恒等式，非学会停稳。完整144OPS+50Tensor+885历史=1079选择PASS/15排除，新增4纯+12神经，非全仓。预审C0/I0；实际无参数结构损坏Important RED→GREEN，历史62/logprob补齐；最终结果审查及发布以真实收据为准。

inner wall6.595s/CPU5.774s，launcher13.660s，whole-worker HWM487.266MiB；模型前Mem8302.387MiB仅单次采样。派生绘图Path/bytes API错误已归档后处理修正，未重跑实验/改运行源码。SOURCEf968de2c…/REG1d32001d…/execution21af8a4e…保持，221授权文件/18events字节不变。余1410176step/54000s、stop=false，无退款/新预留；旧OPS零维护预算不撤销独立grant。新初态ZIP仅local诊断，不作正式训练初始化；历史R11 final也不续训，RESTART_REQUIRED。

下一安全动作：完成R13实际结果独立复审/历史audit/Commit/OPS sync远端，再R14独立65536步/3600s新初态中心化Actor对照R11低LR，其他课程/std/Reward/PPO不变；新类36个新Val、Test0，先新profile/冻结/完整测试/独立GO/预算stop资源核验才预留启动。可能表达受限，不能以结构零输出替代导航评价。普通阶段不结束长期研究；PX4/FASTLIO/实机/封存Test禁止。报告PHASE_AUTORL_R13_CENTERED_ACTOR.md、autonomous_rl_centered_preflight_r13.json。

下方为历史快照，不是重复执行指令。

## AUTORL-R12 最新：固定新模型反馈诊断完成，四panel先验保留FAIL（2026-10-16）

R11已完整推bd9095ff2f1b16a1bcd89a8ae699f4df47d1a05a，不重复。唯一AUTORL_R12_LOW_KL_FEEDBACK_ONLY结果SHAd4578edbd556c10e34189376c84b5554b46799318536e15d8e02c583906b771b、exit0/进程组清理；只读4个R10/R11新32k/final模型×62反事实输入=248新响应，复用R4真实背景/旧62初态参考，非310新推理/非新点云。训练/optimizer/物理/新扫描/episode/Val/Test全0。SOURCEc40fa232…、REGb54713cf…、execution2b19c122…保持。

相同固定输入下，R10@32k/final零state速度指令范数.381546/.739462m/s，R11低LR为.294207/1.190841；四panel先验保留FAIL。R11 final零yaw误差原均值2.10765裁剪到+1，五个yaw输入都+1、响应差0。只是离线指令，不是物理漂移/电机饱和/PX4故障，也不由偏离Scripted本身认定Bug。R11小KL仍伴随累计反馈偏离，未证明唯一根因或任何学习提升。

初始_01判据错误（误要求原先验不存在的负速度反馈）在执行前由基准测试发现，旧GO/源码/协议/失败收据原样归档，独立确认Important修复；最终_02完整140OPS/38Tensor/885历史=1063PASS，15排除、0测试optimizer，另3分析方法契约GREEN，不称全仓。全221授权文件/18events字节不变，余1410176step/54000s、stop=false，无新预留/退款。innerwall6.910s/CPU6.092s，launcher14.028s，wholeworker HWM486.340MiB；四次模型前采样minMem8279.930MiB，非全程最低。3raw/四图local身份完整，报告PHASE_AUTORL_R12_LOW_KL_FEEDBACK.md与autonomous_rl_low_kl_feedback_r12.json，最终独立结果复审/Git远端以收据为准。

下一安全动作：完成R12实际独立结果复审/历史audit/Commit/OPS sync远端；再R13零optimizer的Actor状态归零锚定表达预检，研究mean=f(obs)−f(同当前背景,zero-state)以抑制常量偏置。先新设计、测试、序列化/梯度及独立GO，不自动训练；若可行再独立协议对照R11，在原预算内执行。锚定零输出是结构约束，不是学习成绩。不能重放R12或从R11旧模型续训；普通阶段不终止长期研究。PX4/FASTLIO/实机/封存Test禁止。

以下为历史，不得重跑。

## AUTORL-R12 权威：唯一_02零训练诊断已请求启动，只能监控

140OPS/38Tensor/885历史=1063选择测试已完整PASS（15排除、0optimizer），_02独立C0/I0 GO已精确绑定REGb54713cf…/SOURCEc40fa232…/execution2b19c122…/checks_02 SHA046b8940…，新OPS register/recover/doctor及221授权文件/资源/停止/预算核验通过。唯一launch_autorl_r12已调用，当前无第二次执行授权；只审计实际attempt/result/complete和exit_01，不依据本段认证完成。工具会话不是重启后的进程身份依据。

原_01判据错误已在实际运行前RED→GREEN和独立确认修正，旧GO/源码/协议/收据只读归档。R12只能4新模型×62固定输入，无训练/optimizer/physics/新扫描/Val/Test。账本余额1410176step/54000s、18events须字节不变。已有OUT永久不得重放；若未知状态先recover真实锁/process与证据，不能启动替代run。结果/图/MD/JSON/独立复审/Commit/OPS sync远端后继续下一安全研究，源冻结。PX4/FASTLIO/实机/Test禁止。

下方为已结束准入过程，不是启动指令。

## AUTORL-R12 当前：_02固定反馈诊断准入，未执行

R11已完整推bd9095ff2f1b16a1bcd89a8ae699f4df47d1a05a，phase_complete=true/远端已核；不重试。当前research/lidar-low-kl-feedback-r12，设计AUTORL_R12_FEEDBACK_DESIGN.md，新4个R10/R11@32768/65536模型×62反事实输入，原R4背景/初态panel只复用，0optimizer/训练/物理/新扫描/Val/Test。_01“速度差严格负”错误在实际执行前由基准校准测试发现（原先验恰无速度输入），旧GO已废止用于新版本，source/protocol/收据留archive，不能拿旧139项准入执行新代码。

最终候选_02 .research_local/autorl_r12_registration_02.json REGb54713cf…/SOURCEc40fa232…/execution2b19c122…，新9纯/2数组来源及3分析契约GREEN，完整140OPS/38Tensor/885历史正在执行，15明确排除，不提前完整PASS。等待全部_02checks和精确独立GO→新register/recover/doctor→221grant文件、stop、预算、资源→独立进程组唯一feedback_followup_run。当前OUT.research_local/autorl_r12_feedback/run_01不存在；已有attempt绝不能重放。余额1410176step/54000s、stop=false，不initialize/退款或保守预留。source运行期冻结，完成实际结果审查/Commit/OPS sync远端后继续下一安全研究；PX4/FASTLIO/设备/Test禁止。

下方是已结束历史，不得据此重复启动。

## AUTORL-R11 最新：工程COMPLETE，低KL但导航FAIL（2026-10-16）

R10已推21831fcc90d06756f431373643e75822b633eb7a，未重复。新R11仅将全局PPO LR3e-4→3e-5，唯一ARL_R11_PRIOR_LOWER_LR_SEED0已主管/账本COMPLETE：65536实际step/512rollout/1024epoch/2048Adam，1306104物理tick；summary SHA7f34a1e98eae2ef289becacb4d1086f7e5dbb0d686d2cae750610452d90b8f26、final ZIP9ec87ba5…，SOURCEab7544a0…未变。574结束Train=3success/481collision/90timeout；short106、original468，自然切换32811、尾0，575reset仅冻结Train。3成功均short，不证明学会导航。

短Val初态手写先验6/0/0→32k0/0/6→final0/6/0，原Val初态历史R3 PRIOR1/3/2→32k0/4/2→final0/6/0；30新Val/6旧评价复用/Test0。四导航/保留标准FAIL；首32更新KL从R10.071368降至.002472、机制PASS不等于任务PASS。全512 KL均.005162，Critic解释方差仍低；不唯一归因或盲重跑百万步。

1052选择回归PASS（131OPS/36Tensor/885历史，15排除/2既有warning）及5合成分析契约，不是全仓。23raw/5图身份完整、3checkpoint真实LR/optimizer/预测重载通过；194旧不可变授权文件及R9 17raw不变。主管wall828.729s、whole-worker HWM1790.242MiB含reload/Val，训练记录minMem7296.363MiB，全程含Val最低NOT_AVAILABLE。累计589824step/32400s保守预留，余1410176step/54000s、stop=false，不initialize/退款；旧L9永久FAIL与旧OPS零维护预算不改。完整报告PHASE_AUTORL_R11_LOWER_LR.md、autonomous_rl_lower_lr_r11.json，最终审查/远端以真实OPS收据为准。

下一安全动作：完成R11真实结果独立复审/历史audit/Commit/OPS sync远端，之后R12零optimizer新冻结模型反馈诊断，复用R4背景与62固定反事实输入，只分析R10/R11新32k/final权重，非重做旧R4模型，不发控制/不采新训练数据。先新身份/源码准入/独立GO再执行；根据方向/速度反馈证据决定下一单变量实验。普通科学FAIL不结束长期研究，真实停止/安全/预算/会话限制才交接。PX4/FASTLIO/实机/封存Test仍禁止。

以下为历史，不得据此重复启动实验。

## AUTORL-R11 权威：唯一启动请求已发出，只能监控

完整1052项选择准入PASS（131OPS/36Tensor/885历史，15排除）、独立C0/I0/继承Minor1 GO，新OPS register/recover/doctor/195旧授权文件/资源预算stop检查及prepare精确25c6571ecf76757b052a73ca41d620c5d42e39c42d473c36223ae65ee5b80e39均通过，已唯一调用start-lower-lr。事件17真实RESERVE、child进程身份及flock均已核对，账本实际余1410176step/54000s、stop=false，不initialize/退款。当前最后观察完整更新12032步，不是精确停止数或完成数；初期32更新KL.002472低于R10.071368，不代表导航成功。初始管理辅助脚本误读recover返回形状KeyError，实际register/recover已成功，仅继续doctor/历史核验，不重登记/不重复attempt。

只读监控.research_local/authorized_rl_v1/runs/ARL_R11_PRIOR_LOWER_LR_SEED0及主管/child/flock。存在attempt永久不得再次prepare/start、不得从checkpoint继续；当前会话工具句柄不作重启后进程证据。运行期SOURCEab7544a0…冻结，不新增研究生产/测试py。若失败/未知先离线审计，保留RESTART_REQUIRED。主管及账本COMPLETE后真实分析、五判据/图/MD/JSON/独立结果审查/Commit/OPS sync远端，再自主下一个安全任务。PX4/FASTLIO/实机/Test禁止。

下方为已结束准入快照，不是启动指令。

## AUTORL-R11 当前：低全局LR对照准入中，尚未训练

R10已完整推21831fcc90d06756f431373643e75822b633eb7a，OPS sync远端核验phase_complete=true；四科学FAIL不重试。当前research/lidar-prior-lower-lr-r11，新schema11/ARL_R11_PRIOR_LOWER_LR_SEED0，65536step/3600s，唯一变量全局PPO LR3e-4→3e-5；Actor/Critic同时改变，原先验/std/课程/Reward/CNN/物理不变。新6纯/1真实hook RED→GREEN，131OPS/36Tensor PASS，885历史回归仍在执行，不能提前完整PASS。REG6e6907a3…/SOURCEab7544a0…/protocol25c6571e…，细则AUTORL_R11_LOWER_LR_DESIGN.md。

先等待完整当前准入收据及精确独立GO，再新OPS register/recover/doctor→全部旧授权文件/原历史/预算stop资源→prepare-lower-lr核对精确SHA→唯一start-lower-lr。尚未prepare/预留/start；余1475712step/57600s、stop=false，不initialize/退款或改旧预算。run一旦存在永久不能重试，RESTART_REQUIRED，不加载R10旧权重。30新Val+6同policy历史原step0评价复用/Test0，五独立判据包括机制不等于导航。完成结果审查/Commit/OPS sync远端后继续下一安全研究；PX4/FASTLIO/实机/封存Test禁止。

下方为历史快照，不得重复执行。

## AUTORL-R10 最新：执行COMPLETE，四项科学标准FAIL（2026-10-16）

唯一ARL_R10_PRIOR_DISTANCE_SEED0已由主管及账本COMPLETE，实际65536步/512rollout/1024PPOepoch/2048Adam，summary SHAc9fa8bcc9b14fe441903e1f733e7496393e39a36a72c0d2f8e907fc9778ba2ca、final ZIP90726f17…，SOURCE5170a41…未变。425结束Train=1success/318collision/106timeout，short165、original260，实际reset切换33100，尾137步未结束仍计预算。唯一训练成功不证明学会导航；short Val从初态手写先验6成功降为final0成功0碰撞6超时，original final0/1/5，30新Val/6历史R3 PRIOR step0复用/Test0。四科学标准均FAIL，不重跑本ID/不warmstart。

当前research/lidar-prior-distance-r10，基线已推R9 0bdcfa999ef75a0b4f3d34a6498d76267348b7d3。1045项选择准入PASS（125OPS/35Tensor/885历史，15排除），另5分析纯契约RED→GREEN，不是全仓测试。23raw/5图local原始身份完整；主管wall772.024s，workerHWM1778.152MiB含reload/Val，训练记录minMem7093.902MiB；全程含Val最低NOT_AVAILABLE。168旧不可变授权文件与R9 17raw不变，事件15/16真实追加。累计524288step/28800s保守预留，余1,475,712step/57,600s，stop=false，不initialize/退款。结果与远端以PHASE_AUTORL_R10_PRIOR_DISTANCE.md、autonomous_rl_prior_distance_r10.json和真实审查/OPS收据为准，未全交付不称阶段完成。

下一安全动作：先完整结果独立复审/历史audit/Commit/OPS sync/远端SHA；再R11离线准入研究仅降全局PPO LR3e-4→3e-5（Actor/Critic共用），相对R10保持其他条件、原目标科学标准不降低。先验证真实schedule/optimizer组/step0 ZIP及重载新LR，明确旧L6协议仅base provenance不篡改旧hash；新profile/ID/协议/完整测试/独立GO/stop资源预算核验前不训练。R10先验在32k前已丢失、KL/clip大支持更新幅度假设但不证明唯一原因。PX4/FASTLIO/实机/Test调参禁止。普通科学FAIL不终止长期任务，真正阻塞/停止/预算或会话限制才停止。

以下为已结束历史过程，不能据此重复启动。

## AUTORL-R10 权威：唯一训练已预留并启动，只能监控

ARL_R10_PRIOR_DISTANCE_SEED0已完整通过_02准入1045项选择PASS/独立GO/新OPS登记/recover/doctor/资源/169旧文件检查、prepare精确SHA5534e4ae5a2389e98d34b6d1aa6405f3159f1d53d85fa57084000c94365f7abf后唯一start。真实PPO更新记录已观察到，但不是最终完成数。预算第15预留后累计524288步/28800s，剩1475712步/57600s，stop=false；不initialize/退款。SOURCE5170a41…运行期不得修改、新增research_regression生产/测试源也会改变聚合身份并触发停止。

只读监控`.research_local/authorized_rl_v1/runs/ARL_R10_PRIOR_DISTANCE_SEED0`及真实父子process/flock/主管账本。已有attempt永久不能再prepare/start，同ID不重试，不加载中间checkpoint续训。当前PTY36649只是本会话句柄，重启后不能据此推断活跃。完成worker complete还不够，必须主管和账本COMPLETE，30新Validation+6历史R3 PRIOR原step0复用/Test0，实际raw核验、四独立科学标准/图/MD/JSON/结果复审/Commit/OPS sync远端，再继续新安全假设。不修改Reward/CNN/PPO/课程运行中配置，PX4/FASTLIO/实机/Test调参禁止。若未知/失败先保留证据离线审计，RESTART_REQUIRED，不伪造完成。

下面是已结束准入过程，不能作为重复启动命令。

## AUTORL-R10 当前：先验×距离课程已实现，唯一训练尚未开始

R9已推0bdcfa999ef75a0b4f3d34a6498d76267348b7d3、远端核验phase_complete=true；12/12短任务非学习先验通过，不重放。当前research/lidar-prior-distance-r10，设计AUTORL_R10_PRIOR_DISTANCE_DESIGN.md。_02 REG69fcf055e72c0fee6986be3d6e4f7966e7308c9e958d9f6742bf2e5076bed302 /SOURCE5170a41bb9cade75afa81f215755e5dfbfe15b7f1c9a39e736c07e5f8cca55b1 /protocol5534e4ae5a2389e98d34b6d1aa6405f3159f1d53d85fa57084000c94365f7abf；新metadata RED→GREEN已在实际训练前修，旧_01准入只保留历史。新125OPS/35Tensor/885历史正在完整执行，独立复审C0/I0/继承Minor1条件GO。尚未prepare/预留/启动，当前余额1541248步/61200s、stop=false，不initialize或退款。

下一安全动作：等_02完整收据及runtime_binding生成checks→新OPS register/recover/doctor→核169旧授权文件/资源/预算stop→prepare-prior-distance并比对上述精确SHA→唯一start-prior-distance。新ARL_R10_PRIOR_DISTANCE_SEED0仅65536步/3600s/CPU1，从新随机网络赋解析均值6312…，不加载任何旧权重。相对R8只变初始化，原课程/Reward/CNN/PPO/物理不变；首次预留后余额应1475712/57600。运行期源冻结、同ID不重试、RESTART_REQUIRED。30新Validation/6历史R3 PRIOR原step0复用/Test0，四科学标准分别报告，先验保留不等于学习。完成结果审查/报告/图/Commit/OPS sync远端后继续下一安全研究，不普通阶段即结束长期任务。PX4/FASTLIO/实机禁止，封存Test不调参。

以下为历史，不能据此重跑已完成实验。

## AUTORL-R9 最新：非学习短目标解析Actor可完成性PASS（2026-10-16）

R8已完整推送2c731100f0c8888ffc7b0bf30f0a16c6d923895a，工程完成但三学习标准FAIL，永久不重跑。R9唯一AUTORL_R9_SHORT_PRIOR_FEASIBILITY_ONLY实际12/12短Train成功，701物理策略步/14020tick/28.04s合计仿真，0optimizer/训练/Validation/Test；result SHA3abcb783342e3f4429904a1a2188a0741e297df3eece98315af17589f61915d1。不是训练成果/避障能力，最终仍约8cm误差/.10m/s速度，只是原容差联合条件通过。

当前research/lidar-short-prior-r9，REGa958ac…/SOURCE35a4ef…/protocol2bf3f2…；1037项选择回归PASS（118OPS/34Tensor/885历史，15排除/2既有warning）。预审Important物理失败未停止已先RED→GREEN修复；两个Minor和测试夹具错误诚实记录。三图与逐步原始trace封存，worker wall15.256s/HWM1669.391MiB/runner前置采样minMem8704.117MiB、Health累计记录min8608.676MiB；预算169旧文件全字节不变，余1,541,248step/61,200s，stop=false。报告[PHASE_AUTORL_R9_SHORT_PRIOR.md](PHASE_AUTORL_R9_SHORT_PRIOR.md)及autonomous_rl_short_prior_r9.json，最终审查/Commit/远端以真实OPS收据为准。

下一安全工作：完成R9独立结果复审/历史audit/Commit/OPS sync远端核验，然后R10仅改变Actor初始化的独立距离课程PPO设计与准入。保持R8全部算法/Reward/预算/课程，补上相对R3/R8的先验×短课程交互项；明确区分先验保留与真实学习增益，不从旧checkpoint训练、不重跑已用ID、不因普通阶段结束停整个研究。PX4/FASTLIO/实机/Test调参仍禁止，不初始化/扩大原grant。

以下为历史过程快照，不得作为重复启动指令。

## AUTORL-R9 权威：唯一非学习run已启动，不得重复

_03完整1037项选择回归PASS（118OPS+34Tensor+885历史，15排除/2既有warning）、独立预审GO（2已披露Minor）、新登记/recover/doctor/资源与169旧授权文件已核验。REGa958ac5fc2fe3bcbc5c64b5267bf4a598f61697ec8d7e9a82dfa17ed48510532 /SOURCE35a4efcf224a1a0e9b7c4c878e1948d139b174007c5e93798585ac4e3742772b /protocol2bf3f221a6a4a9f9ec3e481b8aac8e2ce4968df97f0a9616eae24ceac5439766。唯一`.research_local/autorl_r9_short_prior/run_01`已启动；只读monitor，已有attempt永不再次调用runner。0optimizer/训练/Val/Test，旧R6/R8均不重试；余额1541248/61200不变。

完成complete/result/exit后核验逐步物理trace、原停稳标准、预算全目录和源身份、图/MD/JSON、独立结果复审及显式Commit/OPS sync远端，然后继续下一安全阶段。断电/未知状态先recover实际锁和process identity、标INCOMPLETE，不能重放。physical/sensor/未知终止立即停止，普通collision/time_limit仅可作为科学失败继续固定覆盖；不改Reward或判据。json_safe512节点可能截断异常context及R7结构损坏Minor均披露，不承诺完整异常快照。

下面为已结束准入过程，不是启动命令。

## AUTORL-R9 当前：非学习解析Actor短目标预检，尚未运行唯一实验

R8已完整Commit/Push远端核验2c731100f0c8888ffc7b0bf30f0a16c6d923895a，工程PASS/三科学FAIL；永久不重跑。当前research/lidar-short-prior-r9，新设计AUTORL_R9_SHORT_PRIOR_DESIGN.md。新6纯契约/1真实step RED→GREEN；最终注册候选_02 REG e9af4f8705640ef36290764e4563b681248fcc0561c4a4f3d7d4659fc923860e /SOURCE077ae00d0939476d1beda2041fe5dd10c80695bab2ea61032c177e4c93269977，116OPS/34Tensor/885历史及独立预审正在执行。尚未登记或启动唯一run，不从旧权重初始化。

下一安全动作：等待全部完整回归及精确GO后，新OPS register/recover/doctor→验证grant/stop/169既有文件/资源→独立进程组唯一short_prior_runner。固定12short Train、原停稳条件、0optimizer/训练/Val/Test，180s/单worker。已有attempt不可重试；若会话中断先核锁/实际进程及原证据，不能重放。余额1541248step/61200s不initialize/不退款，旧R6永久FAIL不重试。结果复审/图/MD/JSON/Commit/OPS sync远端后继续新独立研究；先验成功不是学习收益，不能直接开下一训练。PX4/FASTLIO/实机/Test调参禁止。

以下为已结束历史，不能作为重复执行命令。

## AUTORL-R8 最新：执行COMPLETE，距离课程学习标准FAIL

唯一ARL_R8_REPAIRED_DISTANCE_SEED0已由主管及账本确认COMPLETE，永久禁止再次prepare/start或从checkpoint继续。actual65536step/512rollout/1024PPOepoch/2048Adam；summary SHA a6760dc61ff03a3895eb451f99d61d77513c643a3e8d4f8e31a55f556475dce4，final ZIP54c95adb…，SOURCE62b330…前后未变。当前branch research/lidar-repaired-distance-r8，基线为已推R7 05c7d9b；最终Commit/远端以OPS真实收据为准。报告PHASE_AUTORL_R8_REPAIRED_DISTANCE.md和autonomous_rl_repaired_distance_r8.json，独立结果审查/发布流程未全部完成前不宣称阶段已交付。

Train280=180碰撞100超时0成功；短任务128、原任务152，实际reset切换33070，末尾122步未完成episode仍计预算。30新Validation/Test0；短final0成功1碰撞5超时，原final0成功2碰撞4超时（原step0六条明确历史复用）。三科学标准FAIL，碰撞下降伴随超时增加，不等于学会导航。短21episode进过目标球却无joint停稳，不能断言从未靠近目标或单独归因速度/保持时长。1028项选择准入PASS/15排除，另3纯分析契约RED→GREEN；不是全仓PASS或实机能力。

主管wall720.338s，整worker HWM1696.816MiB含重载/Val，训练采样minMem9614.242MiB；全程最低Mem含Val NOT_AVAILABLE。无观察OOM/源变化/重试。原历史run/checkpoint/grant完整，R6仍永久FAIL。现保守累计458752step/25200s，余额1,541,248step/61,200s，stop=false，不initialize/退款。旧OPS零维护预算不撤销独立grant。

下一安全操作：先完成R8完整raw/图/MD/JSON独立结果审查、历史audit、显式Commit/OPS sync/远端SHA核验；未同步不跨阶段训练。随后新独立R9 Level1（0optimizer）短目标解析Actor反馈可完成性：新随机网络中按已审计公式赋解析均值，在相同12short Train用确定性动作检验原停稳条件；不是重复R5 Scripted、不是加载R8或L9旧权重、不是学习能力。若可行，再冻结仅改变初始化、相对R8随机均值的新PPO协议并完整准入，在原余额执行，不直接开训。课程/Reward/CNN/PPO/physics不因本次FAIL临时调参，Test封存/PX4/FASTLIO/实机禁止。普通阶段结果FAIL不结束长期任务，无停止/资源/会话等真实阻塞则继续安全研究。

以下是已结束过程快照，不能据此重复启动。

## AUTORL-R8 权威：唯一attempt已启动，只能监控，禁止重试

ARL_R8_REPAIRED_DISTANCE_SEED0已通过1028项选择回归/独立预审GO、新OPS登记/recover/doctor及prepare实际SHA核对，执行协议2eeb62e185b5c1bfe9de5aa6a299dcc5e546248aaa7b88661667ae719a58efbe。原supervisor已唯一预留65536步/3600s并启动；目前已观察到1280步更新行/40Adam，但这不是最终完成数，真实进度与结束只读run目录及complete/账本。运行期SOURCE62b330…冻结。

余额1,541,248步/61,200s保守wall、stop=false，不initialize/不退款；旧R6永久FAIL，R7已推05c7d9b。只监控.research_local/authorized_rl_v1/runs/ARL_R8_REPAIRED_DISTANCE_SEED0，存在attempt时绝不再次prepare/start，不加载中间checkpoint继续；若断电/失联先核实际flock/parent-child identity/完整证据，按RESTART_REQUIRED处理，不推断成功。完成原30新Validation/Test0后独立结果审查/报告/图/Commit/OPS sync远端，再自主继续下一安全研究。当前supervisor会话57064仅是此会话句柄，重启后不能据此假定仍活跃。

下面“尚未开始”为已结束过程快照，不是当前指令。

## AUTORL-R8 当前：已冻结修复后独立课程协议，训练尚未开始

R7已完整推送05c7d9bd63b4e48b461a876407d7f4917e746fe6并远端核验phase_complete=true；R6永久FAIL不重试。当前新分支research/lidar-repaired-distance-r8。新schema8/ARL_R8_REPAIRED_DISTANCE_SEED0、seed0/65536step/3600s/CPU1；原Reward/CNN/PPO/动力学/PI/24k模式不变，仅短.20m目标至32768后自然reset恢复原12目标。R7 adapter已接入真正worker入口，不从任何旧checkpoint初始化。protocol2eeb62e185b5c1bfe9de5aa6a299dcc5e546248aaa7b88661667ae719a58efbe；REGa5c17d…/SOURCE62b330…，预注册见autonomous_rl_repaired_distance_r8_prereg.json。

新5profile/1实际入口测试RED→GREEN，当前110OPS/33Tensor/885历史=1028项选择PASS/15排除/2既有warning，0optimizer。checks88736fa1…已绑定真实历史运行前后SOURCE，独立预审等待最终收据；尚未prepare/预留/start。所有旧run及grant目录143文件SHA不变，余额1,606,784step/64,800s，stop=false，不initialize/退款。旧OPS维护预算0不是独立grant撤销。

下一安全动作：独立预审精确GO后新register/recover/doctor→验证资源/停止/余额→prepare-repaired-distance显式核对上述协议SHA→唯一start-repaired-distance。启动后只监控同一ID/不重试，预留后应余1,541,248step/61,200s，不能把预留当实际完成。若未知或失败先保留现场，不删attempt、不warmstart。运行期SOURCE冻结；checkpoint0/32768/65536，关闭Train后30新Validation，originalstep0六条明确复用历史不计新执行，Test0。完成实际结果/图/MD/JSON/独立复审/Commit/OPS sync远端后继续下一安全阶段；不自动PX4/FASTLIO/实机/Test调参。

下方是历史快照，不是重复执行命令。

## AUTORL-R7 最新：离线记账修复，完整回归通过（2026-10-16）

分支fix/lidar-distance-accounting-r7，基线2f6e5c297c2a12f7e69a8328d90aee11eed97589（R6已诚实FAIL发布且远端核验，永久不重试）。新显式adapter保留12base抽样与24Train任务真实身份，完整TrainingMetrics→FormalHealth真实step GREEN，不改R5/原Navigation/Reward/PI/动力学/旧worker。_01独立NO_GO I1故障身份缺失已先RED2失败→GREEN；最终_02 REG486b2730…/SOURCE1eaa6b1f…完整1022选择测试PASS（105OPS+32Tensor+885历史，15排除/2既有warning），0optimizer/训练/Test；不是学习性能提升或全仓PASS。报告PHASE_AUTORL_R7_TASK_ACCOUNTING.md与autonomous_rl_task_accounting_r7.json；最终独立GO及Git交付以本地真实审查/OPS远端收据为准，不用文字自行认证。

当前独立grant累计393216步/21600s保守预留，余1,606,784步/64,800s、stop=false。R7执行前后整个授权目录143文件、所有旧run/R6原10文件、R5报告字节不变。原L9永久FAIL，R6唯一0accountedstep/一次20tick物理返回/0optimizer失败不退款，不可继续step0。旧OPS维护零预算不撤销独立grant，不能initialize或改旧协议。

下一安全动作：确认R7最终独立GO→新版本register/recover/doctor→历史audit→显式文件Commit/OPS sync/远端核验。随后自主进入独立R8修复后距离课程协议：新固定ID/schema/精确pin R7发布证据，复用原worker加显式adapter，只变课程目标分布，不使用任何旧checkpoint初始化；新协议测试/预审完成前不训练。预拟65536步/3600s、CPU单worker、短目标至32768后自然reset恢复原目标、原Reward/CNN/PPO保持。R8不是重试R6；R6协议/attempt永久保留。Test封存/PX4与设备禁止，资源/stop/未知状态阻塞则安全停。

以下为历史快照，不能作为重复执行命令。

## AUTORL-R6 最新：FAIL，必须先离线修复（2026-10-16）

唯一ARL_R6_DISTANCE_SEED0已FAIL，禁止prepare/start重试或从step0继续。第一条实际20tick/.04s transition返回后，新短目标task身份不在旧TrainingMetricsEnv的12个base白名单，被安全拒绝；已完成训练记账步数0，optimizer/episode/新Validation/Test均0，不是PPO学习失败。完整现场/step0 ZIP已封存local-only，运行源码ca83fb16…未改；原预审GO遗漏此Important消费者契约，1011项选择回归PASS不等于完整链通过。

预算保守预留不退款：累计393216步/21600s，余1,606,784步/64,800s，stop=false。R5已推送d4971529…，R6最终Commit/远端以OPS实时收据为准。阶段报告PHASE_AUTORL_R6_DISTANCE_CURRICULUM.md、机器JSONautonomous_rl_distance_curriculum_r6.json。失败报告可同步但OPS保守phase_complete=false，不伪造PASS。

下一安全动作：先完成R6失败结果独立审查及Commit/Push远端核验，再独立R7 Level1修复。显式保留base/task双身份，只在记账边界使用受审Train任务白名单，不删除过滤/伪造base身份，不准入Val/Test；完整TrainingMetricsEnv→FormalHealthMetricsEnv真实step、跨课程阈值、未知身份/失败锁存/Checkpoint状态先RED→GREEN，0optimizer。修复发布及新审查后才可另立独立训练协议和ID，不重试R6、不重置原grant。PX4/FASTLIO/实机禁止，Test封存。当前训练因已知Important暂BLOCKED，离线修复自动继续。

以下为历史过程快照，不是当前执行命令。

## AUTORL-R6-DISTANCE：实现完成，准入中，尚未训练

R5已独立结果复审GO并远端核验d4971529c63c9f2f305c78c851cb8d94b20d6487。当前research/lidar-distance-curriculum-r6，读AUTORL_R6_DISTANCE_TRAINING_DESIGN.md；固定新schema6/ARL_R6_DISTANCE_SEED0，65536步/3600s/seed0/std.3，只改变目标距离课程。新profile6项与实际初始化/独立短Validation2项测试已RED→GREEN，完整105OPS/21Tensor/885历史与独立预审正在执行。尚未prepare/预留/start，不重复旧实验，余额仍1,672,320步/68,400s。

下一安全步骤：全部当前终证据及精确SOURCE/REG独立GO后建立r6_checks、OPS register/recover/doctor，通过固定R5proof/R3control验证，prepare-distance并核SHA，唯一start-distance。训练时冻结代码，failure/unknown保留attempt不重试，RESTART_REQUIRED。首次训练余额应在预留后变1,606,784步/64,800s，不能退款或initialize。完成后实际30新Validation/Test0、checkpoint/optimizer/权重及源身份核验、图/MD/JSON/独立结果复审/Commit/Push/远端，然后继续新安全假设。不把step0旧6条评价算作新执行，不从旧checkpoint初始化。

## AUTORL-R5-SHORT：已完成历史

R5 run_01实际12/12短Train成功，result SHA395fe3891efc2b7917d087a939922d945bd9d3c1aace4e632efa95b366fd0a94，禁止重放/删除attempt。701策略step/14020物理tick、0optimizer/Val/Test，源码2d576a2e…执行前后相同，1003项相关测试PASS/15排除；预算字节不变。完成独立结果复审/Commit/Push后，以OPS远端收据为权威，自动进入新独立R6课程PPO，而非结束长期任务。

R6准确任务：固定新schema6与ID（保留旧schema1/2/3/4拒绝行为及immutable grant）；pin R5已发布proof SHA和R3 RANDOM历史summary7651327c…，新随机seed0/std.3/65536step/3600s/CPU单worker。只变目标距离分布：前32768步12短目标.20m，下一自然reset恢复原12目标，不中途改目标。原CNN/Reward/PPO/PI/24000 rays/频率不变。复用现有worker与计费，不复制算法。Checkpoint0/32768/65536；独立短Validation6在三个checkpoint评价，原Validation仅32k/final实际跑、step0复用同权重R3历史证据，不能把复用当新episode；Test0。Final选择，预注册短成功≥4/6/碰撞≤1/6/无物理传感器失败、original成功>0、优于自身step0分别判断，不事后改标准。先开发RED→GREEN/完整回归/审查/新登记/doctor再预留原账本并唯一执行；当前R5注册不能启动R6训练。R6发布后再审计预算/stop/资源决定下一安全假设，无需普通阶段确认。

余额仍1,672,320步/68,400s，stop=false；不initialize/退款，不从旧模型继续，RESTART_REQUIRED。PX4/FASTLIO/实机禁止，Test封存。若最新状态SYNC_PENDING先完成同步；未知attempt先审计，不能重试。下面是本阶段过程历史，不是新的执行命令。

## AUTORL-R5-SHORT：实现与准入过程

R4远端ac10e5cccb2e84f9b42b39d4924f4c71bfdd19bd已再次核验完整，当前新分支research/lidar-short-goal-r5。读AUTORL_R5_SHORT_GOAL_DESIGN.md。唯一run_01尚未执行；设计最初0.35m被静态几何拒绝，执行前统一改0.20m，18个Train/Validation派生目标重新检查通过，不改障碍/裕量/Reward。8项OPS/3项实际环境测试含RED→GREEN；预审_01 NO_GO I1可完成性证据不足，已先RED7失败→GREEN8项，_02完整重新冻结与回归/复审中，不用旧准入运行。

剩余独立grant1,672,320步／68,400s、5旧attempt均COMPLETE、stop=false，不initialize/不重放。R5当前只执行经审查的唯一12个短Train非学习物理episode，0optimizer/Val/Test；全部原停稳标准成功才可进入独立R6课程训练。R5实际结果/MD/JSON/独立复审/Commit/Push远端完成后，自动继续R6冻结入口/协议与测试审查，在原预算执行65536步课程，不能只停在文档。R6必须pin已发布R5 proof SHA，不改旧profile/grant。任何安全/未知状态/停止/资源不足先安全交接。

## AUTORL-R4-FEEDBACK：唯一运行已完成历史

当前research/lidar-policy-feedback-audit，基线dafc0fa117913ad40a1bbb9057b7c6b7b4767e06；最终commit／远端以OPS收据为准。R4 result83d3766ebdba586282f23e2b3c39a4284cc03791a4017d963af2fa04cef7a673，已完成，不得再次run或删除attempt。11固定checkpoint×62反事实case、单真实Train reset，零physics/optimizer/episode/Val/Test。_01 I1修复后_02注册42dcf1b3…、源码d789aa04…、协议b4379107…、992项PASS／15排除、独立预审GO。先查最终结果审查／同步收据；待推先sync。

下一项已授权安全研发：新分支设计短距离empty到点任务可完成性及冻结课程。理由是R4发现训练后的零state偏置和方向响应不足，不能唯一归因Reward／LR；先保持Reward/CNN/PPO/物理不变，独立改变任务距离分布并核验新manifest/几何可达性。原R1只切换empty/混合且未缩短目标，不能冒充相同研究。先设计／测试／审查／登记，未冻结不启动训练；不重复旧attempt、不加载旧权重续训。仍只CPU单worker、Test封存、PX4/FASTLIO/实机禁止。剩余1,672,320训练步／68,400s保守wall、stop=false；旧OPS零预算只是维护权限，新grant为权威。新实验仍计累计预算，不initialize、不退款。

下列为本阶段过程历史，不是新的执行命令。

## AUTORL-R4-FEEDBACK：2026-10-16过程记录

R3已完成并远端核验dafc0fa117913ad40a1bbb9057b7c6b7b4767e06，两个std.3训练不重放。当前新分支research/lidar-policy-feedback-audit，正在实现固定11checkpoint／62反事实输入的无optimizer诊断，读AUTORL_R4_FEEDBACK_DESIGN.md及PHASE_AUTORL_R4_FEEDBACK.md。7项标准库／2项真实Tensor已RED→GREEN（含Mask拒绝）；尚待最终源码冻结／完整回归／独立预审／新OPS登记，不先执行唯一run_01。

本阶段训练0、物理step0、episode0、Val/Test0；只新建一个固定Train reset真实射线快照，再关闭环境做推理。预算仍余1,672,320步／68,400s；不initialize、不改旧grant／protocol、不重试已消费ID。当前旧OPS源码身份拒绝新文件属预期，按prepare→review→register处理，不绕过；完成后如实报告／Commit／Push远端核验。

## AUTORL-R3-STD：已完成历史交接

**当前权威：两组均完整COMPLETE，不得再次prepare/start。** RANDOM summary7651327c99f27940ce047d4b03b57b40f4ab59aff37632757ae70350445d36d1；PRIOR summaryafbeac0d092aa4f38c318f4b75909006db65f2f8eaa4435783ae6b7a887ada4e，各65536步／2048Adam。Train均0成功，Final Val均0成功6碰撞，三个科学标准FAIL。源码b7290417…前后不变，982个唯一相关测试PASS／15排除。所有5个grant attempt均COMPLETE，预算余额1,672,320步／68,400s，stop=false；不得initialize或退款。完整原始目录只读。

先核对当前OPS最终结果复审与Git远端收据，若待推先sync；本阶段完成后新分支／新登记进入无optimizer的冻结checkpoint目标反馈／减速诊断。拟检查人工声明的固定观察中位置误差、速度、Yaw与动作响应，使用真实射线编码／合法自身状态，不读取障碍真值或Test，不把输入探针当闭环性能。依据该证据再冻结新的课程／优化对照；不重复旧训练，不盲目扩步或修改运行过的协议。下方均为已结束过程快照，不是新的执行命令。

最新已确认：RANDOM完整COMPLETE，summary SHA7651327c99f27940ce047d4b03b57b40f4ab59aff37632757ae70350445d36d1，65536步／2048Adam／548结束episode=513碰撞35超时0成功。Final Validation0成功6碰撞，科学改善FAIL；不再prepare/start RANDOM。第二组PRIOR已按原冻结计划唯一prepare/start，协议6b9e46d53fabc20f2b064c7588f283f96238b8a9b4e06b6cc2f68b51982bd739；只监控同一ID，禁止并行或自动重试。初始policy6312e99c…，原均值1f3e9a0a…、std.3，其余初态相同。累计预留327680步／18000s，余额1,672,320步／68,400s；完整原始目录及预算为权威状态，下方为过程快照。

R2已完成并远端核验669afdcfa0551496bc9e8262f5c2d7353a5828d7，禁止重跑。当前分支research/lidar-initial-std-r3，先读AUTORL_R3_STD_DESIGN.md及PHASE_AUTORL_R3_INITIAL_STD.md。R3只改变初始Gaussian std=.3，随机均值／解析均值两个65536步新臂，各3600s、顺序CPU单worker；其他算法／Reward／物理不变。当前正在实现固定schema3、无更新测试及新登记，尚未训练。既有R2旧OPS身份拒绝修改后源码属预期，必须新版本审查／register，不改旧协议。

预算以新ledger CLI为准；不得initialize。84 OPS／13 Tensor／885相关历史无更新测试、15排除、源码及准入独立GO已完整，登记955dd536…／源码b7290417…／准入498ecfd9…，recover/doctor PASS。唯一RANDOM新attempt已prepare/start，协议20f0382a1e544c9a9c4419763819e9e24a6ca49c2c48033bbfb7267501ddd80d；核对真实flock和事件，不再启动同一ID。完成后审计complete，再prepare/start唯一PRIOR（6b9e46d5…），不能并行；中断／失败不自动重试。Test0，不从历史checkpoint继续。授权范围内不再问普通阶段是否执行；每阶段先Git推送核验再下一阶段。

## AUTORL-R2-PRIOR：已完成历史交接

**唯一attempt已完整COMPLETE，不可再次prepare/start。** 实际65536步、2048 Adam调用、512rollout；Train983=956碰撞+24自然越界+3超时+0成功；尾部55步未结束，仍计预算。最终Validation0/6成功、6碰撞，step0为1成功3碰撞2超时，改善假设FAIL；不是学会避障。源码44bd92f5…运行前后相同，summary SHA03fdb73a394880432eabbe19b7e28ce0325186e8d96d9346aae1dd319ec3d928。原始目录只读。

累计预算196608步／10800s保守预留，剩余1,803,392步／75,600s；新grant stop=false、三个attempt均COMPLETE。旧OPS维护预算0不是新授权丢失。972项选择测试PASS／15排除，无更新测试与计费训练分开。当前分支research/lidar-actor-prior-r2，Commit及远端完成读真实V2收据；若未推，先完成结果独立复审和sync，不提前启动下一训练。

下一已授权安全工作：研究初始动作std。候选R3A原随机Actor/std0.3对照已冻结R1/std1，R3B相同解析先验/std0.3对照已冻结R2/std1，各65536步／3600s、顺序单worker；仅为待登记设计，不是当前CLI可运行任务。保持其他协议不变、std继续可训练；先无optimizer身份与数学检查、独立预注册／测试／审查／新OPS登记，才可在原累计预算内执行。不能改旧schema1/2协议、重跑旧ID、读取Test或把手写先验能力当作学习。具体方案可由证据重新评估，禁止盲目百万步。

### 恢复与运行过程（已结束的历史快照，非执行指令）

**最新运行状态：R2唯一attempt已启动，禁止再次prepare/start。** 新版本登记、recover/doctor、78 OPS＋9 Tensor＋885历史选择测试（15排除）及准入均已完成；训练协议09a4da2c…已匹配。完整证据在`.research_local/authorized_rl_v1/runs/ARL_R2_PRIOR_SEED0/`。预算已保守预留196608步／10800s，余额1,803,392步／75,600s。是否仍活跃必须核对实际flock、child身份及grant事件；资源日志只代表最近观测，不是最终步数。若没有complete、已有预留或进程状态不明，先审计，不重复已开始的attempt；RESTART_REQUIRED，不从STEP0／中间checkpoint继续训练。

工作分支`research/lidar-actor-prior-r2`，HEAD为已推送的预检提交`161997c0cecbe0b0f1e19d861d927ab774bb7ccf`。工作树未提交修改是上一会话R2独立执行入口开发，不要reset或重新实现。候选登记`.research_local/autorl_r2_registration_02.json` SHA`520b6dfd65c796efe67f15e2eb2799908ae1e45b4b7f648902a049d4ee638757`，源码SHA`44bd92f55daabbda4ce21b07a6563b27d2ed0cec9cb395a250cbe65f6179ca1e`，两者已实际匹配。

2026-10-16已核验：无活跃训练／锁，R1两臂COMPLETE并远端核验；R2尚无protocol或run、未预留预算。已用131072步／7200s，剩余1,868,928步／79,200s；账本stop=false，不initialize、不更改旧授权。最新独立源码审查GO须精确绑定上述_02候选，_01旧GO不能代替。

上次关机使最终历史回归INCOMPLETE：仅有`.research_local/autorl_r2_historical_final.stdout.txt`部分输出，没有final.json；原部分日志保持不变。本次在独立restart stdout补做未完成无optimizer测试，不重放已完成研究实验。78项OPS、9项Tensor完整结果已保存且绑定同一源码；待历史885项完整完成后，再生成`.research_local/autorl_r2_checks.json`。旧OPS recover/doctor因未登记新源码返回BLOCKED是预期；只能用V2新登记迁移解决，不能改旧协议或关闭检查。

上述回归／准入／登记／准备／启动链已执行，下一安全步骤是监控同一活跃attempt，完成后分析原始结果、独立结果复审、Commit/Push及远端核验。执行协议SHA`09a4da2c329ddcf6dc6da79dfa4c49c0bac6eab441e53f6c99e2a6040566f61e`，65536步／3600s、seed0／单CPU；step0/32768/65536关闭Train后各6Validation，Test0。若无complete但进程已经消失，按真实中断处理，不再次start。

读取`docs/AUTORL_R2_TRAINING_DESIGN.md`和本阶段MD／预注册。只改变Actor初始化，std／CNN／Reward／PPO不变；最终必须分别比较R1历史控制与自身step0，不能把手写先验能力说成PPO学习。完成后报告／图／独立复审／Commit/Push远端核验，再依据预算自主推进。禁止PX4／FAST-LIO2／实机及Test开发评价。

## AUTORL-R2-PRIOR-PREFLIGHT：已完成历史

分支research/lidar-actor-prior-preflight，基线为已推送R1的9d3ac31。登记443833a087946a174ef3988c8e7f260ae6ea6d3cd1e8ec80bd12693f020698b0；源码400e25a68a2af0ed3dcf351ed7deeca657d9bc7d78f6f9b7f92ee8ec831d699d。读本阶段MD／JSON和真实OPS收据；结果／Commit／Push完成后再推进。

已完成无optimizer的解析Actor测试及三条真实empty Train probe（全部原标准成功），不得重放`.research_local/autorl_r2_prior/run_01`。没有Validation/Test，没有新训练；不是避障策略。新账本仍131072步已预留、余额1,868,928步／79,200s；不要initialize。

下一安全工作：在独立分支为新R2 PPO实验扩展显式固定入口和版本化协议，只改变Actor初始化、std/Reward/CNN/PPO等不变。先声明原随机初始网络与手写先验差别，预注册初始和最终Validation，历史R1 UNIFORM作为冻结对照而非重新执行。新源码／执行协议／累计预算预留／测试及独立GO完成前不得启动R2训练；当前CLI仅支持已完成R1，不能换ID绕过。用户总预算内的普通开发与研究已授权，无需再次问普通阶段是否执行。

停止和恢复仍用新authorized_rl入口；RESTART_REQUIRED，不宣称精确续训。若停止锁存或未知attempt，先审计，不删除标记。若远端待推，先完成sync；严禁PX4、FAST-LIO2、实机及Test开发评价。

## AUTORL-R1：已完成历史

**已结束**：两臂各65536步均完整，最终Validation均0成功6碰撞，课程改善FAIL。不得再次start两个R1 ID。预算真实余额1,868,928步／79,200秒；两次预留完成记录均在独立账本。先完成本阶段结果独立复审、Commit/Push远端核验，再进行新注册研发阶段。

下一候选：解析state-only教师初始化的单变量诊断。仅改变Actor初始化，原结构／Reward／PPO／std不变；先检验两层Tanh能否在预注册容差内逼近已成功Scripted反馈（不是已证明的精确等价），不调用optimizer；再冻结新实验预算和研究评价。现有CLI尚只支持固定R1，必须先开发／测试／新版本登记／独立GO，不能改R1协议或复用其attempt。普通研发及明确预算内PPO已获授权，不用再次索取普通阶段指令。

当前分支`research/lidar-curriculum-r1`，基线e9b90198732bec00c64173c154bd63b7cbc8c537已推送。R1登记SHA de9a5517c7806fce459411acb9204e010124284d1dd763718e99797684cb7714，冻结协议见`autonomous_rl_curriculum_r1_prereg.json`。同注册版本baseline三条empty Train已真实成功（.research_local/authorized_rl_v1/baseline.json），不重复运行或initialize账本。

上述两个ID已按顺序各执行65536步、各预留wall3600s；这不是待执行指令。先读新预算`python3 -m research_regression.authorized_rl status`；锁被持有或存在未知／失败attempt时不能重放。完整训练结果在`.research_local/authorized_rl_v1/runs/<ID>/`，无complete不能宣称完成。训练器已使用过的ID永久不能自动重试；不是旧L9的续训。

真实剩余额度以新append-only账本为准；公开文字和旧OPS的NO都不是恢复授权来源。训练结束后独立结果复审、报告／图／长期文档Commit/Push和远端核验；若结果未达目标，则先分析并冻结下一新协议，不盲目增步。

## AUTONOMOUS-RL-RESEARCH-V1：授权历史与操作说明

用户已于AUDIT-01之后明确授权：累计2,000,000训练environment steps／86400s wall，单次<=1,000,064、CPU单worker；不授权PX4、FAST-LIO2、实机、Test调参。读`CODEX_AUTONOMOUS_RL_V1.md`、本阶段机器JSON及本地新预算，不能从以下历史NO推断新授权丢失，也不能用本段恢复账本。

工作目录不变，当前分支`feat/autonomous-rl-research-v1`。先`git status`及`python3 -m research_regression.authorized_rl status/doctor/recover`；OPS-V2仍负责维护／历史核验和显式文件发布，其NO/零预算只是旧范围。不得改旧L9R denial或V1/V2账本。

新入口和独立预审完成后，下一安全操作为在新阶段登记下运行三个empty Train的state-only Scripted基线，全部原标准成功再开展两臂65536步课程对照。无SMOKE/PILOT/L9权重初始化，B在32768后下个自然reset切换；两个checkpoint延后至Train关闭后分别Validation，Test=0。完整细则在设计文件。预算先预留全额、失败不退不重试；未知attempt必须recover审计。每阶段先Commit/Push/远端SHA核验再推进下一阶段。

停止用`python3 -m research_regression.authorized_rl request-stop`；此版本停止锁存，不自动清除或重跑。最新Commit和实际阶段完成读V2收据；不存在必须续训的旧checkpoint。以下为历史快照，不构成当前执行权限。


## AUDIT-01交接：优先读取本节

位置 `/home/keyousuke/MineUAV-Learning-px4-sitl`，分支 `research/lidar-ppo-learning-diagnosis`，实际最新Commit读 `git rev-parse HEAD` 及V2同步收据。先读总协议、V2协议、[本轮中文诊断](PHASE_AUDIT_01_PPO_DIAGNOSIS.md)与JSON，再运行status→recover→doctor；若SYNC_PENDING优先sync，不改已封存待推文件。

当前授权Level0/1、训练预算0；**不允许任何PPO/监督optimizer更新、Pilot、正式训练、Test episode、PX4或设备通信**。L9永远FAIL，L9R denial不变，L9S只是4096步链路Pilot；中断策略RESTART_REQUIRED，不能加载L9中间checkpoint继续原实验。

最推荐的下一次实验是“仅改变场景调度”的课程对照：新初始化seed0，两臂各65536步，总131072；单worker顺序执行、wall总上限7200s建议，原Reward/CNN/PPO不变。**这只是提案，不是可执行授权**。需项目负责人独立批准预算，随后新协议和安全预检；不自动运行。

仍可执行已登记的 `./researchctl-v2 start --task audit` 做历史核验。新分析原始结果在 `.research_local/audit_01/run_01/`，已使用目录拒绝覆盖；不把原始数据粒度不足用新训练或伪造轨迹补齐。没有必须自动追加的研究阶段，本轮交付后停止训练相关工作。以下为历史交接，发生冲突以本节和真实本地状态为准。

## OPS-02交接：历史记录

项目 `/home/keyousuke/MineUAV-Learning-px4-sitl`，当前分支 `feat/research-operations-v2`。先读 `CODEX_MASTER_SPEC.md`、`RESEARCH_OPERATIONS_V2.md`、本阶段JSON及 `./researchctl-v2 status`，实际HEAD以 `git rev-parse HEAD` 核对；提交自身SHA不写回静态报告以免循环。

下一次第一步：`./researchctl-v2 recover`，然后 `doctor`。若 `SYNC_PENDING`，先运行 `sync`，不改待推文件；任何身份或历史冲突BLOCKED，禁止reset或重新init。V1本地文件是只读历史，不要继续运行会改变它的旧recover/start/sync。

权限Level0/1，训练剩余timesteps/wall-time=0/0；不允许PPO optimizer、Pilot、正式训练、PX4。可运行 `./researchctl-v2 start --task audit` 维护核验。新增阶段用prepare→只读审查→register，不能直接改protocol或预算。

最新已知有效模型为历史L9S `L9S_REPAIRED_PILOT_ONLY_SEED0.zip`（身份由V1配置只读校验）；不是正式Seed0最终模型，不允许自动Warm Start。当前训练恢复仍RESTART_REQUIRED，不具备精确续训。最近代码问题为本阶段审查发现的启动窗口，最终关闭状态看JSON与审查证据。以下V1块仅为保留历史快照。

<!-- RESEARCH_OPS_V1_BEGIN -->
# 次日恢复入口

先读 `docs/CODEX_MASTER_SPEC.md`。

阶段：OPS-01；状态：SAFE_STOPPED。

分支：`feat/research-operations-v1`。快照基于Commit：`37b6d90e84ba0554db3574c9913413ee93c58e41`。

当前Commit以 `git rev-parse HEAD` 为准；快照不能包含它自身尚未创建的Commit SHA。最近成功推送及实时完成状态见 `./researchctl status` 的核验收据。

训练授权：NO；剩余timesteps/wall-time：0/0；optimizer更新：禁止。

历史L9保持FAIL；L9R修复、L9S4096步链路Pilot已记录。L9S无成功导航，不能当作避障能力。

安全下一步：`./researchctl doctor` 后运行 `./researchctl start --task ops-audit`，或在Codex中只读分析已有实验。 新的训练需要用户提供独立预算与协议。

项目目录：`/home/keyousuke/MineUAV-Learning-px4-sitl`。协议：`59964b4d2bba385308e62ff0b9f9d8d0818e53ba767aaf398ec20fa95007a5a8`。

最新checkpoint：本地冻结L9S Pilot；其文件身份由doctor核验，不授权初始化正式模型。训练中断恢复策略RESTART_REQUIRED；未实现EXACT_RESUME或BOUNDARY_RESUME。

早上：`./researchctl status` → `./researchctl recover` → `./researchctl doctor`。

晚上：`./researchctl request-stop` → `./researchctl verify`；若SYNC_PENDING，运行 `./researchctl sync`。

CLI只运行登记的维护任务；AI开发需用户启动Codex。无开机服务或无人值守训练。
<!-- RESEARCH_OPS_V1_END -->

## 下一项研究工作

优先只读审计旧回归中的固定 `/tmp/mineuav-mujoco-lidar-audit` 路径，设计可配置、可复现的测试入口迁移。当前OPS入口已使用持久化依赖；不要为让旧测试通过而放宽扫描器身份检查。若需修改源码或登记新阶段，保留V1协议/attempt并另行版本化，当前V1不提供静默rebaseline。

L9S已在前一阶段完成，不能再次执行其4096步Pilot。当前训练授权NO、预算0、optimizer禁止。原L9保持FAIL。下一次训练预算和任何PX4/设备任务须由用户另行授权。
