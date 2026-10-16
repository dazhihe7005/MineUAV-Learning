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
