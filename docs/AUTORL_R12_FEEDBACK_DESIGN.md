# AUTORL-R12：低KL训练后固定反馈诊断（事前协议）

R11已完整发布bd9095ff2f1b16a1bcd89a8ae699f4df47d1a05a；早期KL降低但四导航标准FAIL，不能继续盲调或重跑。采用有界设计、RED→GREEN及独立只读审查。这个阶段仅推理，不进行optimizer、训练、rollout、物理推进、新射线扫描、Validation或Test。

## 唯一问题与新证据

检查R10/R11新32k/final冻结模型的控制响应，判断小的单次近似KL是否仍伴随累计的方向/减速反馈偏离。只读4个新checkpoint，每个使用R4已固定的62个反事实输入；不是重复R4旧11模型实验。R4已完成的R3 PRIOR step0输出作历史基准，不再次加载推理；复用原真实背景NPZ，源时间戳不重写，不冒充新观测。

输入依旧LiDAR[3,12,24]+state7+heading2+age1；只替换明确标记OFFLINE_COUNTERFACTUAL_INPUT的state，与R4同一59个不同state向量/62分组。不能将合成state响应当真实运动轨迹。只读推理复用feedback_probe.infer_panel，不重写模型、不改训练入口。

## 身份和权限

固定ID AUTORL_R12_LOW_KL_FEEDBACK_ONLY、独立空目录.research_local/autorl_r12_feedback/run_01，CPU1，guard120s、Mem1536MiB/磁盘2GiB，父进程丢失/stop/身份变化即停保留attempt、不重放。全部grant文件及旧raw/SHA前后不变，不reserve/退款/initialize。允许已审计封存R6 FAIL存在，任何未决/未知状态仍拒绝；不修改旧R4门控，不假装R6变COMPLETE。

固定R11 report8a5c911619a7905172d94674664011e7cc6b3dbbae7db42e3b6706f53b8c7dc3 / summary7f34a1e98eae2ef289becacb4d1086f7e5dbb0d686d2cae750610452d90b8f26；R10 report0847525a… / summaryc9fa8bcc…；R4 report110587bd… / result83d3766e…。每个新checkpoint逐项绑定summary/complete/sidecar完整payload/ZIP SHA/全policy身份，缺失不补造。所有真实独立审查引用核验。原factory/CNN/Reward/PI/动力学/旧checkpoint及上游不改。

## 描述指标与有限的保留判据

与R4同定义：零state的ENU速度指令范数、各X/Y/Z零速度非零目标误差的同向比例（分母6/6/4）；+.1m误差下正速度减负速度的同轴指令差；yaw ±.5的差/1rad；有限panel内raw均值触边比例；相对历史Scripted的最大归一化动作差。不是物理减速度、漂移、真实最小距离或导航成功率。输出raw mean/std/clip、Critic值，Critic无return真值配对不判拟合成功。

预定义PANEL_PRIOR_RETAINED：零state速度指令范数≤.05m/s、panel最大归一化动作差≤.05，且三轴所有固定非零误差同向、三个速度差各自与原先验零差异的绝对误差≤.05m/s、yaw差>0。此判据只评价有限输入中是否保留原反馈，不要求好避障策略处处等同Scripted；偏离本身不证明Bug，保留也不证明导航安全。所有新模型分别PASS/FAIL，描述数据完整展示，不按结果选模型。

事前修订：_01候选误要求三个速度差严格负，而原先验只使用位置/yaw误差、没有速度反馈，旧R4初态三差均0。因此分析器在真实运行前的基准自校验失败，揭示概念Important；原GO遗漏该问题，明确废止其对新版本的使用。_01 source/protocol/完整测试收据另存只读archive，不登记/不执行。_02改为上述与基准一致性的判据，保留带符号差作为描述指标；不是观察新模型结果后放宽标准。重新source/REG/EXEC/完整测试及独立复审后才可运行。原Reward/控制与所有历史结果不变。

## 执行与交付顺序

新固定spec及零优化runner、9纯契约/2数组来源测试先RED→GREEN；其中新增独立已发布raw清单门控及原先验速度判据校准，不能只核checkpoint而把损坏的旧episode日志接受为新基线。完整当前OPS/Tensor/历史选择回归、确切源码/准入与独立GO、新register/recover/doctor、预算/stop/resources/保护检查后唯一运行。源码运行期冻结，PPO.learn/train/Adam.step显式tripwire。新profile不能被BudgetStore当训练协议reserve；无任意模型/脚本/seed/预算CLI覆盖。

4×62=248新输出，原step0只作62条历史参考，不能混成310新推理或新扫描。保存local-only完整result/attempt/complete；MD中文11节、JSON、4真实推理曲线、README/五长期文档、独立结果复审、显式Commit/OPS sync/远端后依据证据选择新PPO实验。R12不进行训练，剩余1410176step/54000s须全字节不变；PX4/FASTLIO/设备/封存Test禁止。
