# AUTORL-R20：单变量gamma时间信用对照

R19已完整提交并正常OPS sync，远端545c6c9a321956e1b89314f1fa8a63583e475cf1。成功轨迹实际折扣回报仍最高，不能认定Reward鼓励失败；R17原域empty已占83.47%采样步，继续小幅加reset权重缺乏强依据。成功bonus第283步在gamma=.99时只留5.8766%，提示一个可反驳候选：更长期的回报权重是否有助于保留到达与停稳。

## 冻结对照

新ARL_R20_TEMPORAL_CREDIT_SEED0，相对R17仅gamma .99→.995。保留lambda=.95、n_steps128、batch64、epochs2、LR3e-5、clip.2、entropy0、value.5、gradclip.5；中心化Actor/CNN170857参数、std.1可训练、解析hand prior与seed0独立初始化不变，不从任何旧ZIP加载初始化。Reward NAV_V2_COLLISION_PREFLIGHT_V1、动作、7kg物理、500/100/25/10Hz和24000rays保持。前32768short均匀，之后自然reset才切原距离，empty概率.5，抽样/RNG规则与R17相同。

本轮不修改R13中心化policy源码；固定factory创建后在任何采样前同步设置model.gamma和rollout_buffer.gamma，并验证lambda及其他超参数未变。零optimizer测试检查参数/Action/Value不变、实际SB3 return递推使用新值、保存/重载保留gamma、实际scheduler/控制器/传感器不变。gamma在首轮即生效，因此32k参数不应再强制等于R17，不能保留错误prefix期望。旧schema17的前缀检查继续保留。

## 预算与评价

唯一65536实际训练step/3600s最大保守墙钟预留，CPU1worker、seed0，全部真实optimizer在原独立grant计费。现在余额1213568/43200，实际准入时重核；预留后应1148032/39600，不退款/重置/新授权。训练失败保留attempt，RESTART_REQUIRED，不自动重试。

0/32768/65536各6short+6original全新独立Validation，总36、Test0，Train关闭后顺序评价；不污染训练RNG。Final Checkpoint，不按中间Val挑最好模型。沿用R17五个合取判据：32k short候选成功>=4且碰撞<=1、无物理/传感器失败；final short成功不少于ownstep0且碰撞不增；原域final成功多于ownstep0且碰撞不增；相对R17final原域成功>0、碰撞<=3；原域empty final成功>=own32k且own32k>=1。失败就如实FAIL，不降低阈值，Train成功率不作为公平泛化指标。

记录原始episode、Reward/目标最小误差/末误差/近目标联合样本、PPO KL/Value/entropy、实际gamma/lambda、checkpoint/Adam/预测重载、场景身份、资源和budget。γ=.995的反事实收益不保证新训练有效；只有实际新Val能检验。GAE核不当作硬信用窗口，单seed/小Val不证明矿井泛化。

预审披露的文案Minor：继承的`evaluation.rehearsal_benefit`字符串保留旧R17协议中“R16 final0/3”的历史字样，数值阈值仍是成功>0、碰撞<=3；R20实际对照是新`control_report_sha256/control_summary_sha256`绑定的R17（其final同样0/3），本设计与已冻结分析的正式判据`temporal_credit_benefit_over_R17`明确只核R17。不把旧字符串误写成运行了新R16对照，不在冻结回归期间改SOURCE或悄悄换协议；新Minor保留在审查及结果中，数值与比较身份没有二义的执行分支。

## 执行顺序与保护

设计→纯标准库RED→GREEN→真实Tensor/GAE/保存重载无更新测试→固定profile/REG/SOURCE/EXEC→全部选择回归→冻结实际分析方法→独立精确GO→新OPS register/recover/doctor/真实预算stop/历史/锁/资源→新prepare精确SHA→唯一受监督训练。运行期间SOURCE与协议冻结。所有旧授权文件（head只允许合法追加结果）、原模型/七checkpoint/七历史raw/R18/R19证据保持；PX4/FASTLIO/实机/封存Test禁止。正常三图以上/中文MDJSON/独立结果复审/完整检查exit0后单独Commit/正常OPS sync核远端；普通FAIL后根据真实证据与预算继续新安全研究，不盲重试。
