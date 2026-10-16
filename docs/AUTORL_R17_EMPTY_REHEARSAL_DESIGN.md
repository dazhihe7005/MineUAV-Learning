# AUTORL-R17：原距离空场景重放比例对照

## 目标与证据

R16已完整发布c8307c0f670787bb2a4897e14034452cbd57588f，不能重跑。36新Validation中原任务初态1成功/3碰撞/2超时，32k为2/3/1，final为0/3/3；两个empty在32k成功、final超时。短Train停稳经验明显增加仍未转化为最终原任务成功。假设：后半程混合难任务的分布可能削弱基础到达停稳；这不是已确认唯一根因。

## 单变量与实现边界

相对R16只改变后半程原距离Train的reset抽样分布。12个冻结原Train含3 empty、9 nonempty；原均匀empty概率3/12=.25，新概率.5：每个empty为1/6，每个nonempty为1/18。前32768步保持原12个short均匀抽样，不增加RNG调用；跨阈值不改当前episode目标，只在下一次自然reset使用新权重。权重作用于场景调度，不能进入策略观测。

复用DistanceTaskAccountingAdapter的身份、几何、target、前后step检查和失效锁存。新的子类只重写reset选择，实际初始化/动力学/PI/射线/编码仍由旧路径执行。覆盖测试显式base Train identity不消耗抽样RNG，不能提供Validation/Test身份。只常驻一个MuJoCo环境。

同R16新seed0初态中心化Actor、std.1可训练、LR3e-5、CNN/Critic、Reward NAV_V2_COLLISION_PREFLIGHT_V1、PPO128/64/2及其他超参数、动作尺度、500/100/25/10Hz、24000rays、7kg物理和原/short清单不变。禁止加载任何旧ZIP初始化。原始状态归零Actor源码不修改。

## 可完成性与公平性

只读复用既有baseline.json（SHA5f482d0711652f14c8d925a8d4d6f24f467686f1a10eb2787c65c66048c4c70c）的三个Train empty Scripted成功及R16自身32k两个empty Validation成功；不新运行旧基线、不把Scripted当LiDAR策略。运行前核源码、模型和原始证据身份。

32k policy SHA预先固定为R16的33bf6699158a35842fc05c3d4ca97f391f778db60c759fdad2564a6e94c87f60，用于前半程相同对照的工程验证。若不同，停止并保留独立attempt，不重试、不掩盖。最终始终FinalCheckpoint，不能挑32k。

## 冻结执行与判据

新ID ARL_R17_EMPTY_REHEARSAL_SEED0，schema17，65536实际step/3600s，CPU1worker，seed0，RESTART_REQUIRED。三checkpoint0/32768/65536各新6原+6short Validation，共36，Test0/历史评价复用0。最终五项科学判据：

1. short candidate成功>=4/6、collision<=1、physical/sensor=0。
2. short保留：final成功>=own初态、collision<=own初态、physical/sensor=0。
3. 原任务学习：final成功>own初态、collision<=own初态、physical/sensor=0。
4. 相对R16 final提升：原final成功>0、collision<=3、physical/sensor=0。
5. empty保持：原域empty final成功>=own32k成功，且own32k至少1成功；不得以没有成功的空分母虚假通过。

科学总结果为五项合取；阶段工程完整不等于学会避障。另报告实际empty抽样次数与概率（小样本非保证）、按布局结果、goal_stop、PPO诊断、资源和所有失败。不改变成功条件：距离<.10m、速度<.15m/s、持续5策略step。无停机/资源/重要审查问题且旧预算足够才能预留，训练前实际余额1279104step/46800s；预计预留后1213568/43200，不重置/退款。

## 实施与测试顺序

1. RED：新profile拒绝越权/变参数/缺少R16证据；真实scheduler测试前半程RNG/trajectory与R16一致、后半程权重可重复/覆盖12场景、自然阈值/缓存积分归零、错身份锁存。
2. GREEN：新增empty_rehearsal_profile.py、empty_rehearsal.py及测试；budget/runtime/CLI/worker仅显式schema17路径，绝不任意shell或参数入口。
3. 当前源码完整无optimizer OPS/Tensor/历史回归；新分析判据RED→GREEN；新登记/独立精确GO→recover/doctor/历史/余额/stop/资源→唯一prepare/start。
4. 主管完整结束后分析原始记录、生成图/JSON/11节中文MD，独立结果复审、Git Commit/OPS sync/远端SHA；再决定下一安全研究。科学FAIL不触发同attempt重试。

禁止PX4/FAST-LIO2/硬件/Test调参；保护历史模型、原始数据和旧授权协议。普通研发设计执行由用户长期委托，不新增权限。
