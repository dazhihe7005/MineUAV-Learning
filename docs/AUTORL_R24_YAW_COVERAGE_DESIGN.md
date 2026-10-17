# AUTORL-R24：Train起始Yaw覆盖单变量实验设计与计划

## 目的与控制组

R23已发布c19e3b2d3ada3992c4b18578b02ff52f7ea7210c：empty开发验证初始Yaw离同layout Train最近约93.6/77.7度；R22 Train有成功但Val仍失败。该差异是可反驳候选，不是已证实根因。新实验相对完整R17（不是R22）仅改变Train起始Yaw分布，γ=.99、λ=.95；R22的λ=.99不延续，不用其checkpoint。

## 冻结干预

每个原12Train几何/start/target保留原Yaw，并加入预定四方位0、+π/2、π、−π/2；每次reset在五者均匀抽取。方位取自通用完整圆周网格，不复制Validation/Test的角度。short与original各12×5=60新任务身份，共120；原L6 manifest及short manifest不改，额外Yaw manifest独立冻结SHA。新任务记录base身份、原distance任务身份、stage、实际起始Yaw/选项、目标、几何SHA。

base任务抽样仍是short均匀12、32768后的下个自然reset才切original，三empty各1/6、九非empty各1/18。Yaw使用单独`random.Random`实例、固定seed派生域，不消耗base NumPy调度RNG或Torch采样RNG。因episode长度会变，不承诺实际访问前缀/32k权重与R17相同。只允许reset设定初始Yaw，正常step必须实际mj_step，不能中途写qpos回放。

## 不变量与预算

新实验ARL_R24_TRAIN_YAW_COVERAGE_SEED0，fresh seed0中心化手写先验/std.1、同CNN/Actor-Critic170857参数、LR3e-5、Reward NAV_V2_COLLISION_PREFLIGHT_V1；PPO n_steps128/batch64/epochs2/γ.99/λ.95/clip.2，其余与R17相同。保持原7kg物理、PI、500/100/25/10Hz、24000rays。模型输入和动作含义不变，不向网络提供障碍或Yaw课程元数据。

拟定唯一65536实际environment步/3600秒全额预留/CPU1，按现有独立grant扣账（本设计不是新授权）。当前余额1082496/36000，预留后应1016960/32400；停止锁存/旧文件/源码/资源/身份/审查任一失败不启动。不运行新Pilot；未知中断不重试，不从旧ZIP续训。R20/L9永久FAIL。

## 预注册评价与停止

Final Checkpoint，step0/32768/65536三个checkpoint；关闭Train后各跑6short+6original开发Validation，共36新Val，Test0。保持原五项门槛：32k short至少4成功且最多1碰撞；final short成功不低于自身初态且碰撞不增；final original成功多于自身初态且碰撞不增；final original成功多于R17 final0且碰撞≤3；两个empty Val final成功数不少于自身32k且32k至少1。各项physical/sensor failure0。缺证NOT TESTED，完整观察任一不满足科学FAIL；不改0.1m/0.15m/s/5样本成功条件。

## 实现计划（自主研发授权下原位执行）

- [ ] 数据契约：新增yaw_coverage.py，纯函数生成/验证120任务、固定lookup与实际模型/rolloutγλ核验；先写5项标准库RED，再最小实现GREEN。保持旧文件不变。
- [ ] 新scheduler：复用原MultiScenarioTrainingEnv的开场景/关闭/真实step，只新增reset级Yaw选择、独立RNG和任务记账。每步核对目标、几何、任务与实际起始Yaw记录；失败锁存、只有一个resident场景。先真实MuJoCo无更新RED，再验证全部120任务有限reset+一步、缓存/PI清零、正确heading/点云/step频率与Train隔离。
- [ ] 固定schema24：profile/CLI/原预算主管与worker分派，不复制PPO；γλ与权重初态必须真实核验，旧schema17的prefix要求保持只对17生效。实际worker在learn方法入口拦截，确认正确新scheduler/初态/temporal配置后立即抛出，无sampling/optimizer；测试必须计数0。
- [ ] 冻结REG/SOURCE/EXEC/manifest和全部旧grant文件，运行当前源码绑定的OPS/Tensor/历史选择回归；独立只读预审C/I为0后正常register/recover/doctor/audit和唯一训练。已有输出/attempt拒绝覆盖。
- [ ] 完整训练、独立Val、实际physics计数、120任务访问/基类访问/实际Yaw/控制RNG等证据，五图/中文阶段MD/JSON/长期文档；独立结果复审后显式白名单Commit/正常OPS sync核远端；再查余额/stop/资源继续下一安全任务。

## 审查重点与限制

不能用原base身份冒充已经改变Yaw的任务身份；checkpoint的L6 metadata只描述不变的base几何，新增Yaw manifest在执行协议、scheduler state、episode日志同时绑定。旧CheckpointManager仍RESTART_REQUIRED，不声明精确恢复。不能让Yaw额外抽样改变base RNG，不能把120初始化测试当120成功导航episode。物理终止可能提前结束一个policy步，实际ticks不是65536×20假定。Val只用于本研究，不运行封存Test；单seed与小Val不证明矿井泛化或真实无人机部署。

## 2026-10-17 实际执行补充（不修改上述预注册设计）

数据契约、新scheduler、固定schema24、146源码冻结与1181选择回归/独立预审/正常登记均已完成。唯一训练因系统重启中断，未完成最后一项完整训练/36Val验收：393完整更新至50304步，精确退出步数未知，科学NOT TESTED。原计划不回填为已完成；只追加第30FAIL，不重试或ZIP续训。启动前分析器同序任务绑定Important曾真实RED→GREEN修复，旧证据保留。重启后旧GAE正例耦合变化真实账本导致2ERROR，明确披露，不修改运行源码或门禁掩盖；详见本阶段中断报告。
