# AUTORL-R10：解析先验×短目标课程（冻结设计）

## 假设和对照

R9已发布0bdcfa999ef75a0b4f3d34a6498d76267348b7d3，12短Train非学习先验成功，R8随机初态短课程仍0成功。假设：先验在短目标可提供较早的成功反馈，可能比原任务直接训练更能保留控制规律，并改善原目标任务。此假设可被失败数据反驳，不保证训练成功。

R3已有原目标随机/解析两臂，R8已有距离课程随机，R10补上距离课程解析。相对R8唯一科学变量为初始Actor均值，std.3仍可训练；新seed0随机网络后复用已审解析赋值，不加载R9/R8/L9或任何checkpoint初始化。CNN/Critic随机初态同既有seed0。非学习先验能力不是学习收益；历史三臂不重跑。

## 固定执行协议

新schema10，ARL_R10_PRIOR_DISTANCE_SEED0，65536实际环境步/3600s完整保守wall预留，CPU1训练worker。原grant余额1541248step/61200s，不初始化/退款；唯一预留后应余1475712/57600。旧ID永不重试，RESTART_REQUIRED，无Test。低于Mem1536MiB/磁盘2GiB、停止、源变化、身份/未知终止/非有限训练数据或梯度异常立即主管停止，保留证据。合法有限物理失败仍沿原导航/PPO契约统计，不引入R9只读可完成性模式的额外门控。

同R8：12short .20m任务至32768，下一自然reset换原12目标，旧qpos/qvel/积分/LiDAR缓存reset；不变几何/起点/Yaw/质量/惯量/kf/km/PI/500-100-25-10Hz/24krays/编码/CNN/Reward NAV_V2_COLLISION_PREFLIGHT_V1/PPO128-64-2/lr3e-4/gamma.99/GAE.95/clip.2/ent0/vf.5/maxgrad.5。初态policy6312e99c87fd5a4dca066bd1ef64acd17d5ee44b563c152c59c133ee659d7f74。

checkpoint0/32768/65536，完整rollout且对应优化后保存。Train关闭后独立short6×3、original6×2共30新Validation；原step0只读复用R3 PRIOR/std.3同完整policy身份的6条原始评价，严格pin afbeac0d092aa4f38c318f4b75909006db65f2f8eaa4435783ae6b7a887ada4e，不能沿用R8 RANDOM的step0。Final Checkpoint，不按Validation择模，不读取Test。原停稳distance<.10/speed<.15连续5步、碰撞优先不变。

## 事前分开验收（不事后改阈值）

1. 短任务候选：final至少4/6成功、至多1/6collision、physical/sensor=0。
2. 短任务先验保留：final成功数>=自身short step0，collision<=自身short step0，physical/sensor=0。初态若6/6，则不能虚构严格提升。
3. 初始化收益：original final成功数>历史R8 final0，collision<=历史R8 final2；不是同seed泛化证明。
4. 原任务学习增益：original final成功数>自身original step0（历史1/6），collision<=自身step0（历史3/6），physical/sensor=0。手写初态成功不能算学习。

上述每项均要求相关final评价physical/sensor=0，每项独立PASS/FAIL。工程完整执行不等于科学成功。统计训练分期成功/碰撞/超时/物理/传感器、目标最小/最终误差、Reward分项、动作/轨迹、所有PPO更新、资源。新增仅诊断的near-goal速度/连续joint计数聚合，不进入观测/Reward/控制，不改变旧schema输出；不能从聚合数据唯一归因PPO内部机制。

## 实现与准入清单

- [ ] 新prior_distance_profile.py：只允许固定schema10/ID/budget/旧R8与R9发布proof/current admission；budget.py仅增加严格dispatch，不改grant。
- [ ] 新prior_distance_training.py：复用R8scheduler及std_training(PRIOR)初始化，复用原ShortValidationEvaluator；独立PRIOR原step0严格复用与纯诊断计数。
- [ ] 原worker只增加schema10选择和评价/diagnostic分支，不复制PPO；新CLI prepare-prior-distance/start-prior-distance，无checkpoint/seed/脚本/预算覆盖参数，严格OPS阶段绑定。
- [ ] 测试先RED→GREEN：固定身份/不可改预算/缺proof拒绝/CLI限制、真实同R7记账链新先验初始化+第一步+checkpoint状态、PRIOR/RANDOMstep0不能混用、诊断计数与合法物理分类不改变。
- [ ] 全选择回归/源码绑定/独立只读GO/新register/recover/doctor/历史保护与stop/resource/budget检查后，精确prepareSHA并唯一执行。运行期源码冻结。
- [ ] 真实完整主管COMPLETE才分析，图表/中文11节MD/JSON/六长期文档/README、独立结果复审、显式Commit/OPS sync/远端SHA。若FAIL保留，不重跑本attempt，依据证据和余额继续安全研究。

## 风险与不做的事项

原先验在R3原任务训练退化，R10也可能失败；不因此偷偷减noise/改Reward。原场景与Validation规模小/单seed，不证明LiDAR依赖或真实避障。R7任意active_task结构损坏Minor保留；R9异常context节点截断Minor仅其独立非学习runner适用，不冒充R10完整故障管理修复。PX4/FASTLIO/实机/封存Test和大规模正式百万步均不在本阶段执行。
