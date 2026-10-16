# AUTORL-R19：既有Reward信用与任务曝光审计

日期：2026-10-16。阶段实现 **PASS**；算法性能提升 **NOT TESTED**。本轮没有新模型推理、物理推进、episode或优化器更新。正常OPS发布及远端核验另以实际收据为准，未完成全部交付不称阶段完整发布。

## 1. 阶段目标

检验两项具体解释：当前Reward是否更偏好R18观察到的两个失败轨迹；R17增加empty重放后，是否仍主要缺乏empty训练样本。先查现有数据，不盲加训练、不改Reward。

## 2. 技术背景

R18的一个开发Validation场景中，32k策略283步成功，R16/R17 final各375步超时，final从未在25Hz采样进入0.1m目标区。它们末速已低于0.15m/s，不能直接归因“到了以后刹不住”。此前R17以50%概率reset到empty，不等于50%的rollout步数来自empty：碰撞任务更早结束。

## 3. 系统架构

已冻结R18三trace + R16/R17 Train episode_details/summary + 原L6清单 → 标准库Reward恒等式/折扣计算/已结束与尾episode曝光核对 → 新JSON和三图。固定ZIP只读取JSON `data`中的gamma和gae_lambda，不反序列化pickle、PTH或策略。既有OPS版本检查子进程会import软件包，但不加载模型或运行仿真。

## 4. 代码实现

- `reward_credit.py`：固定来源与SHA、8项Reward分解、注册gamma白名单、episode与尾步记账、真实字符串reset身份解释、当前回归准入检查。
- `reward_credit_run.py`：固定入口、独立进程组、flock、父进程与绝对期限guard、资源/stop检查、唯一attempt与原子result/complete，异常留failure。
- `tests/research_ops_v2/test_reward_credit.py`：9项纯算术/身份/缺证据契约。先7项missing-module RED，后真实字符串reset接口另2项RED，全部GREEN。未修改历史环境、Reward、CNN或控制器。

`reset_exposure`只用于已封存、身份已核验的short/original序列，不声称是任意未知任务流的通用验证器。尾episode只加真实步数，没有补造Reward或终止事件。

## 5. 算法原理

原Reward保持：`10Δd − .005||a||² − .5w(d)||v||² +10success −30collision/physical −10sensor −.01每执行step −2timeout`，`w=clip((.5−d)/.4,0,1)`；碰撞/物理/传感器终止step的progress为0，先前进度不回溯删除。

每条实际轨迹逐项校验 `reward=sum(8分项)`，`total`不重复计入。`Gγ=Σγ^(step−1) reward_step`。只有γ=.99是本次读取的真实训练参数；.995/.999/1只是同一记录的反事实算术，不是重新训练结果。成功bonus在283步的保留因子=.99^282=.05876583，原+10只贡献+.5876583。

γ=.99的几何半衰期68.9676步=2.75870s、几何和长度100步=4s；γλ=.9405的半衰期11.2994步=.451977s、几何和长度16.8067步=.672269s。这是衰减量级，不是硬截断或已观测的因果学习范围。真实逐transition advantage没有保存，**NOT OBSERVABLE**；不从奖励累计量推断梯度方向。

## 6. 实验协议与身份

新 `AUTORL_R19_REWARD_CREDIT_ONLY`，CPU1worker、120s上限、所有train/optimizer/inference/physics/new episodes/Test为0。唯一`.research_local/autorl_r19_reward_credit/run_01`，不得重试。三个历史checkpoint对应R17@32768、R16final、R17final；只读来源13文件SHA见prereg。

REG `96e61d65ff78c13e8bedbff7e279ad67d55635cd7945f877ce586795d121a075`；124文件SOURCE `7c363f499ae400a56c96cc87a57b38fb23d6d442bccdaec9539139096f71b6d3`；EXEC `643d79a05381ccfc094d5d3a2f79325ef0113fae08e3452cf805bba1b5e28814`；result `e4b187eec7da23bcae213c92c9b36ec68ed751f7625ec8e19fd769fd7ee9ef1f`。

独立授权账本24事件/12attempt保持；全部299授权文件与7额外历史raw逐字节不变，余额1213568step/43200s保守预留墙钟、stop=false，无预留、退款、重置。旧OPS零维护预算不撤销后续独立授权，但不授权绕过下一实验准入。历史模型、L9永久FAIL、87基线文件保护核验通过。

## 7. 实际结果

| 既有轨迹 | γ=.99真实 | γ=.995仅算术 | γ=.999仅算术 | 未折扣 |
| --- | ---: | ---: | ---: | ---: |
| 32k成功，283步 | 8.059904 | 11.968044 | 19.037651 | 22.019979 |
| R16final超时，375步 | 6.500101 | 9.098223 | 11.505582 | 11.982312 |
| R17final超时，375步 | 5.725403 | 8.623735 | 11.919330 | 12.816864 |

| 原距离Train阶段 | R16 | R17 |
| --- | ---: | ---: |
| 实际reset数/empty reset | 223/51 | 144/77 |
| empty reset占比 | 22.87% | 53.47% |
| 结束episode/未结束尾步 | 222/20 | 143/238 |
| 全部真实policy steps（含尾） | 32686 | 32686 |
| empty真实policy steps（含尾） | 18166 | 27282 |
| empty采样步占比 | 55.58% | 83.47% |
| empty结束成功/超时 | 17/34 | 19/57 |

每轮闭合 `65536−32850=32686`。R17尾238步属于empty，只计曝光；R16尾20步属side_pair。reset占比与结束episode占比分开，未将场景曝光当作独立同分布梯度贡献。

1135选定回归PASS：187 OPS（含9新增）+63 Tensor +885历史，15项明确排除，测试优化器0，不是全仓测试。当前checks SHA `ee0890fdf6e810b158b7393e234cdf698953c6e1e7cf2f7b20b1b53394f76352`。固定独立预审C0/I0/新Minor0，真实结果复审另见JSON authentic receipt。

唯一运行exit0、无超时、进程组已消失。内层算术计时.213469s/CPU.213458s，launcher12.439168s含OPS探针；worker HWM63.4727MiB，整个回收子进程树HWM237.6445MiB，guard采样minMem7864.4063MiB。后者不是imports/hash全程连续最低，连续值 **NOT AVAILABLE**。未观察OOM。既有optional cv2/NumPy ABI警告及Matplotlib3D warning保持披露，不改依赖，2D图正常。

实际派生图：[折扣回报](../mujoco/reports/autonomous_rl_reward_credit_r19/figures/discounted_returns.png)、[Reward分项](../mujoco/reports/autonomous_rl_reward_credit_r19/figures/reward_components.png)、[reset与采样曝光](../mujoco/reports/autonomous_rl_reward_credit_r19/figures/sample_exposure.png)。没有用反事实算术绘制新训练性能。

## 8. 问题分析

**CONFIRMED**：三条轨迹上成功的实际折扣回报更高，故“Reward偏好这两个观察到的失败而非成功”的预注册解释被反驳。成功bonus到达时已衰减至5.88%，但不等于PPO没收到信号。R17大部分实际原域采样已是empty；简单称“empty数据太少”不完整。

**SUSPECTED**：末端较晚的反馈时间信用可能影响到达与停稳；少量困难任务的负反馈也可能影响共享参数。两者都未由本审计证明因果。

**UNVERIFIED**：真实advantages、逐状态优化器梯度归因、泛化稳定性。R18的发布流程Minor与R7历史范围Minor分别保留，不因本轮新Minor0抹去；R18原空白检查exit2仍是历史事实。

## 9. 局限性

仅一个开发Validation场景的三条轨迹、两轮Train日志。反事实γ下没有新动作分布、Value拟合或PPO训练结果。γλ核不是硬时间窗口，collision惩罚总量不是梯度大小。本轮不提供算法性能提升、避障能力或实机安全结论。Test始终未运行。

## 10. 下一阶段

优先设计R20：相对R17只将γ=.99改为.995，保持GAE λ、Reward、中心化Actor、初态std.1、CNN、LR、场景/RNG抽样规则、65,536步预算及所有动力学/频率不变，新seed0独立初始化、不从旧ZIP继续；训练前新profile、无更新测试、精确GO、当前预算/stop/资源准入。预注册原距离Val能力与short能力保留，Final Checkpoint，不选择最好中间模型。

其理由是83.47%的实际empty曝光已较高，而γ改变在相同轨迹上增加晚期反馈相对权重；这只是最值得反驳的候选，不是必定有效。备选仅原域empty的极端隔离用于排除困难任务负迁移，不先运行，也不与γ同时修改。完整发布后依据真实余额继续，不自动重试旧实验。

## 11. 给项目负责人的总结

这阶段没有再训练，而是把已有成功和失败轨迹的奖励账算清楚。成功轨迹并没有比失败“更吃亏”，但到达目标奖励发生得晚，折扣后只剩一小部分；同时空场景其实已经占了多数训练步。下一轮只检验让算法更重视稍远未来是否有帮助，不改奖励、不加模型、不盲目延长训练。目前仍不能说已学会稳定避障。已有独立授权余额足以考虑一次65,536步新实验，不需要扩大权限；仍需完成本阶段审查与推送，再过新实验安全门。
